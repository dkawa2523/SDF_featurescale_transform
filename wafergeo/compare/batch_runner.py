from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

from wafergeo.compare.feature_outputs import write_target_feature_outputs
from wafergeo.compare.loader import is_label_input_kind
from wafergeo.compare.metric_defs import METRIC_DEFINITIONS
from wafergeo.compare.output_artifacts import (
    write_metric_summary_csv,
    write_per_material_sdf_csv,
    write_ranking_top_png,
)
from wafergeo.compare.output_cleanup import clean_batch_compare_output_dir
from wafergeo.compare.runner import (
    PreparedTarget,
    build_input_summary_payload,
    prepare_label_target_for_compare,
    run_compare_spec,
)
from wafergeo.compare.runtime_io import resolve_path, write_json, write_run_info
from wafergeo.compare.schema import (
    BatchCompareSpec,
    CompareSpec,
    SimulationInputSpec,
    TargetInputSpec,
    load_batch_compare_spec_yaml,
)
from wafergeo.compare.scoring import objective_csv_row


def _safe_case_id(value: str, *, row_number: int) -> str:
    raw = value.strip() or f"case_{row_number:04d}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    safe = safe.strip("._-")
    if not safe:
        safe = f"case_{row_number:04d}"
    if safe in {".", ".."}:
        raise ValueError(f"invalid case_id at row {row_number}: {value!r}")
    return safe


def _read_batch_index(index_path: Path) -> list[dict[str, str]]:
    required = {"case_id", "simulation_kind", "simulation_path", "target_path"}
    rows: list[dict[str, str]] = []
    with index_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(f"batch compare index is missing required columns: {missing}")
        seen_case_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=1):
            cleaned = {str(k): str(v or "").strip() for k, v in row.items() if k is not None}
            case_id = _safe_case_id(cleaned.get("case_id", ""), row_number=row_number)
            if case_id in seen_case_ids:
                raise ValueError(f"duplicate case_id in batch compare index: {case_id}")
            seen_case_ids.add(case_id)
            cleaned["case_id"] = case_id
            for key in required:
                if not cleaned.get(key):
                    raise ValueError(f"batch compare index row {row_number} has empty {key}")
            rows.append(cleaned)
    if not rows:
        raise ValueError("batch compare index has no rows")
    return rows


def read_compare_index(index_path: Path) -> list[dict[str, str]]:
    return _read_batch_index(index_path)


def _compare_spec_from_batch_row(
    row: dict[str, str],
    *,
    batch_spec: BatchCompareSpec,
    output_dir: Path,
) -> CompareSpec:
    target_kind = row.get("target_kind") or "contour_json"
    target_units = row.get("target_units") or "nm"
    return CompareSpec(
        task="compare",
        simulation=SimulationInputSpec(
            kind=row["simulation_kind"],  # type: ignore[arg-type]
            path=row["simulation_path"],
            void_id=None if not row.get("simulation_void_id") else int(row["simulation_void_id"]),
        ),
        target=TargetInputSpec(
            kind=target_kind,  # type: ignore[arg-type]
            path=row["target_path"],
            units=target_units,
            void_id=None if not row.get("target_void_id") else int(row["target_void_id"]),
        ),
        view=batch_spec.view,
        features=batch_spec.features,
        metrics=batch_spec.metrics,
        output=type(batch_spec.output)(
            dir=str(output_dir),
            difference_image=batch_spec.output.difference_images,
            difference_images=batch_spec.output.difference_images,
            ranking=batch_spec.output.ranking,
        ),
    )


def compare_spec_from_index_row(
    row: dict[str, str],
    *,
    batch_spec: BatchCompareSpec,
    output_dir: Path,
) -> CompareSpec:
    return _compare_spec_from_batch_row(row, batch_spec=batch_spec, output_dir=output_dir)


def _write_batch_ranking(output_dir: Path, ranking_rows: list[dict[str, str | float]]) -> None:
    with (output_dir / "ranking.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "case_id", "normalized_total_score", "total_score"],
        )
        writer.writeheader()
        for rank, ranking_row in enumerate(ranking_rows, start=1):
            writer.writerow({"rank": rank, **ranking_row})


def _write_batch_metrics(output_dir: Path, metric_rows: list[dict[str, object]]) -> None:
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "name",
                "loss",
                "value",
                "weight",
                "loss_scale",
                "normalized_loss",
                "status",
            ],
        )
        writer.writeheader()
        for metric_row in metric_rows:
            writer.writerow(metric_row)


def _write_batch_difference_summary(
    output_dir: Path,
    difference_rows: list[dict[str, object]],
) -> None:
    if not difference_rows:
        return
    fieldnames = [
        "case_id",
        "mode",
        "height",
        "width",
        "background_pixels",
        "match_pixels",
        "mismatch_pixels",
        "simulation_only_pixels",
        "target_only_pixels",
        "changed_pixels",
    ]
    with (output_dir / "difference_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for difference_row in difference_rows:
            writer.writerow({key: difference_row.get(key, "") for key in fieldnames})


def _write_batch_objectives(output_dir: Path, objective_rows: list[dict[str, object]]) -> None:
    with (output_dir / "objectives.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "status",
                "objective",
                "objective_name",
                "direction",
                "total_score",
                "skipped_metrics",
            ],
        )
        writer.writeheader()
        for row in objective_rows:
            writer.writerow(row)


def _metric_scales(metric_names: tuple[str, ...]) -> dict[str, float]:
    return {name: float(METRIC_DEFINITIONS[name].loss_scale) for name in metric_names}


def run_batch_compare_from_config(config_path: str | Path) -> dict[str, object]:
    config_file = Path(config_path).resolve()
    spec = load_batch_compare_spec_yaml(config_file)
    base_dir = config_file.parent
    index_path = resolve_path(spec.index, base_dir=base_dir)
    index_dir = index_path.parent
    out_dir = resolve_path(spec.output.dir, base_dir=base_dir)
    cases_dir = out_dir / "cases"
    differences_dir = out_dir / "differences"
    shared_targets_dir = out_dir / "shared_targets"
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_batch_compare_output_dir(out_dir)
    cases_dir.mkdir(parents=True, exist_ok=True)
    if spec.output.difference_images:
        differences_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_batch_index(index_path)
    metric_rows: list[dict[str, object]] = []
    per_material_rows: list[dict[str, object]] = []
    material_confusion_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, str | float]] = []
    objective_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    target_cache: dict[tuple[object, ...], PreparedTarget] = {}
    target_cache_paths: dict[tuple[object, ...], str] = {}
    target_cache_hits = 0

    for row in rows:
        case_id = row["case_id"]
        case_out = cases_dir / case_id
        case_row = dict(row)
        case_row["simulation_path"] = str(
            resolve_path(case_row["simulation_path"], base_dir=index_dir)
        )
        case_row["target_path"] = str(resolve_path(case_row["target_path"], base_dir=index_dir))
        compare_spec = _compare_spec_from_batch_row(
            case_row,
            batch_spec=spec,
            output_dir=case_out,
        )
        prepared_target: PreparedTarget | None = None
        if is_label_input_kind(compare_spec.target.kind):
            cache_key: tuple[object, ...] = (
                compare_spec.target.kind,
                case_row["target_path"],
                compare_spec.target.void_id,
                compare_spec.view.axes,
                compare_spec.view.depth_axis,
            )
            prepared_target = target_cache.get(cache_key)
            if prepared_target is None:
                prepared_target = prepare_label_target_for_compare(
                    compare_spec,
                    Path(case_row["target_path"]),
                )
                target_cache[cache_key] = prepared_target
                target_dir = shared_targets_dir / f"target_{len(target_cache):04d}"
                write_target_feature_outputs(
                    target_feature=prepared_target.feature,
                    feature_names=spec.features.use,
                    output_dir=target_dir / "features",
                )
                write_json(
                    target_dir / "target_label_summary.json",
                    build_input_summary_payload(
                        feature=prepared_target.feature,
                        label=prepared_target.label,
                    ),
                )
                write_json(
                    target_dir / "target_info.json",
                    {
                        "kind": compare_spec.target.kind,
                        "path": case_row["target_path"],
                        "void_id": compare_spec.target.void_id,
                        "view": {
                            "axes": list(compare_spec.view.axes),
                            "depth_axis": compare_spec.view.depth_axis,
                        },
                    },
                )
                target_cache_paths[cache_key] = str(target_dir.relative_to(out_dir))
            else:
                target_cache_hits += 1
        score, _summary = run_compare_spec(
            spec=compare_spec,
            config_file=config_file,
            base_dir=index_dir,
            output_dir=case_out,
            write_run_metadata=False,
            prepared_target=prepared_target,
            write_target_features=prepared_target is None,
        )
        objective_rows.append(objective_csv_row(case_id, score))
        for metric in score.metrics:
            metric_rows.append({"case_id": case_id, **metric.to_dict()})
        for detail in score.metric_details:
            if detail.get("metric") != "sdf_material":
                continue
            per_material = detail.get("per_material", [])
            if not isinstance(per_material, list):
                continue
            for material_row in per_material:
                if isinstance(material_row, dict):
                    per_material_rows.append({"case_id": case_id, **material_row})
        ranking_rows.append(
            {
                "case_id": case_id,
                "normalized_total_score": score.normalized_total_score,
                "total_score": score.total_score,
            }
        )

        diff_summary_path = case_out / "difference_summary.json"
        if diff_summary_path.exists():
            diff_summary = json.loads(diff_summary_path.read_text(encoding="utf-8"))
            if isinstance(diff_summary, dict):
                difference_rows.append({"case_id": case_id, **diff_summary})
        diff_src = case_out / "difference.png"
        if spec.output.difference_images and diff_src.exists():
            shutil.copyfile(diff_src, differences_dir / f"{case_id}.png")
        confusion_path = case_out / "material_confusion.csv"
        if confusion_path.exists():
            with confusion_path.open("r", encoding="utf-8", newline="") as f:
                for confusion_row in csv.DictReader(f):
                    material_confusion_rows.append({"case_id": case_id, **confusion_row})

    ranking_rows.sort(key=lambda item: float(item["normalized_total_score"]))
    if spec.output.ranking:
        _write_batch_ranking(out_dir, ranking_rows)
        write_ranking_top_png(out_dir / "ranking_top.png", ranking_rows)
    _write_batch_objectives(out_dir, objective_rows)
    _write_batch_metrics(out_dir, metric_rows)
    write_metric_summary_csv(out_dir / "metric_summary.csv", metric_rows)
    if per_material_rows:
        write_per_material_sdf_csv(
            out_dir / "per_material_sdf.csv",
            [{"metric": "sdf_material", "per_material": per_material_rows}],
        )
    if material_confusion_rows:
        with (out_dir / "material_confusion.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(material_confusion_rows[0]))
            writer.writeheader()
            writer.writerows(material_confusion_rows)
    _write_batch_difference_summary(out_dir, difference_rows)

    score_summary: dict[str, object] = {
        "task": "batch-compare",
        "status": "OK",
        "case_count": len(rows),
        "best_case_id": ranking_rows[0]["case_id"],
        "best_total_score": ranking_rows[0]["total_score"],
        "best_normalized_total_score": ranking_rows[0]["normalized_total_score"],
        "metric_scales": _metric_scales(spec.metrics.use),
        "target_cache": {
            "entries": len(target_cache),
            "hits": target_cache_hits,
            "shared_targets": sorted(set(target_cache_paths.values())),
        },
    }
    write_json(out_dir / "score_summary.json", score_summary)
    write_run_info(
        config_path=config_file,
        output_dir=out_dir,
        task="batch-compare",
        inputs={"index": str(index_path)},
    )
    return score_summary
