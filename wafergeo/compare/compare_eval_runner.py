from __future__ import annotations

import csv
import re
import shutil
import time
from pathlib import Path

import numpy as np

from wafergeo.compare.batch_runner import compare_spec_from_index_row, read_compare_index
from wafergeo.compare.compare_eval_figures import write_compare_eval_figures
from wafergeo.compare.loader import is_label_input_kind
from wafergeo.compare.runner import (
    PreparedTarget,
    prepare_label_target_for_compare,
    run_compare_spec,
)
from wafergeo.compare.runtime_io import resolve_path, write_json, write_run_info
from wafergeo.compare.schema import (
    BatchCompareSpec,
    CompareEvalMetricSetSpec,
    CompareSpec,
    OutputSpec,
    ViewSpec,
    load_compare_eval_spec_yaml,
)


def _safe_metric_set_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if not safe or safe in {".", ".."}:
        raise ValueError(f"invalid compare-eval metric_set name: {value!r}")
    return safe


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _clean_eval_output_dirs(out_dir: Path) -> None:
    for name in ("metric_sets", "figures"):
        path = out_dir / name
        if path.exists():
            shutil.rmtree(path)


def _metric_set_batch_spec(
    *,
    index: str,
    metric_set: CompareEvalMetricSetSpec,
    output_dir: Path,
    view: ViewSpec,
) -> BatchCompareSpec:
    return BatchCompareSpec(
        task="batch-compare",
        index=index,
        view=view,
        features=metric_set.features,
        metrics=metric_set.metrics,
        output=OutputSpec(
            dir=str(output_dir),
            difference_image=False,
            difference_images=False,
            ranking=False,
        ),
    )


def _target_cache_key(compare_spec: CompareSpec, target_path: str) -> tuple[object, ...]:
    target = compare_spec.target
    view = compare_spec.view
    return (
        target.kind,
        target_path,
        target.void_id,
        view.axes,
        view.depth_axis,
    )


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.asarray(values, dtype=np.float64).std())


def _as_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"expected numeric value, got {type(value).__name__}")


def _metric_family(metric: str) -> str:
    if metric == "iou":
        return "shape_overlap"
    if metric == "sdf":
        return "shape_distance"
    if metric == "sdf_band":
        return "boundary_band_distance"
    if metric == "sdf_material":
        return "material_distance"
    if metric in {"cd", "chamfer", "corner", "profile", "topology"}:
        return "geometry_measure"
    return "other"


def _metric_set_summary_rows(
    *,
    metric_set_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    ranking_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in metric_set_rows:
        groups.setdefault(str(row["metric_set"]), []).append(row)

    skipped_by_metric_set: dict[str, int] = {}
    for row in metric_rows:
        if row.get("status") == "SKIPPED":
            metric_set = str(row["metric_set"])
            skipped_by_metric_set[metric_set] = skipped_by_metric_set.get(metric_set, 0) + 1

    rank_deltas_by_metric_set: dict[str, list[int]] = {}
    changed_rank_counts: dict[str, int] = {}
    for row in ranking_rows:
        metric_set = str(row["metric_set"])
        ranking_shift = row.get("ranking_shift")
        if ranking_shift == "":
            continue
        delta = abs(int(str(ranking_shift)))
        rank_deltas_by_metric_set.setdefault(metric_set, []).append(delta)
        if delta != 0:
            changed_rank_counts[metric_set] = changed_rank_counts.get(metric_set, 0) + 1

    summaries: list[dict[str, object]] = []
    for metric_set, rows in groups.items():
        comparison_loss = [_as_float(row["comparison_loss"]) for row in rows]
        raw_loss = [_as_float(row["raw_loss"]) for row in rows]
        runtime = [_as_float(row["runtime_sec"]) for row in rows]
        partial_count = sum(1 for row in rows if row.get("status") == "PARTIAL")
        first = rows[0]
        best = min(rows, key=lambda row: _as_float(row["comparison_loss"]))
        abs_rank_deltas = rank_deltas_by_metric_set.get(metric_set, [])
        summaries.append(
            {
                "metric_set": metric_set,
                "status": "OK" if partial_count == 0 else "PARTIAL",
                "direction": "minimize",
                "features": first.get("features", ""),
                "metrics": first.get("metrics", ""),
                "case_count": len(rows),
                "ok_case_count": len(rows) - partial_count,
                "partial_case_count": partial_count,
                "best_case_id": best.get("case_id", ""),
                "best_comparison_loss": best.get("comparison_loss", ""),
                "mean_comparison_loss": float(np.mean(comparison_loss)),
                "case_separation": _std(comparison_loss),
                "min_comparison_loss": float(np.min(comparison_loss)),
                "max_comparison_loss": float(np.max(comparison_loss)),
                "mean_raw_loss": float(np.mean(raw_loss)),
                "std_raw_loss": _std(raw_loss),
                "ranking_shift_mean": (
                    float(np.mean(abs_rank_deltas)) if abs_rank_deltas else 0.0
                ),
                "ranking_shift_max": int(max(abs_rank_deltas)) if abs_rank_deltas else 0,
                "changed_rank_count": changed_rank_counts.get(metric_set, 0),
                "mean_runtime_sec": float(np.mean(runtime)),
                "skipped_metric_count": skipped_by_metric_set.get(metric_set, 0),
            }
        )
    return summaries


def _metric_summary_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in metric_rows:
        groups.setdefault((str(row["metric_set"]), str(row["name"])), []).append(row)

    summaries: list[dict[str, object]] = []
    for (metric_set, metric), rows in sorted(groups.items()):
        losses = [_as_float(row["loss"]) for row in rows]
        normalized_losses = [_as_float(row["normalized_loss"]) for row in rows]
        values = [_as_float(row["value"]) for row in rows]
        summaries.append(
            {
                "metric_set": metric_set,
                "metric_family": _metric_family(metric),
                "metric": metric,
                "case_count": len(rows),
                "mean_metric_loss": float(np.mean(normalized_losses)),
                "mean_raw_metric_loss": float(np.mean(losses)),
                "std_raw_metric_loss": _std(losses),
                "min_raw_metric_loss": float(np.min(losses)),
                "max_raw_metric_loss": float(np.max(losses)),
                "mean_normalized_loss": float(np.mean(normalized_losses)),
                "mean_value": float(np.mean(values)),
                "skipped_count": sum(1 for row in rows if row.get("status") == "SKIPPED"),
            }
        )
    return summaries


def _ranking_consistency_rows(case_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_set_order: list[str] = []
    groups: dict[str, list[dict[str, object]]] = {}
    for row in case_rows:
        metric_set = str(row["metric_set"])
        if metric_set not in groups:
            metric_set_order.append(metric_set)
        groups.setdefault(metric_set, []).append(row)
    if not metric_set_order:
        return []

    ranks_by_metric_set: dict[str, dict[str, int]] = {}
    for metric_set, rows in groups.items():
        ordered = sorted(rows, key=lambda item: _as_float(item["comparison_loss"]))
        ranks_by_metric_set[metric_set] = {
            str(row["case_id"]): rank for rank, row in enumerate(ordered, start=1)
        }

    baseline = metric_set_order[0]
    baseline_ranks = ranks_by_metric_set[baseline]
    consistency_rows: list[dict[str, object]] = []
    for metric_set in metric_set_order:
        for case_id, metric_set_rank in sorted(ranks_by_metric_set[metric_set].items()):
            baseline_rank = baseline_ranks.get(case_id)
            consistency_rows.append(
                {
                    "baseline_metric_set": baseline,
                    "metric_set": metric_set,
                    "case_id": case_id,
                    "baseline_rank": baseline_rank,
                    "metric_set_rank": metric_set_rank,
                    "ranking_shift": (
                        int(metric_set_rank) - int(baseline_rank)
                        if baseline_rank is not None
                        else ""
                    ),
                }
            )
    return consistency_rows


def _axis_agreement_rows(case_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_set_order: list[str] = []
    by_axis: dict[str, dict[str, dict[str, object]]] = {}
    for row in case_rows:
        axis = str(row["metric_set"])
        if axis not in by_axis:
            metric_set_order.append(axis)
        by_axis.setdefault(axis, {})[str(row["case_id"])] = row

    ranks_by_axis: dict[str, dict[str, int]] = {}
    for axis in metric_set_order:
        ordered = sorted(
            by_axis[axis].values(),
            key=lambda item: _as_float(item["comparison_loss"]),
        )
        ranks_by_axis[axis] = {
            str(row["case_id"]): rank for rank, row in enumerate(ordered, start=1)
        }

    rows: list[dict[str, object]] = []
    for left_index, left_axis in enumerate(metric_set_order):
        for right_axis in metric_set_order[left_index + 1 :]:
            case_ids = sorted(set(by_axis[left_axis]).intersection(by_axis[right_axis]))
            left_losses = np.asarray(
                [_as_float(by_axis[left_axis][case_id]["comparison_loss"]) for case_id in case_ids],
                dtype=np.float64,
            )
            right_losses = np.asarray(
                [
                    _as_float(by_axis[right_axis][case_id]["comparison_loss"])
                    for case_id in case_ids
                ],
                dtype=np.float64,
            )
            if len(case_ids) >= 2 and np.std(left_losses) > 0.0 and np.std(right_losses) > 0.0:
                loss_correlation: float | str = float(np.corrcoef(left_losses, right_losses)[0, 1])
            elif left_losses.size and np.allclose(left_losses, right_losses):
                loss_correlation = 1.0
            else:
                loss_correlation = ""

            rank_shifts = [
                abs(ranks_by_axis[left_axis][case_id] - ranks_by_axis[right_axis][case_id])
                for case_id in case_ids
            ]
            max_possible_shift = max(len(case_ids) - 1, 1)
            mean_abs_rank_shift = float(np.mean(rank_shifts)) if rank_shifts else 0.0
            rank_agreement = max(0.0, 1.0 - mean_abs_rank_shift / max_possible_shift)
            rows.append(
                {
                    "axis_a": left_axis,
                    "axis_b": right_axis,
                    "case_count": len(case_ids),
                    "loss_correlation": loss_correlation,
                    "rank_agreement": rank_agreement,
                    "mean_abs_rank_shift": mean_abs_rank_shift,
                    "max_abs_rank_shift": max(rank_shifts) if rank_shifts else 0,
                    "changed_rank_count": sum(1 for value in rank_shifts if value != 0),
                }
            )
    return rows


def _case_score_fieldnames(metric_names: list[str]) -> list[str]:
    fields = [
        "metric_set",
        "case_id",
        "status",
        "direction",
        "comparison_loss",
        "raw_loss",
        "runtime_sec",
        "features",
        "metrics",
        "skipped_metrics",
    ]
    for metric in metric_names:
        fields.extend(
            [
                f"{metric}_loss",
                f"{metric}_value",
                f"{metric}_normalized_loss",
                f"{metric}_status",
            ]
        )
    return fields


def run_compare_eval_from_config(config_path: str | Path) -> dict[str, object]:
    config_file = Path(config_path).resolve()
    spec = load_compare_eval_spec_yaml(config_file)
    base_dir = config_file.parent
    index_path = resolve_path(spec.index, base_dir=base_dir)
    index_dir = index_path.parent
    out_dir = resolve_path(spec.output.dir, base_dir=base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_eval_output_dirs(out_dir)

    index_rows = read_compare_index(index_path)
    case_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    target_cache: dict[tuple[object, ...], PreparedTarget] = {}
    target_cache_hits = 0
    metric_names: list[str] = []
    seen_metric_set_ids: dict[str, str] = {}

    for metric_set_name, metric_set in spec.metric_sets.items():
        metric_set_id = _safe_metric_set_id(metric_set_name)
        existing_name = seen_metric_set_ids.get(metric_set_id)
        if existing_name is not None:
            raise ValueError(
                "compare-eval metric_set names collide after path sanitization: "
                f"{existing_name!r} and {metric_set_name!r} -> {metric_set_id!r}"
            )
        seen_metric_set_ids[metric_set_id] = metric_set_name
        metric_set_dir = out_dir / "metric_sets" / metric_set_id
        batch_spec = _metric_set_batch_spec(
            index=str(index_path),
            metric_set=metric_set,
            output_dir=metric_set_dir,
            view=spec.view,
        )
        for metric_name in metric_set.metrics.use:
            if metric_name not in metric_names:
                metric_names.append(metric_name)

        for row in index_rows:
            case_id = row["case_id"]
            case_row = dict(row)
            case_row["simulation_path"] = str(
                resolve_path(case_row["simulation_path"], base_dir=index_dir)
            )
            case_row["target_path"] = str(
                resolve_path(case_row["target_path"], base_dir=index_dir)
            )
            case_out = metric_set_dir / "cases" / case_id
            compare_spec = compare_spec_from_index_row(
                case_row,
                batch_spec=batch_spec,
                output_dir=case_out,
            )

            prepared_target: PreparedTarget | None = None
            if is_label_input_kind(compare_spec.target.kind):
                cache_key = _target_cache_key(compare_spec, case_row["target_path"])
                prepared_target = target_cache.get(cache_key)
                if prepared_target is None:
                    prepared_target = prepare_label_target_for_compare(
                        compare_spec,
                        Path(case_row["target_path"]),
                    )
                    target_cache[cache_key] = prepared_target
                else:
                    target_cache_hits += 1

            start = time.perf_counter()
            score, _summary = run_compare_spec(
                spec=compare_spec,
                config_file=config_file,
                base_dir=index_dir,
                output_dir=case_out,
                write_run_metadata=False,
                prepared_target=prepared_target,
                write_target_features=prepared_target is None,
                write_case_outputs=False,
            )
            runtime_sec = time.perf_counter() - start
            skipped = [metric.name for metric in score.metrics if metric.status == "SKIPPED"]

            case_row_out: dict[str, object] = {
                "metric_set": metric_set_name,
                "case_id": case_id,
                "status": "OK" if not skipped else "PARTIAL",
                "direction": "minimize",
                "comparison_loss": score.normalized_total_score,
                "raw_loss": score.total_score,
                "runtime_sec": runtime_sec,
                "features": "|".join(metric_set.features.use),
                "metrics": "|".join(metric_set.metrics.use),
                "skipped_metrics": "|".join(skipped),
            }
            for metric in score.metrics:
                metric_rows.append(
                    {
                        "metric_set": metric_set_name,
                        "case_id": case_id,
                        **metric.to_dict(),
                    }
                )
                case_row_out[f"{metric.name}_loss"] = metric.loss
                case_row_out[f"{metric.name}_value"] = metric.value
                case_row_out[f"{metric.name}_normalized_loss"] = metric.normalized_loss
                case_row_out[f"{metric.name}_status"] = metric.status
            case_rows.append(case_row_out)

    ranking_rows = _ranking_consistency_rows(case_rows)
    axis_agreement_rows = _axis_agreement_rows(case_rows)
    metric_set_rows = _metric_set_summary_rows(
        metric_set_rows=case_rows,
        metric_rows=metric_rows,
        ranking_rows=ranking_rows,
    )
    metric_summary_rows = _metric_summary_rows(metric_rows)

    _write_csv(
        out_dir / "metric_set_summary.csv",
        metric_set_rows,
        [
            "metric_set",
            "status",
            "direction",
            "features",
            "metrics",
            "case_count",
            "ok_case_count",
            "partial_case_count",
            "best_case_id",
            "best_comparison_loss",
            "mean_comparison_loss",
            "case_separation",
            "min_comparison_loss",
            "max_comparison_loss",
            "mean_raw_loss",
            "std_raw_loss",
            "ranking_shift_mean",
            "ranking_shift_max",
            "changed_rank_count",
            "mean_runtime_sec",
            "skipped_metric_count",
        ],
    )
    _write_csv(out_dir / "case_scores.csv", case_rows, _case_score_fieldnames(metric_names))
    _write_csv(
        out_dir / "metric_summary.csv",
        metric_summary_rows,
        [
            "metric_set",
            "metric_family",
            "metric",
            "case_count",
            "mean_metric_loss",
            "mean_raw_metric_loss",
            "std_raw_metric_loss",
            "min_raw_metric_loss",
            "max_raw_metric_loss",
            "mean_value",
            "skipped_count",
        ],
    )
    _write_csv(
        out_dir / "ranking_consistency.csv",
        ranking_rows,
        [
            "baseline_metric_set",
            "metric_set",
            "case_id",
            "baseline_rank",
            "metric_set_rank",
            "ranking_shift",
        ],
    )
    _write_csv(
        out_dir / "axis_agreement.csv",
        axis_agreement_rows,
        [
            "axis_a",
            "axis_b",
            "case_count",
            "loss_correlation",
            "rank_agreement",
            "mean_abs_rank_shift",
            "max_abs_rank_shift",
            "changed_rank_count",
        ],
    )
    figures_index = write_compare_eval_figures(
        out_dir=out_dir,
        spec=spec,
        index_rows=index_rows,
        index_dir=index_dir,
        case_rows=case_rows,
        metric_set_rows=metric_set_rows,
        metric_summary_rows=metric_summary_rows,
        ranking_rows=ranking_rows,
        axis_agreement_rows=axis_agreement_rows,
    )
    summary: dict[str, object] = {
        "task": "compare-eval",
        "status": "OK",
        "case_count": len(index_rows),
        "metric_set_count": len(spec.metric_sets),
        "metric_sets": list(spec.metric_sets),
        "baseline_metric_set": next(iter(spec.metric_sets), ""),
        "figures": figures_index,
        "target_cache": {
            "entries": len(target_cache),
            "hits": target_cache_hits,
        },
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "summary.json", summary)
    write_run_info(
        config_path=config_file,
        output_dir=out_dir,
        task="compare-eval",
        inputs={"index": str(index_path)},
    )
    return summary

