from __future__ import annotations

import csv
import re
import shutil
import time
from pathlib import Path

import numpy as np

from wafergeo.compare.batch_runner import compare_spec_from_index_row, read_compare_index
from wafergeo.compare.eval_figures import write_compare_eval_figures
from wafergeo.compare.loader import is_label_input_kind
from wafergeo.compare.runner import (
    PreparedTarget,
    prepare_label_target_for_compare,
    run_compare_spec,
)
from wafergeo.compare.runtime_io import resolve_path, write_json, write_run_info
from wafergeo.compare.schema import (
    BatchCompareSpec,
    CompareEvalCandidateSpec,
    CompareSpec,
    OutputSpec,
    ViewSpec,
    load_compare_eval_spec_yaml,
)


def _safe_candidate_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if not safe or safe in {".", ".."}:
        raise ValueError(f"invalid compare-eval candidate name: {value!r}")
    return safe


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _clean_eval_output_dirs(out_dir: Path) -> None:
    for name in ("candidates", "figures"):
        path = out_dir / name
        if path.exists():
            shutil.rmtree(path)


def _candidate_batch_spec(
    *,
    index: str,
    candidate: CompareEvalCandidateSpec,
    output_dir: Path,
    view: ViewSpec,
) -> BatchCompareSpec:
    return BatchCompareSpec(
        task="batch-compare",
        index=index,
        view=view,
        features=candidate.features,
        metrics=candidate.metrics,
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


def _candidate_summary_rows(
    *,
    candidate_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    ranking_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in candidate_rows:
        groups.setdefault(str(row["candidate"]), []).append(row)

    skipped_by_candidate: dict[str, int] = {}
    for row in metric_rows:
        if row.get("status") == "SKIPPED":
            candidate = str(row["candidate"])
            skipped_by_candidate[candidate] = skipped_by_candidate.get(candidate, 0) + 1

    rank_deltas_by_candidate: dict[str, list[int]] = {}
    changed_rank_counts: dict[str, int] = {}
    for row in ranking_rows:
        candidate = str(row["candidate"])
        rank_delta = row.get("rank_delta")
        if rank_delta == "":
            continue
        delta = abs(int(str(rank_delta)))
        rank_deltas_by_candidate.setdefault(candidate, []).append(delta)
        if delta != 0:
            changed_rank_counts[candidate] = changed_rank_counts.get(candidate, 0) + 1

    summaries: list[dict[str, object]] = []
    for candidate, rows in groups.items():
        normalized = [_as_float(row["normalized_total_score"]) for row in rows]
        total = [_as_float(row["total_score"]) for row in rows]
        runtime = [_as_float(row["runtime_sec"]) for row in rows]
        partial_count = sum(1 for row in rows if row.get("status") == "PARTIAL")
        first = rows[0]
        best = min(rows, key=lambda row: _as_float(row["normalized_total_score"]))
        abs_rank_deltas = rank_deltas_by_candidate.get(candidate, [])
        summaries.append(
            {
                "candidate": candidate,
                "status": "OK" if partial_count == 0 else "PARTIAL",
                "objective_name": "normalized_total_score",
                "direction": "minimize",
                "features": first.get("features", ""),
                "metrics": first.get("metrics", ""),
                "case_count": len(rows),
                "ok_case_count": len(rows) - partial_count,
                "partial_case_count": partial_count,
                "best_case_id": best.get("case_id", ""),
                "best_objective": best.get("objective", ""),
                "best_normalized_total_score": best.get("normalized_total_score", ""),
                "mean_objective": float(np.mean(normalized)),
                "mean_normalized_total_score": float(np.mean(normalized)),
                "std_normalized_total_score": _std(normalized),
                "min_normalized_total_score": float(np.min(normalized)),
                "max_normalized_total_score": float(np.max(normalized)),
                "mean_total_score": float(np.mean(total)),
                "std_total_score": _std(total),
                "mean_abs_rank_delta": (
                    float(np.mean(abs_rank_deltas)) if abs_rank_deltas else 0.0
                ),
                "max_abs_rank_delta": int(max(abs_rank_deltas)) if abs_rank_deltas else 0,
                "changed_rank_count": changed_rank_counts.get(candidate, 0),
                "mean_runtime_sec": float(np.mean(runtime)),
                "skipped_metric_count": skipped_by_candidate.get(candidate, 0),
            }
        )
    return summaries


def _metric_summary_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in metric_rows:
        groups.setdefault((str(row["candidate"]), str(row["name"])), []).append(row)

    summaries: list[dict[str, object]] = []
    for (candidate, metric), rows in sorted(groups.items()):
        losses = [_as_float(row["loss"]) for row in rows]
        normalized_losses = [_as_float(row["normalized_loss"]) for row in rows]
        values = [_as_float(row["value"]) for row in rows]
        summaries.append(
            {
                "candidate": candidate,
                "metric": metric,
                "case_count": len(rows),
                "mean_loss": float(np.mean(losses)),
                "std_loss": _std(losses),
                "min_loss": float(np.min(losses)),
                "max_loss": float(np.max(losses)),
                "mean_normalized_loss": float(np.mean(normalized_losses)),
                "mean_value": float(np.mean(values)),
                "skipped_count": sum(1 for row in rows if row.get("status") == "SKIPPED"),
            }
        )
    return summaries


def _ranking_consistency_rows(case_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    candidate_order: list[str] = []
    groups: dict[str, list[dict[str, object]]] = {}
    for row in case_rows:
        candidate = str(row["candidate"])
        if candidate not in groups:
            candidate_order.append(candidate)
        groups.setdefault(candidate, []).append(row)
    if not candidate_order:
        return []

    ranks_by_candidate: dict[str, dict[str, int]] = {}
    for candidate, rows in groups.items():
        ordered = sorted(rows, key=lambda item: _as_float(item["normalized_total_score"]))
        ranks_by_candidate[candidate] = {
            str(row["case_id"]): rank for rank, row in enumerate(ordered, start=1)
        }

    baseline = candidate_order[0]
    baseline_ranks = ranks_by_candidate[baseline]
    consistency_rows: list[dict[str, object]] = []
    for candidate in candidate_order:
        for case_id, candidate_rank in sorted(ranks_by_candidate[candidate].items()):
            baseline_rank = baseline_ranks.get(case_id)
            consistency_rows.append(
                {
                    "baseline_candidate": baseline,
                    "candidate": candidate,
                    "case_id": case_id,
                    "baseline_rank": baseline_rank,
                    "candidate_rank": candidate_rank,
                    "rank_delta": (
                        int(candidate_rank) - int(baseline_rank)
                        if baseline_rank is not None
                        else ""
                    ),
                }
            )
    return consistency_rows


def _case_score_fieldnames(metric_names: list[str]) -> list[str]:
    fields = [
        "candidate",
        "case_id",
        "status",
        "objective",
        "objective_name",
        "direction",
        "normalized_total_score",
        "total_score",
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
    seen_candidate_ids: dict[str, str] = {}

    for candidate_name, candidate in spec.candidates.items():
        candidate_id = _safe_candidate_id(candidate_name)
        existing_name = seen_candidate_ids.get(candidate_id)
        if existing_name is not None:
            raise ValueError(
                "compare-eval candidate names collide after path sanitization: "
                f"{existing_name!r} and {candidate_name!r} -> {candidate_id!r}"
            )
        seen_candidate_ids[candidate_id] = candidate_name
        candidate_dir = out_dir / "candidates" / candidate_id
        batch_spec = _candidate_batch_spec(
            index=str(index_path),
            candidate=candidate,
            output_dir=candidate_dir,
            view=spec.view,
        )
        for metric_name in candidate.metrics.use:
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
            case_out = candidate_dir / "cases" / case_id
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
                "candidate": candidate_name,
                "case_id": case_id,
                "status": "OK" if not skipped else "PARTIAL",
                "objective": score.normalized_total_score,
                "objective_name": "normalized_total_score",
                "direction": "minimize",
                "normalized_total_score": score.normalized_total_score,
                "total_score": score.total_score,
                "runtime_sec": runtime_sec,
                "features": "|".join(candidate.features.use),
                "metrics": "|".join(candidate.metrics.use),
                "skipped_metrics": "|".join(skipped),
            }
            for metric in score.metrics:
                metric_rows.append(
                    {
                        "candidate": candidate_name,
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
    candidate_rows = _candidate_summary_rows(
        candidate_rows=case_rows,
        metric_rows=metric_rows,
        ranking_rows=ranking_rows,
    )
    metric_summary_rows = _metric_summary_rows(metric_rows)

    _write_csv(
        out_dir / "candidate_summary.csv",
        candidate_rows,
        [
            "candidate",
            "status",
            "objective_name",
            "direction",
            "features",
            "metrics",
            "case_count",
            "ok_case_count",
            "partial_case_count",
            "best_case_id",
            "best_objective",
            "best_normalized_total_score",
            "mean_objective",
            "mean_normalized_total_score",
            "std_normalized_total_score",
            "min_normalized_total_score",
            "max_normalized_total_score",
            "mean_total_score",
            "std_total_score",
            "mean_abs_rank_delta",
            "max_abs_rank_delta",
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
            "candidate",
            "metric",
            "case_count",
            "mean_loss",
            "std_loss",
            "min_loss",
            "max_loss",
            "mean_normalized_loss",
            "mean_value",
            "skipped_count",
        ],
    )
    _write_csv(
        out_dir / "ranking_consistency.csv",
        ranking_rows,
        [
            "baseline_candidate",
            "candidate",
            "case_id",
            "baseline_rank",
            "candidate_rank",
            "rank_delta",
        ],
    )
    figures_manifest = write_compare_eval_figures(
        out_dir=out_dir,
        spec=spec,
        index_rows=index_rows,
        index_dir=index_dir,
        case_rows=case_rows,
        candidate_rows=candidate_rows,
        metric_summary_rows=metric_summary_rows,
        ranking_rows=ranking_rows,
    )
    summary: dict[str, object] = {
        "task": "compare-eval",
        "status": "OK",
        "case_count": len(index_rows),
        "candidate_count": len(spec.candidates),
        "candidates": list(spec.candidates),
        "baseline_candidate": next(iter(spec.candidates), ""),
        "figures": figures_manifest,
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
