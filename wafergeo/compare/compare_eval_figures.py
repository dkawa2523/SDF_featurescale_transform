from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wafergeo.compare.eval_figure_common import (
    COMPARE_FIGURE_NOTES,
    _as_float,
    _clamp01,
    _difference_rgb,
    _imshow_or_note,
    _load_matplotlib_pyplot,
    _matrix_from_rows,
    _ok_figure_index,
    _optional_int,
    _plot_grouped_bars,
    _plot_heatmap,
    _rel,
    _safe_file_stem,
    _shape_section_xz,
    _shape_section_yz,
    _write_csv,
)
from wafergeo.compare.features import contour_feature_on_grid, extract_view_feature
from wafergeo.compare.loader import CONTOUR_LOADERS, load_simulation_label
from wafergeo.compare.runtime_io import resolve_path
from wafergeo.compare.schema_types import CompareEvalSpec
from wafergeo.core.types import LabelVolume


def _evaluation_axis_summary_rows(
    metric_set_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    max_std = max(
        (_as_float(row.get("case_separation")) for row in metric_set_rows),
        default=0.0,
    )
    rows: list[dict[str, object]] = []
    for row in metric_set_rows:
        metrics = [value for value in str(row.get("metrics", "")).split("|") if value]
        case_count = max(_as_float(row.get("case_count")), 1.0)
        metric_count = max(float(len(metrics)), 1.0)
        case_coverage = _clamp01(_as_float(row.get("ok_case_count")) / case_count)
        skipped = _as_float(row.get("skipped_metric_count"))
        metric_coverage = _clamp01(1.0 - skipped / (case_count * metric_count))
        case_separation = (
            _clamp01(_as_float(row.get("case_separation")) / max_std)
            if max_std > 0.0
            else 0.0
        )
        rows.append(
            {
                "metric_set": row.get("metric_set", ""),
                "case_coverage": case_coverage,
                "metric_coverage": metric_coverage,
                "case_separation": case_separation,
                "ranking_shift_mean": _as_float(row.get("ranking_shift_mean")),
            }
        )
    return rows


def _plot_evaluation_axis_summary(
    *,
    plt: Any,
    path: Path,
    rows: list[dict[str, object]],
) -> bool:
    labels = [str(row.get("metric_set", "")) for row in rows]
    series = {
        "case_coverage": [_as_float(row.get("case_coverage")) for row in rows],
        "metric_coverage": [_as_float(row.get("metric_coverage")) for row in rows],
        "case_separation": [_as_float(row.get("case_separation")) for row in rows],
        "ranking_shift_mean": [_as_float(row.get("ranking_shift_mean")) for row in rows],
    }
    return _plot_grouped_bars(
        plt=plt,
        path=path,
        labels=labels,
        series=series,
        title="evaluation axis summary",
        ylabel="coverage / normalized separation / rank shift",
    )


def _axis_with_metric(spec: CompareEvalSpec, metric: str) -> str | None:
    for name, metric_set in spec.metric_sets.items():
        if metric in metric_set.metrics.use:
            return name
    return None


def _cd_sdf_axis_pair(spec: CompareEvalSpec) -> tuple[str, str] | None:
    cd_axis = "height_cd" if "height_cd" in spec.metric_sets else _axis_with_metric(spec, "cd")
    sdf_axis = (
        "shape_distance"
        if "shape_distance" in spec.metric_sets
        else _axis_with_metric(spec, "sdf")
    )
    if cd_axis is None or sdf_axis is None or cd_axis == sdf_axis:
        return None
    return cd_axis, sdf_axis


def _plot_cd_vs_sdf_scatter(
    *,
    plt: Any,
    path: Path,
    spec: CompareEvalSpec,
    case_rows: list[dict[str, object]],
) -> bool:
    pair = _cd_sdf_axis_pair(spec)
    if pair is None:
        return False
    cd_axis, sdf_axis = pair
    by_key = {(str(row["metric_set"]), str(row["case_id"])): row for row in case_rows}
    case_ids = [
        str(row["case_id"])
        for row in case_rows
        if row.get("metric_set") == cd_axis
        and (sdf_axis, str(row.get("case_id"))) in by_key
    ]
    if not case_ids:
        return False

    x = np.asarray(
        [_as_float(by_key[(cd_axis, case_id)].get("comparison_loss")) for case_id in case_ids],
        dtype=np.float64,
    )
    y = np.asarray(
        [_as_float(by_key[(sdf_axis, case_id)].get("comparison_loss")) for case_id in case_ids],
        dtype=np.float64,
    )

    grouped: dict[tuple[float, float], list[str]] = {}
    for case_id, x_value, y_value in zip(case_ids, x, y, strict=True):
        key = (round(float(x_value), 12), round(float(y_value), 12))
        grouped.setdefault(key, []).append(case_id)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    for (x_value, y_value), labels in grouped.items():
        ax.scatter([x_value], [y_value], s=35 + 10 * min(len(labels), 4))
        if len(labels) <= 2:
            text = ", ".join(labels)
        else:
            text = f"{labels[0]}, {labels[1]} +{len(labels) - 2}"
        ax.annotate(text, (x_value, y_value), xytext=(4, 4), textcoords="offset points")
    max_value = float(max(np.max(x), np.max(y), 1e-9))
    ax.plot([0.0, max_value], [0.0, max_value], color="0.5", linestyle="--", linewidth=1.0)
    ax.set_title("height CD vs SDF shape loss")
    ax.set_xlabel(f"{cd_axis} comparison_loss")
    ax.set_ylabel(f"{sdf_axis} comparison_loss")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _representative_compare_keys(
    *,
    spec: CompareEvalSpec,
    case_rows: list[dict[str, object]],
    ranking_rows: list[dict[str, object]],
) -> list[tuple[str, str]]:
    baseline = next(iter(spec.metric_sets), "")
    selected: list[tuple[str, str]] = []
    baseline_rows = [row for row in case_rows if row.get("metric_set") == baseline]
    if baseline_rows:
        best = min(baseline_rows, key=lambda row: _as_float(row.get("comparison_loss")))
        worst = max(baseline_rows, key=lambda row: _as_float(row.get("comparison_loss")))
        selected.extend([(baseline, str(best["case_id"])), (baseline, str(worst["case_id"]))])
    if ranking_rows:
        rank_row = max(
            ranking_rows,
            key=lambda row: abs(int(_as_float(row.get("ranking_shift")))),
        )
        selected.append((str(rank_row.get("metric_set", "")), str(rank_row.get("case_id", ""))))
    pair = _cd_sdf_axis_pair(spec)
    if pair is not None:
        cd_axis, sdf_axis = pair
        by_key = {(str(row["metric_set"]), str(row["case_id"])): row for row in case_rows}
        case_ids = [
            case_id
            for axis, case_id in by_key
            if axis == cd_axis and (sdf_axis, case_id) in by_key
        ]
        if case_ids:
            case_id = max(
                case_ids,
                key=lambda value: abs(
                    _as_float(by_key[(cd_axis, value)].get("comparison_loss"))
                    - _as_float(by_key[(sdf_axis, value)].get("comparison_loss"))
                ),
            )
            selected.append((sdf_axis, case_id))

    unique: list[tuple[str, str]] = []
    for item in selected:
        if item[0] and item[1] and item not in unique:
            unique.append(item)
    return unique[:3]


def _compare_view_features(
    *,
    spec: CompareEvalSpec,
    index_row: dict[str, str],
    index_dir: Path,
) -> tuple[np.ndarray, np.ndarray, LabelVolume, LabelVolume | None]:
    sim_path = resolve_path(index_row["simulation_path"], base_dir=index_dir)
    target_path = resolve_path(index_row["target_path"], base_dir=index_dir)
    target_kind = index_row.get("target_kind") or "contour_json"
    sim_label = load_simulation_label(
        index_row["simulation_kind"],
        sim_path,
        void_id=_optional_int(index_row.get("simulation_void_id")),
    )
    sim_feature = extract_view_feature(
        sim_label,
        axes=spec.view.axes,
        depth_axis=spec.view.depth_axis,
        contour_mode="outer" if target_kind == "contour_json" else "material",
    )
    if target_kind in CONTOUR_LOADERS:
        target_data = CONTOUR_LOADERS[target_kind](
            target_path,
            units_override=index_row.get("target_units") or "nm",
            view_axes=spec.view.axes,
        )
        target_feature = contour_feature_on_grid(
            target_data,
            sim_feature.grid2d,
            sim_feature.mask.shape,
            axes=spec.view.axes,
        )
        target_label = None
    else:
        target_label = load_simulation_label(
            target_kind,
            target_path,
            void_id=_optional_int(index_row.get("target_void_id")),
        )
        target_feature = extract_view_feature(
            target_label,
            axes=spec.view.axes,
            depth_axis=spec.view.depth_axis,
            contour_mode="material",
        )
    return sim_feature.label2d, target_feature.label2d, sim_label, target_label


def _show_diff_or_note(
    axis: Any,
    *,
    simulation: np.ndarray,
    target: np.ndarray | None,
    title: str,
) -> None:
    axis.set_title(title)
    if target is None:
        axis.text(0.5, 0.5, "2D contour target", ha="center", va="center")
    elif simulation.shape != target.shape:
        axis.text(0.5, 0.5, "shape mismatch", ha="center", va="center")
    else:
        axis.imshow(_difference_rgb(simulation, target), origin="lower")
    axis.set_xticks([])
    axis.set_yticks([])


def _show_label_or_note(
    axis: Any,
    *,
    image: np.ndarray | None,
    title: str,
) -> None:
    axis.set_title(title)
    if image is None:
        axis.text(0.5, 0.5, "2D contour target", ha="center", va="center")
    else:
        axis.imshow(image, cmap="tab20", origin="lower")
    axis.set_xticks([])
    axis.set_yticks([])


def _plot_representative_difference(
    *,
    plt: Any,
    path: Path,
    spec: CompareEvalSpec,
    index_row: dict[str, str],
    index_dir: Path,
    score_row: dict[str, object],
) -> bool:
    sim_label, target_label, sim_volume, target_volume = _compare_view_features(
        spec=spec,
        index_row=index_row,
        index_dir=index_dir,
    )
    loss_items = [
        f"{key[:-5]}={_as_float(value):.3g}"
        for key, value in score_row.items()
        if key.endswith("_loss") and value != ""
    ][:3]
    title = (
        f"{score_row.get('metric_set')} / {score_row.get('case_id')} / "
        f"comparison_loss={_as_float(score_row.get('comparison_loss')):.3g}"
    )
    if loss_items:
        title = f"{title} / {', '.join(loss_items)}"

    sim_xz = _shape_section_xz(sim_volume)
    sim_yz = _shape_section_yz(sim_volume)
    target_xz = _shape_section_xz(target_volume) if target_volume is not None else None
    target_yz = _shape_section_yz(target_volume) if target_volume is not None else None

    fig, axes = plt.subplots(3, 3, figsize=(12.0, 10.0), constrained_layout=True)
    axes[0][0].imshow(target_label, cmap="tab20", origin="lower")
    axes[0][0].set_title("target view")
    axes[0][1].imshow(sim_label, cmap="tab20", origin="lower")
    axes[0][1].set_title("simulation view")
    axes[0][2].imshow(_difference_rgb(sim_label, target_label), origin="lower")
    axes[0][2].set_title("view diff")

    _show_label_or_note(axes[1][0], image=target_xz, title="target [x,z]")
    _imshow_or_note(axes[1][1], sim_xz, "simulation [x,z]", cmap="tab20")
    _show_diff_or_note(axes[1][2], simulation=sim_xz, target=target_xz, title="[x,z] diff")

    _show_label_or_note(axes[2][0], image=target_yz, title="target [y,z]")
    _imshow_or_note(axes[2][1], sim_yz, "simulation [y,z]", cmap="tab20")
    _show_diff_or_note(axes[2][2], simulation=sim_yz, target=target_yz, title="[y,z] diff")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def write_compare_eval_figures(
    *,
    out_dir: Path,
    spec: CompareEvalSpec,
    index_rows: list[dict[str, str]],
    index_dir: Path,
    case_rows: list[dict[str, object]],
    metric_set_rows: list[dict[str, object]],
    metric_summary_rows: list[dict[str, object]],
    ranking_rows: list[dict[str, object]],
    axis_agreement_rows: list[dict[str, object]],
) -> dict[str, object]:
    plt = _load_matplotlib_pyplot()
    figures_dir = out_dir / "figures"

    figures: list[str] = []
    data: list[str] = []
    axis_summary_rows = _evaluation_axis_summary_rows(metric_set_rows)
    axis_summary_csv = figures_dir / "evaluation_axis_summary.csv"
    _write_csv(
        axis_summary_csv,
        axis_summary_rows,
        [
            "metric_set",
            "case_coverage",
            "metric_coverage",
            "case_separation",
            "ranking_shift_mean",
        ],
    )
    data.append(_rel(axis_summary_csv, out_dir))
    if axis_agreement_rows:
        data.append("axis_agreement.csv")

    row_labels, col_labels, matrix = _matrix_from_rows(
        case_rows,
        row_key="metric_set",
        col_key="case_id",
        value_key="comparison_loss",
    )
    comparison_loss_path = figures_dir / "comparison_loss_heatmap.png"
    if _plot_heatmap(
        plt=plt,
        path=comparison_loss_path,
        row_labels=row_labels,
        col_labels=col_labels,
        values=matrix,
        title="comparison loss heatmap (lower is better)",
        vmax=None,
    ):
        figures.append(_rel(comparison_loss_path, out_dir))

    rank_rows, rank_cols, rank_matrix = _matrix_from_rows(
        ranking_rows,
        row_key="metric_set",
        col_key="case_id",
        value_key="ranking_shift",
    )
    max_delta = float(np.max(np.abs(rank_matrix))) if rank_matrix.size else 0.0
    rank_path = figures_dir / "ranking_shift_heatmap.png"
    if _plot_heatmap(
        plt=plt,
        path=rank_path,
        row_labels=rank_rows,
        col_labels=rank_cols,
        values=rank_matrix,
        title="ranking shift heatmap",
        cmap="coolwarm",
        vmin=-max(max_delta, 1.0),
        vmax=max(max_delta, 1.0),
    ):
        figures.append(_rel(rank_path, out_dir))

    contrib_rows, contrib_cols, contrib_matrix = _matrix_from_rows(
        metric_summary_rows,
        row_key="metric_set",
        col_key="metric",
        value_key="mean_metric_loss",
    )
    contrib_path = figures_dir / "metric_loss_breakdown.png"
    if _plot_heatmap(
        plt=plt,
        path=contrib_path,
        row_labels=contrib_rows,
        col_labels=contrib_cols,
        values=contrib_matrix,
        title="metric loss breakdown",
        vmax=None,
    ):
        figures.append(_rel(contrib_path, out_dir))

    cd_sdf_path = figures_dir / "cd_vs_sdf_scatter.png"
    if _plot_cd_vs_sdf_scatter(
        plt=plt,
        path=cd_sdf_path,
        spec=spec,
        case_rows=case_rows,
    ):
        figures.append(_rel(cd_sdf_path, out_dir))

    axis_summary_path = figures_dir / "evaluation_axis_summary.png"
    if _plot_evaluation_axis_summary(
        plt=plt,
        path=axis_summary_path,
        rows=axis_summary_rows,
    ):
        figures.append(_rel(axis_summary_path, out_dir))

    case_by_key = {(str(row["metric_set"]), str(row["case_id"])): row for row in case_rows}
    index_by_case = {row["case_id"]: row for row in index_rows}
    for metric_set, case_id in _representative_compare_keys(
        spec=spec,
        case_rows=case_rows,
        ranking_rows=ranking_rows,
    ):
        index_row = index_by_case.get(case_id)
        score_row = case_by_key.get((metric_set, case_id))
        if index_row is None or score_row is None:
            continue
        path = (
            figures_dir
            / "representative_differences"
            / f"{_safe_file_stem(metric_set)}_{_safe_file_stem(case_id)}.png"
        )
        if _plot_representative_difference(
            plt=plt,
            path=path,
            spec=spec,
            index_row=index_row,
            index_dir=index_dir,
            score_row=score_row,
        ):
            figures.append(_rel(path, out_dir))

    return _ok_figure_index(out_dir, figures, data, COMPARE_FIGURE_NOTES)

