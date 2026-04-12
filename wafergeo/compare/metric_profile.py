from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wafergeo.compare.features import ViewFeature
from wafergeo.compare.metric_types import MetricComputation, MetricContext


def _line_from_view(arr: np.ndarray, *, line_idx: int, height_is_row: bool) -> np.ndarray:
    return arr[line_idx, :] if height_is_row else arr[:, line_idx]


@dataclass(frozen=True)
class _ProfileGrid:
    width_axis: str
    height_axis: str
    height_is_row: bool
    width_spacing: float
    width_origin: float
    width_size: int
    height_spacing: float
    height_origin: float
    height_size: int
    center_nm: float


def _resolve_profile_grid(feature: ViewFeature) -> _ProfileGrid | None:
    height_axis = "z"
    if height_axis not in feature.axes:
        return None
    width_axis = next(axis_name for axis_name in feature.axes if axis_name != height_axis)
    height_is_row = feature.axes[1] == height_axis
    if height_is_row:
        width_spacing = float(feature.grid2d.spacing[1])
        width_origin = float(feature.grid2d.origin[1])
        width_size = int(feature.mask.shape[1])
        height_spacing = float(feature.grid2d.spacing[0])
        height_origin = float(feature.grid2d.origin[0])
        height_size = int(feature.mask.shape[0])
    else:
        width_spacing = float(feature.grid2d.spacing[0])
        width_origin = float(feature.grid2d.origin[0])
        width_size = int(feature.mask.shape[0])
        height_spacing = float(feature.grid2d.spacing[1])
        height_origin = float(feature.grid2d.origin[1])
        height_size = int(feature.mask.shape[1])
    center_nm = width_origin + ((width_size - 1) / 2.0) * width_spacing
    return _ProfileGrid(
        width_axis=width_axis,
        height_axis=height_axis,
        height_is_row=height_is_row,
        width_spacing=width_spacing,
        width_origin=width_origin,
        width_size=width_size,
        height_spacing=height_spacing,
        height_origin=height_origin,
        height_size=height_size,
        center_nm=float(center_nm),
    )


def _transition_positions_nm(
    *,
    label_line: np.ndarray,
    mask_line: np.ndarray,
    width_origin: float,
    width_spacing: float,
) -> list[float]:
    labels = np.asarray(label_line)
    mask = np.asarray(mask_line, dtype=bool)
    if labels.size < 2:
        return []

    left_solid = mask[:-1]
    right_solid = mask[1:]
    solid_edges = left_solid != right_solid
    material_edges = left_solid & right_solid & (labels[:-1] != labels[1:])
    edge_idx = np.flatnonzero(solid_edges | material_edges)
    return [
        float(width_origin + (float(idx) + 0.5) * width_spacing)
        for idx in edge_idx
    ]


def _mask_extent_nm(
    *,
    mask_line: np.ndarray,
    width_origin: float,
    width_spacing: float,
) -> tuple[float, float] | None:
    present_idx = np.flatnonzero(np.asarray(mask_line, dtype=bool))
    if present_idx.size == 0:
        return None
    left_nm = float(width_origin + int(present_idx[0]) * width_spacing)
    right_nm = float(width_origin + int(present_idx[-1]) * width_spacing)
    if left_nm == right_nm:
        return None
    return left_nm, right_nm


def _nearest_width_around_center(
    *,
    positions_nm: list[float],
    mask_line: np.ndarray,
    center_nm: float,
    width_origin: float,
    width_spacing: float,
) -> tuple[float, float] | None:
    left = [value for value in positions_nm if value < center_nm]
    right = [value for value in positions_nm if value > center_nm]
    if left and right:
        return max(left), min(right)
    return _mask_extent_nm(
        mask_line=mask_line,
        width_origin=width_origin,
        width_spacing=width_spacing,
    )


def _profile_rows(feature: ViewFeature) -> tuple[list[dict[str, float]], dict[str, object]]:
    grid = _resolve_profile_grid(feature)
    summary: dict[str, object] = {
        "metric": "profile",
        "status": "SKIPPED",
        "profile_mode": "width_center_edges",
        "view_axes": list(feature.axes),
        "height_axis": "z",
        "width_axis": None,
        "height_count": 0,
        "profile_rows": 0,
        "skipped_reason": None,
    }
    if grid is None:
        summary["skipped_reason"] = "profile requires a cross-section view that includes z"
        return [], summary

    summary.update(
        {
            "width_axis": grid.width_axis,
            "height_count": grid.height_size,
            "center_nm": grid.center_nm,
        }
    )
    rows: list[dict[str, float]] = []
    for line_idx in range(grid.height_size):
        height_nm = float(grid.height_origin + line_idx * grid.height_spacing)
        mask_line = _line_from_view(
            feature.mask,
            line_idx=line_idx,
            height_is_row=grid.height_is_row,
        )
        label_line = _line_from_view(
            feature.label2d,
            line_idx=line_idx,
            height_is_row=grid.height_is_row,
        )
        positions = _transition_positions_nm(
            label_line=label_line,
            mask_line=mask_line,
            width_origin=grid.width_origin,
            width_spacing=grid.width_spacing,
        )
        edges = _nearest_width_around_center(
            positions_nm=positions,
            mask_line=mask_line,
            center_nm=grid.center_nm,
            width_origin=grid.width_origin,
            width_spacing=grid.width_spacing,
        )
        if edges is None:
            continue
        left_nm, right_nm = edges
        width_nm = abs(right_nm - left_nm)
        center_nm = 0.5 * (left_nm + right_nm)
        rows.append(
            {
                "height_nm": height_nm,
                "left_nm": float(left_nm),
                "right_nm": float(right_nm),
                "width_nm": float(width_nm),
                "center_nm": float(center_nm),
            }
        )

    summary["profile_rows"] = len(rows)
    if not rows:
        summary["skipped_reason"] = "profile has no measurable width rows"
        return [], summary
    summary["status"] = "OK"
    return rows, summary


def _compare_profiles(
    sim_rows: list[dict[str, float]],
    target_rows: list[dict[str, float]],
) -> tuple[float, list[dict[str, float]], dict[str, object]]:
    sim_by_height = {round(row["height_nm"], 6): row for row in sim_rows}
    target_by_height = {round(row["height_nm"], 6): row for row in target_rows}
    shared_heights = sorted(set(sim_by_height).intersection(target_by_height))
    summary: dict[str, object] = {
        "shared_height_count": len(shared_heights),
        "width_abs_diff_mean_nm": None,
        "width_abs_diff_max_nm": None,
        "center_abs_diff_mean_nm": None,
        "center_abs_diff_max_nm": None,
        "edge_loss_mean_nm": None,
        "edge_loss_max_nm": None,
        "profile_loss_mean_nm": None,
        "selected_loss_source": None,
    }
    if not shared_heights:
        return 0.0, [], summary

    rows: list[dict[str, float]] = []
    width_diffs: list[float] = []
    center_diffs: list[float] = []
    edge_losses: list[float] = []
    for height in shared_heights:
        sim_row = sim_by_height[height]
        target_row = target_by_height[height]
        width_diff = float(sim_row["width_nm"] - target_row["width_nm"])
        center_diff = float(sim_row["center_nm"] - target_row["center_nm"])
        left_diff = float(sim_row["left_nm"] - target_row["left_nm"])
        right_diff = float(sim_row["right_nm"] - target_row["right_nm"])
        edge_loss = 0.5 * (abs(left_diff) + abs(right_diff))
        width_diffs.append(abs(width_diff))
        center_diffs.append(abs(center_diff))
        edge_losses.append(edge_loss)
        rows.append(
            {
                "height_nm": float(height),
                "sim_width_nm": float(sim_row["width_nm"]),
                "target_width_nm": float(target_row["width_nm"]),
                "width_diff_nm": width_diff,
                "width_abs_diff_nm": abs(width_diff),
                "sim_center_nm": float(sim_row["center_nm"]),
                "target_center_nm": float(target_row["center_nm"]),
                "center_diff_nm": center_diff,
                "center_abs_diff_nm": abs(center_diff),
                "sim_left_nm": float(sim_row["left_nm"]),
                "sim_right_nm": float(sim_row["right_nm"]),
                "target_left_nm": float(target_row["left_nm"]),
                "target_right_nm": float(target_row["right_nm"]),
                "left_diff_nm": left_diff,
                "right_diff_nm": right_diff,
                "edge_loss_nm": edge_loss,
            }
        )

    width_mean = float(np.mean(width_diffs))
    center_mean = float(np.mean(center_diffs))
    edge_mean = float(np.mean(edge_losses))
    candidates = {
        "width_abs_diff_mean_nm": width_mean,
        "center_abs_diff_mean_nm": center_mean,
        "edge_loss_mean_nm": edge_mean,
    }
    selected_loss_source = max(candidates, key=lambda name: candidates[name])
    loss = float(candidates[selected_loss_source])
    summary.update(
        {
            "width_abs_diff_mean_nm": width_mean,
            "width_abs_diff_max_nm": float(np.max(width_diffs)),
            "center_abs_diff_mean_nm": center_mean,
            "center_abs_diff_max_nm": float(np.max(center_diffs)),
            "edge_loss_mean_nm": edge_mean,
            "edge_loss_max_nm": float(np.max(edge_losses)),
            "profile_loss_mean_nm": loss,
            "selected_loss_source": selected_loss_source,
        }
    )
    return loss, rows, summary


def compute_profile(
    sim: ViewFeature,
    target: ViewFeature,
    context: MetricContext,
) -> MetricComputation:
    del context
    sim_profile, sim_summary = _profile_rows(sim)
    target_profile, target_summary = _profile_rows(target)
    summary: dict[str, object] = {
        "metric": "profile",
        "status": "SKIPPED",
        "profile_mode": "width_center_edges",
        "view_axes": list(sim.axes),
        "sim": sim_summary,
        "target": target_summary,
        "shared_height_count": 0,
        "skipped_reason": None,
    }
    if not sim_profile or not target_profile:
        summary["skipped_reason"] = (
            "simulation profile is empty"
            if not sim_profile
            else "target profile is empty"
        )
        return MetricComputation(
            name="profile",
            loss=0.0,
            value=0.0,
            status="SKIPPED",
            profile_summary=summary,
        )

    value, rows, compare_summary = _compare_profiles(sim_profile, target_profile)
    summary.update(compare_summary)
    if not rows:
        summary["skipped_reason"] = "simulation and target profiles have no shared height samples"
        return MetricComputation(
            name="profile",
            loss=0.0,
            value=0.0,
            status="SKIPPED",
            profile_summary=summary,
        )

    summary["status"] = "OK"
    return MetricComputation(
        name="profile",
        loss=value,
        value=value,
        status="OK",
        profile_rows=rows,
        profile_summary=summary,
    )
