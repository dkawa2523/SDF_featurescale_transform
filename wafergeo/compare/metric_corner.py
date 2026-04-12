from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wafergeo.compare.features import ViewFeature
from wafergeo.compare.metric_types import MetricComputation, MetricContext


@dataclass(frozen=True)
class _CornerGrid:
    width_axis: str
    height_axis: str
    height_is_row: bool
    width_spacing: float
    width_origin: float
    height_spacing: float
    height_origin: float
    height_size: int


def _resolve_corner_grid(feature: ViewFeature) -> _CornerGrid | None:
    height_axis = "z"
    if height_axis not in feature.axes:
        return None
    width_axis = next(axis_name for axis_name in feature.axes if axis_name != height_axis)
    height_is_row = feature.axes[1] == height_axis
    if height_is_row:
        width_spacing = float(feature.grid2d.spacing[1])
        width_origin = float(feature.grid2d.origin[1])
        height_spacing = float(feature.grid2d.spacing[0])
        height_origin = float(feature.grid2d.origin[0])
        height_size = int(feature.mask.shape[0])
    else:
        width_spacing = float(feature.grid2d.spacing[0])
        width_origin = float(feature.grid2d.origin[0])
        height_spacing = float(feature.grid2d.spacing[1])
        height_origin = float(feature.grid2d.origin[1])
        height_size = int(feature.mask.shape[1])
    return _CornerGrid(
        width_axis=width_axis,
        height_axis=height_axis,
        height_is_row=height_is_row,
        width_spacing=width_spacing,
        width_origin=width_origin,
        height_spacing=height_spacing,
        height_origin=height_origin,
        height_size=height_size,
    )


def _line_from_view(arr: np.ndarray, *, line_idx: int, height_is_row: bool) -> np.ndarray:
    return arr[line_idx, :] if height_is_row else arr[:, line_idx]


def _bottom_corners(feature: ViewFeature) -> tuple[dict[str, float], dict[str, object]]:
    grid = _resolve_corner_grid(feature)
    summary: dict[str, object] = {
        "status": "SKIPPED",
        "view_axes": list(feature.axes),
        "height_axis": "z",
        "width_axis": None,
        "skipped_reason": None,
    }
    if grid is None:
        summary["skipped_reason"] = "corner requires a cross-section view that includes z"
        return {}, summary
    summary["width_axis"] = grid.width_axis

    mask = np.asarray(feature.mask, dtype=bool)
    candidate_lines: list[tuple[float, int]] = []
    for line_idx in range(grid.height_size):
        line = _line_from_view(mask, line_idx=line_idx, height_is_row=grid.height_is_row)
        if np.any(line):
            height_nm = float(grid.height_origin + line_idx * grid.height_spacing)
            candidate_lines.append((height_nm, line_idx))
    if not candidate_lines:
        summary["skipped_reason"] = "no non-void pixels in projected view"
        return {}, summary

    bottom_height_nm, bottom_idx = min(candidate_lines, key=lambda item: item[0])
    bottom_line = _line_from_view(mask, line_idx=bottom_idx, height_is_row=grid.height_is_row)
    width_indices = np.flatnonzero(bottom_line)
    if width_indices.size == 0:
        summary["skipped_reason"] = "bottom line has no non-void pixels"
        return {}, summary

    left_idx = int(width_indices[0])
    right_idx = int(width_indices[-1])
    left_width_nm = float(grid.width_origin + left_idx * grid.width_spacing)
    right_width_nm = float(grid.width_origin + right_idx * grid.width_spacing)
    corners = {
        "left_width_nm": left_width_nm,
        "left_height_nm": bottom_height_nm,
        "right_width_nm": right_width_nm,
        "right_height_nm": bottom_height_nm,
    }
    summary.update(
        {
            "status": "OK",
            "bottom_height_nm": bottom_height_nm,
            "left_width_nm": left_width_nm,
            "right_width_nm": right_width_nm,
        }
    )
    return corners, summary


def _corner_distance(lhs: dict[str, float], rhs: dict[str, float], side: str) -> float:
    width_delta = float(lhs[f"{side}_width_nm"] - rhs[f"{side}_width_nm"])
    height_delta = float(lhs[f"{side}_height_nm"] - rhs[f"{side}_height_nm"])
    return float(np.hypot(width_delta, height_delta))


def compute_corner(
    sim: ViewFeature,
    target: ViewFeature,
    context: MetricContext,
) -> MetricComputation:
    del context
    sim_corners, sim_summary = _bottom_corners(sim)
    target_corners, target_summary = _bottom_corners(target)
    summary: dict[str, object] = {
        "metric": "corner",
        "status": "SKIPPED",
        "mode": "bottom_left_right_position",
        "view_axes": list(sim.axes),
        "sim": sim_summary,
        "target": target_summary,
        "skipped_reason": None,
    }
    if not sim_corners or not target_corners:
        summary["skipped_reason"] = (
            "simulation corners are unavailable"
            if not sim_corners
            else "target corners are unavailable"
        )
        return MetricComputation(
            name="corner",
            loss=0.0,
            value=0.0,
            status="SKIPPED",
            corner_summary=summary,
        )

    left_error_nm = _corner_distance(sim_corners, target_corners, "left")
    right_error_nm = _corner_distance(sim_corners, target_corners, "right")
    value = float(0.5 * (left_error_nm + right_error_nm))
    summary.update(
        {
            "status": "OK",
            "left_error_nm": left_error_nm,
            "right_error_nm": right_error_nm,
            "corner_loss_mean_nm": value,
            "sim_corners": sim_corners,
            "target_corners": target_corners,
        }
    )
    return MetricComputation(
        name="corner",
        loss=value,
        value=value,
        status="OK",
        corner_summary=summary,
    )
