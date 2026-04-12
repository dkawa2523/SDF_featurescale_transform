from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wafergeo.compare.features import ViewFeature
from wafergeo.compare.metric_types import MetricComputation, MetricContext


def _line_from_view(arr: np.ndarray, *, line_idx: int, height_is_row: bool) -> np.ndarray:
    return arr[line_idx, :] if height_is_row else arr[:, line_idx]


@dataclass(frozen=True)
class _ResolvedGauge:
    axis: str
    height_axis: str
    center_nm: float
    height_range_nm: tuple[float, float] | None
    height_is_row: bool
    horizontal_spacing: float
    horizontal_origin: float
    height_spacing: float
    height_origin: float
    line_count: int
    source: str

    def to_summary(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "height_axis": self.height_axis,
            "center_nm": self.center_nm,
            "height_range_nm": (
                None
                if self.height_range_nm is None
                else [float(self.height_range_nm[0]), float(self.height_range_nm[1])]
            ),
            "source": self.source,
        }


def _resolve_gauge(feature: ViewFeature, context: MetricContext) -> _ResolvedGauge | None:
    height_axis = context.cd_gauge_height_axis or "z"
    if height_axis not in feature.axes:
        return None
    axis = context.cd_gauge_axis
    if axis is None:
        axis = next(axis_name for axis_name in feature.axes if axis_name != height_axis)
    if axis not in feature.axes or axis == height_axis:
        return None

    height_is_row = feature.axes[1] == height_axis
    if height_is_row:
        horizontal_spacing = float(feature.grid2d.spacing[1])
        horizontal_origin = float(feature.grid2d.origin[1])
        horizontal_size = int(feature.mask.shape[1])
        height_spacing = float(feature.grid2d.spacing[0])
        height_origin = float(feature.grid2d.origin[0])
        line_count = int(feature.mask.shape[0])
    else:
        horizontal_spacing = float(feature.grid2d.spacing[0])
        horizontal_origin = float(feature.grid2d.origin[0])
        horizontal_size = int(feature.mask.shape[0])
        height_spacing = float(feature.grid2d.spacing[1])
        height_origin = float(feature.grid2d.origin[1])
        line_count = int(feature.mask.shape[1])

    default_center = horizontal_origin + ((horizontal_size - 1) / 2.0) * horizontal_spacing
    center_nm = (
        float(default_center)
        if context.cd_gauge_center_nm is None
        else float(context.cd_gauge_center_nm)
    )
    return _ResolvedGauge(
        axis=axis,
        height_axis=height_axis,
        center_nm=center_nm,
        height_range_nm=context.cd_gauge_height_range_nm,
        height_is_row=height_is_row,
        horizontal_spacing=horizontal_spacing,
        horizontal_origin=horizontal_origin,
        height_spacing=height_spacing,
        height_origin=height_origin,
        line_count=line_count,
        source="default" if context.cd_gauge_height_axis is None else "yaml",
    )


def _nearest_boundary_width_nm(
    *,
    boundary_line: np.ndarray | None,
    label_line: np.ndarray,
    mask_line: np.ndarray,
    center_nm: float,
    horizontal_origin: float,
    horizontal_spacing: float,
    allow_mask_fallback: bool,
) -> tuple[float, float, float] | None:
    candidate_positions: list[float] = []
    if boundary_line is not None:
        boundary_idx = np.flatnonzero(np.asarray(boundary_line, dtype=bool))
        candidate_positions.extend(
            float(horizontal_origin + idx * horizontal_spacing) for idx in boundary_idx
        )

    if len(candidate_positions) < 2:
        labels = np.asarray(label_line)
        transition_idx = np.flatnonzero(labels[:-1] != labels[1:])
        candidate_positions = [
            float(horizontal_origin + (idx + 0.5) * horizontal_spacing)
            for idx in transition_idx
        ]

    if len(candidate_positions) < 2 and allow_mask_fallback:
        present_idx = np.flatnonzero(np.asarray(mask_line, dtype=bool))
        candidate_positions = [
            float(horizontal_origin + idx * horizontal_spacing) for idx in present_idx
        ]

    left = [value for value in candidate_positions if value < center_nm]
    right = [value for value in candidate_positions if value > center_nm]
    if not left or not right:
        return None
    left_nm = max(left)
    right_nm = min(right)
    if left_nm == right_nm:
        return None
    return left_nm, right_nm, abs(right_nm - left_nm)


def _transition_positions_nm(
    *,
    boundary_line: np.ndarray | None,
    label_line: np.ndarray,
    mask_line: np.ndarray,
    horizontal_origin: float,
    horizontal_spacing: float,
) -> np.ndarray:
    if boundary_line is not None:
        boundary_idx = np.flatnonzero(np.asarray(boundary_line, dtype=bool))
        if boundary_idx.size:
            return (
                float(horizontal_origin)
                + boundary_idx.astype(np.float32) * float(horizontal_spacing)
            ).astype(np.float32, copy=False)

    labels = np.asarray(label_line)
    mask = np.asarray(mask_line, dtype=bool)
    if labels.size < 2:
        return np.empty((0,), dtype=np.float32)
    left_solid = mask[:-1]
    right_solid = mask[1:]
    label_changes = labels[:-1] != labels[1:]
    solid_edges = left_solid != right_solid
    material_edges = left_solid & right_solid & label_changes
    edge_idx = np.flatnonzero(solid_edges | material_edges)
    if edge_idx.size == 0:
        return np.empty((0,), dtype=np.float32)
    return (
        float(horizontal_origin) + (edge_idx.astype(np.float32) + 0.5) * float(horizontal_spacing)
    ).astype(np.float32, copy=False)


def _symmetric_1d_nearest_loss(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    penalty_nm: float,
) -> float | None:
    if lhs.size == 0 and rhs.size == 0:
        return None
    if lhs.size == 0 or rhs.size == 0:
        return float(penalty_nm)
    lhs_sorted = np.sort(lhs.astype(np.float32, copy=False))
    rhs_sorted = np.sort(rhs.astype(np.float32, copy=False))

    def nearest(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        insertion = np.searchsorted(dst, src)
        right_idx = np.clip(insertion, 0, len(dst) - 1)
        left_idx = np.clip(insertion - 1, 0, len(dst) - 1)
        right_dist = np.abs(src - dst[right_idx])
        left_dist = np.abs(src - dst[left_idx])
        return np.minimum(left_dist, right_dist)

    lhs_loss = nearest(lhs_sorted, rhs_sorted)
    rhs_loss = nearest(rhs_sorted, lhs_sorted)
    return float(0.5 * (float(np.mean(lhs_loss)) + float(np.mean(rhs_loss))))


def _transition_profile_loss(
    sim: ViewFeature,
    target: ViewFeature,
    *,
    context: MetricContext,
) -> tuple[float | None, dict[str, object]]:
    summary: dict[str, object] = {
        "transition_profile_rows": 0,
        "transition_loss_mean_nm": None,
        "transition_loss_max_nm": None,
        "transition_penalty_nm": None,
    }
    if sim.source != "label_volume" or target.source != "label_volume":
        return None, summary
    if context.cd_material_ids is not None:
        return None, summary
    if (
        context.cd_gauge_axis is not None
        or context.cd_gauge_height_axis is not None
        or context.cd_gauge_center_nm is not None
        or context.cd_gauge_height_range_nm is not None
    ):
        return None, summary

    gauge = _resolve_gauge(sim, context)
    target_gauge = _resolve_gauge(target, context)
    if gauge is None or target_gauge is None:
        return None, summary

    line_count = min(gauge.line_count, target_gauge.line_count)
    horizontal_span = abs(gauge.horizontal_spacing) * max(int(sim.mask.shape[1]) - 1, 1)
    if not gauge.height_is_row:
        horizontal_span = abs(gauge.horizontal_spacing) * max(int(sim.mask.shape[0]) - 1, 1)
    summary["transition_penalty_nm"] = float(horizontal_span)

    losses: list[float] = []
    for line_idx in range(line_count):
        sim_positions = _transition_positions_nm(
            boundary_line=(
                _line_from_view(
                    sim.boundary_mask,
                    line_idx=line_idx,
                    height_is_row=gauge.height_is_row,
                )
                if sim.boundary_mask is not None
                else None
            ),
            label_line=_line_from_view(
                sim.label2d,
                line_idx=line_idx,
                height_is_row=gauge.height_is_row,
            ),
            mask_line=_line_from_view(
                sim.mask,
                line_idx=line_idx,
                height_is_row=gauge.height_is_row,
            ),
            horizontal_origin=gauge.horizontal_origin,
            horizontal_spacing=gauge.horizontal_spacing,
        )
        target_positions = _transition_positions_nm(
            boundary_line=(
                _line_from_view(
                    target.boundary_mask,
                    line_idx=line_idx,
                    height_is_row=target_gauge.height_is_row,
                )
                if target.boundary_mask is not None
                else None
            ),
            label_line=_line_from_view(
                target.label2d,
                line_idx=line_idx,
                height_is_row=target_gauge.height_is_row,
            ),
            mask_line=_line_from_view(
                target.mask,
                line_idx=line_idx,
                height_is_row=target_gauge.height_is_row,
            ),
            horizontal_origin=target_gauge.horizontal_origin,
            horizontal_spacing=target_gauge.horizontal_spacing,
        )
        loss = _symmetric_1d_nearest_loss(
            sim_positions,
            target_positions,
            penalty_nm=float(horizontal_span),
        )
        if loss is not None:
            losses.append(loss)

    if not losses:
        return None, summary
    value = float(np.mean(losses))
    summary.update(
        {
            "transition_profile_rows": len(losses),
            "transition_loss_mean_nm": value,
            "transition_loss_max_nm": float(np.max(losses)),
        }
    )
    return value, summary


def _cd_width_profile(
    feature: ViewFeature,
    *,
    material_ids: tuple[int, ...] | None,
    context: MetricContext,
) -> tuple[list[dict[str, float]], dict[str, object] | None]:
    gauge = _resolve_gauge(feature, context)
    if gauge is None:
        return [], None

    label2d = np.asarray(feature.label2d)
    if material_ids is not None and feature.source == "label_volume":
        mask = np.isin(label2d, np.asarray(material_ids, dtype=label2d.dtype))
        cd_labels = mask.astype(np.int8, copy=False)
        boundary_mask = None
        allow_mask_fallback = True
    else:
        mask = np.asarray(feature.mask, dtype=bool)
        cd_labels = label2d
        boundary_mask = (
            np.asarray(feature.boundary_mask, dtype=bool)
            if feature.boundary_mask is not None and np.any(feature.boundary_mask)
            else None
        )
        allow_mask_fallback = feature.source != "label_volume"

    height_is_row = gauge.height_is_row
    line_count = gauge.line_count
    center_nm = gauge.center_nm
    horizontal_origin = gauge.horizontal_origin
    horizontal_spacing = gauge.horizontal_spacing
    height_origin = gauge.height_origin
    height_spacing = gauge.height_spacing
    height_range = gauge.height_range_nm
    rows: list[dict[str, float]] = []
    for line_idx in range(line_count):
        height_nm = height_origin + line_idx * height_spacing
        if height_range is not None:
            start_nm, stop_nm = height_range
            if height_nm < float(start_nm) or height_nm > float(stop_nm):
                continue
        boundary_line = (
            _line_from_view(boundary_mask, line_idx=line_idx, height_is_row=height_is_row)
            if boundary_mask is not None
            else None
        )
        width = _nearest_boundary_width_nm(
            boundary_line=boundary_line,
            label_line=_line_from_view(cd_labels, line_idx=line_idx, height_is_row=height_is_row),
            mask_line=_line_from_view(mask, line_idx=line_idx, height_is_row=height_is_row),
            center_nm=center_nm,
            horizontal_origin=horizontal_origin,
            horizontal_spacing=horizontal_spacing,
            allow_mask_fallback=allow_mask_fallback,
        )
        if width is None:
            continue
        left_nm, right_nm, width_nm = width
        rows.append(
            {
                "z_nm": float(height_nm),
                "left_nm": float(left_nm),
                "right_nm": float(right_nm),
                "width_nm": float(width_nm),
            }
        )
    return rows, {
        **gauge.to_summary(),
    }


def _cd_profile_loss(
    sim: ViewFeature,
    target: ViewFeature,
    *,
    material_ids: tuple[int, ...] | None,
    context: MetricContext,
) -> tuple[float, list[dict[str, float]], str, dict[str, object]]:
    sim_profile, gauge = _cd_width_profile(sim, material_ids=material_ids, context=context)
    target_profile, target_gauge = _cd_width_profile(
        target,
        material_ids=material_ids,
        context=context,
    )
    if gauge is None:
        gauge = target_gauge
    summary: dict[str, object] = {
        "metric": "cd",
        "status": "SKIPPED",
        "profile_mode": "gauge_width",
        "selected_material_ids": list(material_ids) if material_ids is not None else None,
        "view_axes": list(sim.axes),
        "gauge": gauge,
        "sim_profile_rows": len(sim_profile),
        "target_profile_rows": len(target_profile),
        "shared_height_count": 0,
        "cd_profile_rows": 0,
        "skipped_reason": None,
        "cd_loss_mean_nm": None,
        "edge_loss_mean_nm": None,
        "edge_loss_max_nm": None,
        "width_abs_diff_mean_nm": None,
        "width_abs_diff_max_nm": None,
    }
    transition_value, transition_summary = _transition_profile_loss(sim, target, context=context)
    summary.update(transition_summary)
    if not sim_profile or not target_profile:
        if transition_value is not None:
            summary.update(
                {
                    "status": "OK",
                    "profile_mode": "auto_material_boundary",
                    "cd_loss_mean_nm": transition_value,
                    "skipped_reason": None,
                }
            )
            return transition_value, [], "OK", summary
        summary["skipped_reason"] = (
            "simulation CD profile is empty"
            if not sim_profile
            else "target CD profile is empty"
        )
        return 0.0, [], "SKIPPED", summary

    sim_by_z = {round(row["z_nm"], 6): row for row in sim_profile}
    target_by_z = {round(row["z_nm"], 6): row for row in target_profile}
    shared_z = sorted(set(sim_by_z).intersection(target_by_z))
    summary["shared_height_count"] = len(shared_z)
    if not shared_z:
        summary["skipped_reason"] = "simulation and target have no shared height samples"
        return 0.0, [], "SKIPPED", summary

    profile_rows: list[dict[str, float]] = []
    diffs: list[float] = []
    width_diffs: list[float] = []
    for z_key in shared_z:
        sim_row = sim_by_z[z_key]
        target_row = target_by_z[z_key]
        diff = float(sim_row["width_nm"] - target_row["width_nm"])
        left_diff = float(sim_row["left_nm"] - target_row["left_nm"])
        right_diff = float(sim_row["right_nm"] - target_row["right_nm"])
        edge_loss = 0.5 * (abs(left_diff) + abs(right_diff))
        diffs.append(edge_loss)
        width_diffs.append(abs(diff))
        profile_rows.append(
            {
                "z_nm": float(z_key),
                "sim_width_nm": float(sim_row["width_nm"]),
                "target_width_nm": float(target_row["width_nm"]),
                "diff_nm": diff,
                "abs_diff_nm": abs(diff),
                "left_diff_nm": left_diff,
                "right_diff_nm": right_diff,
                "left_abs_diff_nm": abs(left_diff),
                "right_abs_diff_nm": abs(right_diff),
                "edge_loss_nm": edge_loss,
                "sim_left_nm": float(sim_row["left_nm"]),
                "sim_right_nm": float(sim_row["right_nm"]),
                "target_left_nm": float(target_row["left_nm"]),
                "target_right_nm": float(target_row["right_nm"]),
            }
        )
    width_value = float(np.mean(diffs))
    value = width_value if transition_value is None else max(width_value, transition_value)
    summary.update(
        {
            "status": "OK",
            "profile_mode": (
                "auto_material_boundary"
                if transition_value is not None
                else "gauge_width"
            ),
            "cd_profile_rows": len(profile_rows),
            "cd_loss_mean_nm": value,
            "edge_loss_mean_nm": width_value,
            "edge_loss_max_nm": float(np.max(diffs)),
            "width_abs_diff_mean_nm": float(np.mean(width_diffs)),
            "width_abs_diff_max_nm": float(np.max(width_diffs)),
            "gauge_width_loss_mean_nm": width_value,
        }
    )
    return value, profile_rows, "OK", summary


def compute_cd(sim: ViewFeature, target: ViewFeature, context: MetricContext) -> MetricComputation:
    value, profile, status, summary = _cd_profile_loss(
        sim,
        target,
        material_ids=context.cd_material_ids,
        context=context,
    )
    return MetricComputation(
        name="cd",
        loss=value,
        value=value,
        status=status,
        cd_profile=profile,
        cd_profile_summary=summary,
    )
