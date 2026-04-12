from __future__ import annotations

import numpy as np

from wafergeo.compare.features import ViewFeature
from wafergeo.compare.metric_types import MetricComputation, MetricContext
from wafergeo.compare.sdf_helpers import (
    clipped_signed_distance_from_mask_2d,
    signed_distance_from_mask_2d,
    unsigned_distance_from_mask_2d,
)

NARROW_BAND_NM = 10.0


def _view_diagonal_nm(feature: ViewFeature) -> float:
    height = max(int(feature.label2d.shape[0]) - 1, 1)
    width = max(int(feature.label2d.shape[1]) - 1, 1)
    dy = height * float(feature.grid2d.spacing[0])
    dx = width * float(feature.grid2d.spacing[1])
    return float(max(np.hypot(dy, dx), 1.0))


def _has_open_contour(feature: ViewFeature) -> bool:
    return feature.source == "contour" and any(
        not contour.closed for contour in feature.contours.contours
    )


def _unsigned_boundary_distance(feature: ViewFeature) -> np.ndarray:
    spacing = (float(feature.grid2d.spacing[0]), float(feature.grid2d.spacing[1]))
    boundary = (
        np.asarray(feature.boundary_mask, dtype=bool)
        if feature.boundary_mask is not None and np.any(feature.boundary_mask)
        else np.asarray(feature.mask, dtype=bool)
    )
    return unsigned_distance_from_mask_2d(boundary, spacing)


def _mask_iou(lhs: np.ndarray, rhs: np.ndarray) -> float:
    a = np.asarray(lhs, dtype=bool)
    b = np.asarray(rhs, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    inter = int(np.logical_and(a, b).sum())
    return float(inter / union)


def _label_iou(lhs_label: np.ndarray, rhs_label: np.ndarray) -> float:
    a = np.asarray(lhs_label)
    b = np.asarray(rhs_label)
    labels = sorted(set(np.unique(a).tolist()).union(np.unique(b).tolist()))
    if not labels:
        return 1.0
    values: list[float] = []
    for label in labels:
        lhs_mask = a == label
        rhs_mask = b == label
        union = int(np.logical_or(lhs_mask, rhs_mask).sum())
        if union == 0:
            continue
        inter = int(np.logical_and(lhs_mask, rhs_mask).sum())
        values.append(float(inter / union))
    if not values:
        return 1.0
    return float(np.mean(values))


def _boundary_sdf_loss(sim: ViewFeature, target: ViewFeature) -> float | None:
    if sim.boundary_mask is None or target.boundary_mask is None:
        return None
    sim_boundary = np.asarray(sim.boundary_mask, dtype=bool)
    target_boundary = np.asarray(target.boundary_mask, dtype=bool)
    if not np.any(sim_boundary) or not np.any(target_boundary):
        return None

    spacing = (float(sim.grid2d.spacing[0]), float(sim.grid2d.spacing[1]))
    sim_sdf = signed_distance_from_mask_2d(sim_boundary, spacing)
    target_sdf = signed_distance_from_mask_2d(target_boundary, spacing)
    return float(np.mean(np.abs(sim_sdf - target_sdf)))


def _narrow_band_loss_from_sdf(
    sim_sdf: np.ndarray,
    target_sdf: np.ndarray,
    *,
    band_nm: float,
) -> tuple[float, int] | None:
    sim_arr = np.asarray(sim_sdf, dtype=np.float32)
    target_arr = np.asarray(target_sdf, dtype=np.float32)
    band = (np.abs(sim_arr) <= float(band_nm)) | (np.abs(target_arr) <= float(band_nm))
    count = int(band.sum())
    if count == 0:
        return None
    return float(np.mean(np.abs(sim_arr[band] - target_arr[band]))), count


def _narrow_band_sdf_loss(
    sim: ViewFeature,
    target: ViewFeature,
    *,
    band_nm: float = NARROW_BAND_NM,
) -> tuple[float, str, dict[str, object]]:
    spacing = (float(sim.grid2d.spacing[0]), float(sim.grid2d.spacing[1]))
    details: dict[str, object] = {
        "metric": "sdf_band",
        "band_nm": float(band_nm),
        "band_pixel_count": 0,
        "mode": "mask_sdf_band",
        "selected_loss_source": "mask_sdf_band",
    }
    if _has_open_contour(sim) or _has_open_contour(target):
        sim_distance = _unsigned_boundary_distance(sim)
        target_distance = _unsigned_boundary_distance(target)
        band = (sim_distance <= float(band_nm)) | (target_distance <= float(band_nm))
        count = int(band.sum())
        if count == 0:
            details.update(
                {
                    "mode": "open_contour_unsigned_distance_band",
                    "distance_semantics": "unsigned",
                    "selected_loss_source": "open_contour_unsigned_distance_band",
                    "skipped_reason": "narrow band has no pixels",
                }
            )
            return 0.0, "SKIPPED", details
        value = float(np.mean(np.abs(sim_distance[band] - target_distance[band])))
        details.update(
            {
                "mode": "open_contour_unsigned_distance_band",
                "distance_semantics": "unsigned",
                "selected_loss_source": "open_contour_unsigned_distance_band",
                "band_pixel_count": count,
                "unsigned_distance_band_loss_nm": value,
            }
        )
        return value, "OK", details
    if sim.source == "label_volume" and target.source == "label_volume":
        sim_boundary = (
            np.asarray(sim.boundary_mask, dtype=bool) if sim.boundary_mask is not None else None
        )
        target_boundary = (
            np.asarray(target.boundary_mask, dtype=bool)
            if target.boundary_mask is not None
            else None
        )
        if (
            sim_boundary is not None
            and target_boundary is not None
            and np.any(sim_boundary)
            and np.any(target_boundary)
        ):
            sim_boundary_sdf = signed_distance_from_mask_2d(sim_boundary, spacing)
            target_boundary_sdf = signed_distance_from_mask_2d(target_boundary, spacing)
            boundary_loss = _narrow_band_loss_from_sdf(
                sim_boundary_sdf,
                target_boundary_sdf,
                band_nm=band_nm,
            )
            if boundary_loss is not None:
                value, count = boundary_loss
                details.update(
                    {
                        "mode": "boundary_sdf_band",
                        "selected_loss_source": "boundary_sdf_band",
                        "band_pixel_count": count,
                        "boundary_sdf_band_loss_nm": value,
                    }
                )
                return value, "OK", details

    mask_loss = _narrow_band_loss_from_sdf(sim.sdf_nm, target.sdf_nm, band_nm=band_nm)
    if mask_loss is None:
        details["skipped_reason"] = "narrow band has no pixels"
        return 0.0, "SKIPPED", details
    value, count = mask_loss
    details.update(
        {
            "band_pixel_count": count,
            "mask_sdf_band_loss_nm": value,
        }
    )
    return value, "OK", details


def _sdf_loss(sim: ViewFeature, target: ViewFeature) -> tuple[float, dict[str, object]]:
    sim_labels = np.asarray(sim.label2d)
    target_labels = np.asarray(target.label2d)
    if _has_open_contour(sim) or _has_open_contour(target):
        sim_distance = _unsigned_boundary_distance(sim)
        target_distance = _unsigned_boundary_distance(target)
        value = float(np.mean(np.abs(sim_distance - target_distance)))
        return value, {
            "metric": "sdf",
            "mode": "open_contour_unsigned_distance",
            "distance_semantics": "unsigned",
            "selected_loss_source": "open_contour_unsigned_distance",
            "unsigned_distance_loss_nm": value,
        }
    if sim.source != "label_volume" or target.source != "label_volume":
        sdf_delta = sim.sdf_nm.astype(np.float32) - target.sdf_nm.astype(np.float32)
        value = float(np.mean(np.abs(sdf_delta)))
        return value, {
            "metric": "sdf",
            "mode": "mask_sdf",
            "selected_loss_source": "mask_sdf",
            "mask_sdf_loss_nm": value,
        }

    labels = sorted(set(np.unique(sim_labels).tolist()).union(np.unique(target_labels).tolist()))
    if set(labels).issubset({0, 1}):
        sdf_delta = sim.sdf_nm.astype(np.float32) - target.sdf_nm.astype(np.float32)
        value = float(np.mean(np.abs(sdf_delta)))
        return value, {
            "metric": "sdf",
            "mode": "binary_mask_sdf",
            "selected_loss_source": "mask_sdf",
            "mask_sdf_loss_nm": value,
        }

    spacing = (float(sim.grid2d.spacing[0]), float(sim.grid2d.spacing[1]))
    losses: list[float] = []
    per_label: list[dict[str, float | int]] = []
    for label in labels:
        sim_sdf = signed_distance_from_mask_2d(sim_labels == label, spacing)
        target_sdf = signed_distance_from_mask_2d(target_labels == label, spacing)
        label_loss = float(np.mean(np.abs(sim_sdf - target_sdf)))
        losses.append(label_loss)
        per_label.append({"label": int(label), "sdf_loss_nm": label_loss})
    label_loss = float(np.mean(losses)) if losses else 0.0
    boundary_loss = _boundary_sdf_loss(sim, target)
    if boundary_loss is None:
        return label_loss, {
            "metric": "sdf",
            "mode": "label_sdf",
            "selected_loss_source": "label_sdf",
            "label_sdf_loss_nm": label_loss,
            "boundary_sdf_loss_nm": None,
            "per_label": per_label,
        }
    value = max(label_loss, boundary_loss)
    return value, {
        "metric": "sdf",
        "mode": "label_and_boundary_sdf",
        "selected_loss_source": "boundary_sdf" if boundary_loss >= label_loss else "label_sdf",
        "label_sdf_loss_nm": label_loss,
        "boundary_sdf_loss_nm": boundary_loss,
        "per_label": per_label,
    }


def _material_sdf_loss(
    sim: ViewFeature,
    target: ViewFeature,
) -> tuple[float, str, dict[str, object]]:
    details: dict[str, object] = {
        "metric": "sdf_material",
        "mode": "per_material_sdf",
        "selected_material_ids": [],
        "per_material": [],
    }
    if sim.source != "label_volume" or target.source != "label_volume":
        details["skipped_reason"] = "sdf_material requires label-volume simulation and target"
        return 0.0, "SKIPPED", details

    sim_masks = sim.material_masks or {}
    target_masks = target.material_masks or {}
    labels = sorted(set(sim_masks).union(target_masks))
    mask_source = "projected_material_masks"
    if not labels:
        sim_labels = np.asarray(sim.label2d)
        target_labels = np.asarray(target.label2d)
        excluded = {int(sim.void_id), int(target.void_id)}
        label_ids = set(np.unique(sim_labels).tolist()).union(
            np.unique(target_labels).tolist()
        )
        labels = sorted(
            label
            for label in label_ids
            if int(label) not in excluded
        )
        sim_masks = {int(label): sim_labels == label for label in labels}
        target_masks = {int(label): target_labels == label for label in labels}
        mask_source = "topmost_label2d_fallback"
    details["selected_material_ids"] = [int(label) for label in labels]
    details["mask_source"] = mask_source
    if not labels:
        details["skipped_reason"] = "no non-void material labels in projected view"
        return 0.0, "SKIPPED", details

    spacing = (float(sim.grid2d.spacing[0]), float(sim.grid2d.spacing[1]))
    cap_nm = max(_view_diagonal_nm(sim), _view_diagonal_nm(target))
    empty = np.zeros(sim.mask.shape, dtype=bool)
    rows: list[dict[str, float | int]] = []
    losses: list[float] = []
    weights: list[int] = []
    for label in labels:
        sim_mask = np.asarray(sim_masks.get(int(label), empty), dtype=bool)
        target_mask = np.asarray(target_masks.get(int(label), empty), dtype=bool)
        sim_sdf = clipped_signed_distance_from_mask_2d(
            sim_mask,
            spacing,
            clip_nm=cap_nm,
        )
        target_sdf = clipped_signed_distance_from_mask_2d(
            target_mask,
            spacing,
            clip_nm=cap_nm,
        )
        loss = float(np.mean(np.abs(sim_sdf - target_sdf)))
        union_pixels = int(np.logical_or(sim_mask, target_mask).sum())
        losses.append(loss)
        weights.append(union_pixels)
        rows.append(
            {
                "material_id": int(label),
                "sdf_loss_nm": loss,
                "sim_pixels": int(sim_mask.sum()),
                "target_pixels": int(target_mask.sum()),
                "union_pixels": union_pixels,
            }
        )
    value = float(np.average(np.asarray(losses, dtype=np.float64), weights=np.asarray(weights)))
    details.update(
        {
            "sdf_material_loss_nm": value,
            "sdf_material_unweighted_mean_nm": float(np.mean(losses)),
            "aggregation": "union_pixel_weighted_mean",
            "per_material": rows,
            "distance_cap_nm": float(cap_nm),
        }
    )
    return value, "OK", details


def _feature_iou(sim: ViewFeature, target: ViewFeature) -> tuple[float, dict[str, object]]:
    if sim.source == "label_volume" and target.source == "label_volume":
        label_value = _label_iou(sim.label2d, target.label2d)
        if sim.boundary_mask is None or target.boundary_mask is None:
            return label_value, {
                "metric": "iou",
                "mode": "label_iou",
                "selected_value_source": "label_iou",
                "label_iou": label_value,
                "boundary_iou": None,
            }
        sim_boundary = np.asarray(sim.boundary_mask, dtype=bool)
        target_boundary = np.asarray(target.boundary_mask, dtype=bool)
        if not np.any(sim_boundary) or not np.any(target_boundary):
            return label_value, {
                "metric": "iou",
                "mode": "label_iou",
                "selected_value_source": "label_iou",
                "label_iou": label_value,
                "boundary_iou": None,
            }
        boundary_value = _mask_iou(sim_boundary, target_boundary)
        value = min(label_value, boundary_value)
        return value, {
            "metric": "iou",
            "mode": "label_and_boundary_iou",
            "selected_value_source": (
                "boundary_iou" if boundary_value <= label_value else "label_iou"
            ),
            "label_iou": label_value,
            "boundary_iou": boundary_value,
        }
    value = _mask_iou(sim.mask, target.mask)
    return value, {
        "metric": "iou",
        "mode": "mask_iou",
        "selected_value_source": "mask_iou",
        "mask_iou": value,
    }


def compute_sdf(
    sim: ViewFeature,
    target: ViewFeature,
    _context: MetricContext,
) -> MetricComputation:
    value, details = _sdf_loss(sim, target)
    return MetricComputation(name="sdf", loss=value, value=value, details=details)


def compute_sdf_band(
    sim: ViewFeature,
    target: ViewFeature,
    _context: MetricContext,
) -> MetricComputation:
    value, status, details = _narrow_band_sdf_loss(sim, target)
    return MetricComputation(
        name="sdf_band",
        loss=value,
        value=value,
        status=status,
        details=details,
    )


def compute_sdf_material(
    sim: ViewFeature,
    target: ViewFeature,
    _context: MetricContext,
) -> MetricComputation:
    value, status, details = _material_sdf_loss(sim, target)
    return MetricComputation(
        name="sdf_material",
        loss=value,
        value=value,
        status=status,
        details=details,
    )


def compute_iou(
    sim: ViewFeature,
    target: ViewFeature,
    _context: MetricContext,
) -> MetricComputation:
    if _has_open_contour(sim) or _has_open_contour(target):
        return MetricComputation(
            name="iou",
            loss=0.0,
            value=0.0,
            status="SKIPPED",
            details={
                "metric": "iou",
                "mode": "open_contour",
                "skipped_reason": "IoU is not defined for open contour targets",
            },
        )
    value, details = _feature_iou(sim, target)
    return MetricComputation(name="iou", loss=float(1.0 - value), value=value, details=details)
