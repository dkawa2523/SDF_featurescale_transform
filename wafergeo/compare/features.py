from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from wafergeo.compare.loader import ContourData, ContourItem, contour_data_to_json
from wafergeo.compare.sdf_helpers import (
    signed_distance_from_mask_2d,
    unsigned_distance_from_mask_2d,
)
from wafergeo.core.grid import GridSpec
from wafergeo.core.types import LabelVolume

AxisName = Literal["x", "y", "z"]


@dataclass(frozen=True)
class ViewFeature:
    mask: np.ndarray
    grid2d: GridSpec
    contours: ContourData
    sdf_nm: np.ndarray
    label2d: np.ndarray
    source: Literal["label_volume", "contour"] = "label_volume"
    axes: tuple[AxisName, AxisName] = ("x", "y")
    boundary_mask: np.ndarray | None = None
    void_id: int = 0
    material_masks: dict[int, np.ndarray] | None = None

def _project_mask(
    label: LabelVolume,
    *,
    axes: tuple[AxisName, AxisName],
    depth_axis: AxisName,
) -> tuple[np.ndarray, np.ndarray, GridSpec]:
    axis_to_dim = {"z": 0, "y": 1, "x": 2}
    depth_dim = axis_to_dim[depth_axis]
    void_id = int(label.material.void_id)
    solid3d = label.material_id != int(label.material.void_id)
    mask_any = np.any(solid3d, axis=depth_dim)

    remaining_axes = [axis for axis in ("z", "y", "x") if axis != depth_axis]
    target_order = [axes[1], axes[0]]
    transpose_order = [remaining_axes.index(axis) for axis in target_order]
    reversed_labels = np.flip(label.material_id, axis=depth_dim)
    reversed_solid = np.flip(solid3d, axis=depth_dim)
    first_solid = np.argmax(reversed_solid, axis=depth_dim)
    labels_work = np.take_along_axis(
        reversed_labels,
        np.expand_dims(first_solid, axis=depth_dim),
        axis=depth_dim,
    ).squeeze(axis=depth_dim)
    labels_work = np.where(mask_any, labels_work, void_id)
    label2d = np.transpose(labels_work, transpose_order).astype(np.int32, copy=False)

    mask = np.transpose(mask_any, transpose_order).astype(bool, copy=False)

    spacing_by_axis = {
        "z": float(label.grid.spacing[0]),
        "y": float(label.grid.spacing[1]),
        "x": float(label.grid.spacing[2]),
    }
    origin_by_axis = {
        "z": float(label.grid.origin[0]),
        "y": float(label.grid.origin[1]),
        "x": float(label.grid.origin[2]),
    }
    grid2d = GridSpec(
        dim=2,
        spacing=(spacing_by_axis[axes[1]], spacing_by_axis[axes[0]]),
        origin=(origin_by_axis[axes[1]], origin_by_axis[axes[0]]),
        axis_order="YX",
        sample_location="cell_center",
        units=label.grid.units,
    )
    return mask, label2d, grid2d


def _points_from_pixels(rows: np.ndarray, cols: np.ndarray, grid2d: GridSpec) -> np.ndarray:
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.float32)
    y = float(grid2d.origin[0]) + rows.astype(np.float32) * float(grid2d.spacing[0])
    x = float(grid2d.origin[1]) + cols.astype(np.float32) * float(grid2d.spacing[1])
    return np.column_stack([x, y]).astype(np.float32)


def _boundary_mask_from_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    neighbor_all = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return center & ~neighbor_all


def _boundary_points_from_mask(mask: np.ndarray, grid2d: GridSpec) -> np.ndarray:
    boundary = _boundary_mask_from_mask(mask)
    rows, cols = np.nonzero(boundary)
    return _points_from_pixels(rows, cols, grid2d)


def _material_boundary_mask_3d(label: LabelVolume) -> np.ndarray:
    labels = label.material_id
    solid = labels != int(label.material.void_id)
    boundary = np.zeros(labels.shape, dtype=bool)

    for axis in range(3):
        before = [slice(None), slice(None), slice(None)]
        after = [slice(None), slice(None), slice(None)]
        before[axis] = slice(0, -1)
        after[axis] = slice(1, None)
        before_t = tuple(before)
        after_t = tuple(after)
        edge = solid[before_t] & solid[after_t] & (labels[before_t] != labels[after_t])
        boundary[before_t] |= edge
        boundary[after_t] |= edge

    return boundary


def _project_3d_mask(
    mask_zyx: np.ndarray,
    *,
    axes: tuple[AxisName, AxisName],
    depth_axis: AxisName,
) -> np.ndarray:
    axis_to_dim = {"z": 0, "y": 1, "x": 2}
    depth_dim = axis_to_dim[depth_axis]
    mask_any = np.any(mask_zyx, axis=depth_dim)
    remaining_axes = [axis for axis in ("z", "y", "x") if axis != depth_axis]
    target_order = [axes[1], axes[0]]
    transpose_order = [remaining_axes.index(axis) for axis in target_order]
    return np.transpose(mask_any, transpose_order).astype(bool, copy=False)


def _project_material_masks(
    label: LabelVolume,
    *,
    axes: tuple[AxisName, AxisName],
    depth_axis: AxisName,
) -> dict[int, np.ndarray]:
    labels = label.material_id
    void_id = int(label.material.void_id)
    masks: dict[int, np.ndarray] = {}
    for material_id in sorted(int(v) for v in np.unique(labels)):
        if material_id == void_id:
            continue
        projected = _project_3d_mask(labels == material_id, axes=axes, depth_axis=depth_axis)
        if np.any(projected):
            masks[material_id] = projected
    return masks


def extract_view_feature(
    label: LabelVolume,
    *,
    axes: tuple[AxisName, AxisName] = ("x", "y"),
    depth_axis: AxisName = "z",
    contour_mode: Literal["material", "outer"] = "material",
) -> ViewFeature:
    mask, label2d, grid2d = _project_mask(label, axes=axes, depth_axis=depth_axis)
    material_masks = _project_material_masks(label, axes=axes, depth_axis=depth_axis)
    if contour_mode == "material":
        boundary_mask = _project_3d_mask(
            _material_boundary_mask_3d(label),
            axes=axes,
            depth_axis=depth_axis,
        )
        rows, cols = np.nonzero(boundary_mask)
        points = _points_from_pixels(rows, cols, grid2d)
        contour_id = "material_boundary"
        contour_label = "material_boundary"
        contour_meta: dict[str, object] = {
            "source": "label_volume",
            "geometry": "point_cloud",
        }
        closed = False
    else:
        boundary_mask = np.zeros(mask.shape, dtype=bool)
        points = np.empty((0, 2), dtype=np.float32)
        contour_id = "simulation_boundary"
        contour_label = "simulation_union"
        contour_meta = {"source": "label_volume", "geometry": "polyline"}
        closed = True

    if contour_mode == "outer" or points.shape[0] < 2:
        boundary_mask = _boundary_mask_from_mask(mask)
        points = _boundary_points_from_mask(mask, grid2d)
        contour_id = "simulation_boundary"
        contour_label = "simulation_union"
        contour_meta = {
            "source": "label_volume",
            "geometry": "polyline",
            "fallback": "outer" if contour_mode == "material" else False,
        }
        closed = True
    if points.shape[0] < 2:
        points = np.array(
            [
                [grid2d.origin[1], grid2d.origin[0]],
                [grid2d.origin[1], grid2d.origin[0]],
            ],
            dtype=np.float32,
        )
    contour = ContourItem(
        contour_id=contour_id,
        label=contour_label,
        closed=closed,
        points_xy_nm=points,
        meta=contour_meta,
    )
    sdf = signed_distance_from_mask_2d(mask, (float(grid2d.spacing[0]), float(grid2d.spacing[1])))
    return ViewFeature(
        mask=mask,
        grid2d=grid2d,
        contours=ContourData(units="nm", contours=[contour], meta={"source": "simulation"}),
        sdf_nm=sdf,
        label2d=label2d,
        source="label_volume",
        axes=axes,
        boundary_mask=boundary_mask,
        void_id=int(label.material.void_id),
        material_masks=material_masks,
    )


def _points_to_pixel(points_xy: np.ndarray, grid2d: GridSpec) -> np.ndarray:
    cols = (points_xy[:, 0] - float(grid2d.origin[1])) / float(grid2d.spacing[1])
    rows = (points_xy[:, 1] - float(grid2d.origin[0])) / float(grid2d.spacing[0])
    return np.column_stack([rows, cols]).astype(np.float32)


def _polygon_mask(points_xy: np.ndarray, grid2d: GridSpec, shape_yx: tuple[int, int]) -> np.ndarray:
    pix = _points_to_pixel(points_xy, grid2d)
    rows, cols = np.indices(shape_yx)
    y = rows.astype(np.float32)
    x = cols.astype(np.float32)
    inside = np.zeros(shape_yx, dtype=bool)
    poly_y = pix[:, 0]
    poly_x = pix[:, 1]
    n = len(poly_x)
    j = n - 1
    for i in range(n):
        yi = poly_y[i]
        yj = poly_y[j]
        xi = poly_x[i]
        xj = poly_x[j]
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) if abs(yj - yi) > 1e-12 else 1e-12) + xi
        )
        inside ^= crosses
        j = i
    return inside


def _draw_polyline_mask(
    points_xy: np.ndarray,
    grid2d: GridSpec,
    shape_yx: tuple[int, int],
) -> np.ndarray:
    pix = _points_to_pixel(points_xy, grid2d)
    out = np.zeros(shape_yx, dtype=bool)
    for start, stop in zip(pix[:-1], pix[1:], strict=False):
        steps = max(int(np.ceil(np.linalg.norm(stop - start))) + 1, 2)
        rr = np.linspace(start[0], stop[0], steps)
        cc = np.linspace(start[1], stop[1], steps)
        r_idx = np.clip(np.rint(rr).astype(int), 0, shape_yx[0] - 1)
        c_idx = np.clip(np.rint(cc).astype(int), 0, shape_yx[1] - 1)
        out[r_idx, c_idx] = True
    return out


def contour_feature_on_grid(
    data: ContourData,
    grid2d: GridSpec,
    shape_yx: tuple[int, int],
    *,
    axes: tuple[AxisName, AxisName] = ("x", "y"),
) -> ViewFeature:
    mask = np.zeros(shape_yx, dtype=bool)
    boundary_mask = np.zeros(shape_yx, dtype=bool)
    has_open_contour = any(not contour.closed for contour in data.contours)
    for contour in data.contours:
        if contour.closed and contour.points_xy_nm.shape[0] >= 3:
            mask |= _polygon_mask(contour.points_xy_nm, grid2d, shape_yx)
        boundary_mask |= _draw_polyline_mask(contour.points_xy_nm, grid2d, shape_yx)
    if not np.any(mask):
        mask = boundary_mask
    spacing_yx = (float(grid2d.spacing[0]), float(grid2d.spacing[1]))
    sdf = (
        unsigned_distance_from_mask_2d(boundary_mask, spacing_yx)
        if has_open_contour
        else signed_distance_from_mask_2d(mask, spacing_yx)
    )
    return ViewFeature(
        mask=mask,
        grid2d=grid2d,
        contours=data,
        sdf_nm=sdf,
        label2d=mask.astype(np.int32),
        source="contour",
        axes=axes,
        boundary_mask=boundary_mask,
        void_id=0,
        material_masks=None,
    )


def write_view_features(
    *,
    feature: ViewFeature,
    output_dir: Path,
    prefix: str,
    include_sdf: bool,
    include_contour: bool,
    include_slice: bool = False,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    if include_sdf:
        path = output_dir / f"{prefix}_sdf.npz"
        np.savez(
            path,
            sdf_nm=feature.sdf_nm,
            mask=feature.mask.astype(np.uint8),
            spacing=np.asarray(feature.grid2d.spacing, dtype=np.float32),
            origin=np.asarray(feature.grid2d.origin, dtype=np.float32),
        )
        written["sdf"] = str(path.name)
    if include_contour:
        path = output_dir / f"{prefix}_contours.json"
        path.write_text(
            json.dumps(contour_data_to_json(feature.contours), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        written["contour"] = str(path.name)
    if include_slice:
        path = output_dir / f"{prefix}_slice.npy"
        np.save(path, feature.label2d)
        written["slice"] = str(path.name)
    return written
