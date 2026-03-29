from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import isfinite
from typing import Literal

import numpy as np

from wafergeo.core.grid import GridSpec
from wafergeo.core.hashing import hash_config
from wafergeo.core.meta import Meta
from wafergeo.core.types import LabelVolume, MaterialSpec
from wafergeo.io.vti_reader import (
    ArrayLocation,
    FlatArrayLayout,
    RawVtiImage,
    infer_flat_array_layout,
)
from wafergeo.label.errors import (
    InvalidArrayShapeError,
    InvalidGridMetaError,
    MissingLabelArrayError,
    PointToCellConversionError,
    UnknownMaterialIdError,
)
from wafergeo.label.qa import LabelQA, compute_label_qa

UnknownLabelPolicy = Literal["error", "map_to_void"]
PointToCellPolicy = Literal["majority", "majority_nearest_tie", "nearest", "error"]
MaskMergePolicy = Literal["priority", "error_on_conflict"]
MajorityTieBreaker = Literal["priority", "smallest", "nearest_anchor"]


@dataclass(frozen=True)
class LabelNormalizeConfig:
    label_array_candidates: list[str] = field(
        default_factory=lambda: ["material_id", "MaterialId", "label", "labels"]
    )
    prefer_cell_data: bool = True
    unknown_label_policy: UnknownLabelPolicy = "error"
    unknown_to_void_id: int = 0
    remap_ids: dict[int, int] = field(default_factory=dict)
    point_to_cell_policy: PointToCellPolicy = "nearest"
    mask_merge_policy: MaskMergePolicy = "priority"
    force_units: str | None = "nm"
    spacing_override_xyz: tuple[float, float, float] | None = None
    origin_override_xyz: tuple[float, float, float] | None = None
    enforce_cell_center: bool = True
    flat_array_layout: FlatArrayLayout = "auto"
    schema_version: str = "label/v1"
    profile_id: str = "ingest_label_v1"


@dataclass(frozen=True)
class SelectedLabelSource:
    kind: Literal["scalar", "masks"]
    name: str | None = None
    mask_names: dict[int, str] = field(default_factory=dict)


def select_label_source(
    raw: RawVtiImage,
    materials: MaterialSpec,
    config: LabelNormalizeConfig,
) -> SelectedLabelSource:
    for candidate in config.label_array_candidates:
        if candidate in raw.arrays:
            return SelectedLabelSource(kind="scalar", name=candidate)

    mask_names: dict[int, str] = {}
    for material_id, material_name in zip(materials.ids, materials.names, strict=True):
        key = f"mask_{material_name}"
        if key in raw.arrays:
            mask_names[material_id] = key

    if mask_names:
        return SelectedLabelSource(kind="masks", mask_names=mask_names)

    raise MissingLabelArrayError(
        "No label source found. "
        f"candidates={config.label_array_candidates}, available={sorted(raw.arrays.keys())}"
    )


def to_zyx(
    raw_array: np.ndarray,
    dims_xyz: tuple[int, int, int],
    location: ArrayLocation,
    *,
    flat_layout: FlatArrayLayout = "auto",
) -> np.ndarray:
    array = np.asarray(raw_array)
    nx, ny, nz = (int(v) for v in dims_xyz)
    point_shape_xyz = (nx, ny, nz)
    cell_shape_xyz = (max(nx - 1, 1), max(ny - 1, 1), max(nz - 1, 1))

    if location == "point":
        candidates = [point_shape_xyz]
    else:
        candidates = [point_shape_xyz, cell_shape_xyz]

    return _coerce_to_zyx(array, candidates, flat_layout=flat_layout)


def point_to_cell(
    point_labels_zyx: np.ndarray,
    *,
    policy: PointToCellPolicy,
    materials: MaterialSpec,
) -> np.ndarray:
    priority_map = {
        int(mid): int(prio)
        for mid, prio in zip(materials.ids, materials.priority, strict=True)
    }
    return convert_point_labels_to_cell_zyx(
        point_labels_zyx,
        policy=policy,
        majority_tie_breaker="priority",
        priority_map=priority_map,
    )


def convert_point_labels_to_cell_zyx(
    point_labels_zyx: np.ndarray,
    *,
    policy: PointToCellPolicy,
    majority_tie_breaker: MajorityTieBreaker = "smallest",
    priority_map: Mapping[int, int] | None = None,
) -> np.ndarray:
    """Convert point-sampled label volume to cell-sampled canonical ZYX labels."""
    if policy == "error":
        raise PointToCellConversionError("PointData input requires point_to_cell_policy != 'error'")

    if point_labels_zyx.ndim != 3:
        raise InvalidArrayShapeError(
            f"PointData labels must be 3D after ZYX conversion, got ndim={point_labels_zyx.ndim}"
        )

    z_size, y_size, x_size = point_labels_zyx.shape
    cell_shape = (max(z_size - 1, 1), max(y_size - 1, 1), max(x_size - 1, 1))
    if policy == "nearest":
        return point_labels_zyx[: cell_shape[0], : cell_shape[1], : cell_shape[2]].copy()
    if policy == "majority_nearest_tie":
        return _point_to_cell_majority_zyx(point_labels_zyx, tie_breaker="nearest_anchor")

    tie_breaker: MajorityTieBreaker = majority_tie_breaker
    if tie_breaker == "priority" and priority_map is None:
        tie_breaker = "smallest"
    return _point_to_cell_majority_zyx(
        point_labels_zyx,
        tie_breaker=tie_breaker,
        priority_map=priority_map,
    )


def _point_to_cell_majority_zyx(
    point_labels_zyx: np.ndarray,
    *,
    tie_breaker: Literal["priority", "smallest", "nearest_anchor"] = "smallest",
    priority_map: Mapping[int, int] | None = None,
) -> np.ndarray:
    z_size, y_size, x_size = point_labels_zyx.shape
    zc, yc, xc = max(z_size - 1, 1), max(y_size - 1, 1), max(x_size - 1, 1)
    z0 = slice(0, zc)
    z1 = slice(0, zc) if z_size == 1 else slice(1, zc + 1)
    y0 = slice(0, yc)
    y1 = slice(0, yc) if y_size == 1 else slice(1, yc + 1)
    x0 = slice(0, xc)
    x1 = slice(0, xc) if x_size == 1 else slice(1, xc + 1)

    neighbors = np.stack(
        [
            point_labels_zyx[z0, y0, x0],
            point_labels_zyx[z0, y0, x1],
            point_labels_zyx[z0, y1, x0],
            point_labels_zyx[z0, y1, x1],
            point_labels_zyx[z1, y0, x0],
            point_labels_zyx[z1, y0, x1],
            point_labels_zyx[z1, y1, x0],
            point_labels_zyx[z1, y1, x1],
        ],
        axis=0,
    )
    unique_ids = np.sort(np.unique(neighbors).astype(np.int32))
    best_count = np.full((zc, yc, xc), -1, dtype=np.int16)
    best_id = np.full((zc, yc, xc), np.iinfo(np.int32).max, dtype=np.int32)
    best_prio = np.full((zc, yc, xc), -10_000, dtype=np.int32)
    nearest_anchor = neighbors[0].astype(np.int32, copy=False)
    prio_map = priority_map or {}

    for material_id in unique_ids:
        count = np.asarray(
            np.sum(neighbors == material_id, axis=0, dtype=np.int16),
            dtype=np.int16,
        )
        better = count > best_count
        tie = count == best_count
        if tie_breaker == "priority":
            prio = int(prio_map.get(int(material_id), -10_000))
            better_prio = tie & (prio > best_prio)
            same_prio_small = tie & (prio == best_prio) & (material_id < best_id)
            update = better | better_prio | same_prio_small
        elif tie_breaker == "nearest_anchor":
            update = better | (tie & (material_id == nearest_anchor))
            prio = -10_000
        else:
            update = better | (tie & (material_id < best_id))
            prio = -10_000

        best_count[update] = count[update]
        best_id[update] = int(material_id)
        if tie_breaker == "priority":
            best_prio[update] = prio
    return best_id.astype(np.int64, copy=False)


def merge_material_masks(
    masks_by_id: dict[int, np.ndarray],
    materials: MaterialSpec,
    *,
    policy: MaskMergePolicy,
) -> tuple[np.ndarray, int]:
    if not masks_by_id:
        raise MissingLabelArrayError("No mask arrays found for mask merge")

    shapes = {mask.shape for mask in masks_by_id.values()}
    if len(shapes) != 1:
        raise InvalidArrayShapeError(f"Mask arrays must share one shape, got {sorted(shapes)}")

    shape = next(iter(shapes))
    void_id = materials.void_id
    merged = np.full(shape, void_id, dtype=np.int64)

    ids = list(masks_by_id.keys())
    stack = np.stack([masks_by_id[mid].astype(bool, copy=False) for mid in ids], axis=0)
    conflict_mask = np.sum(stack, axis=0) > 1
    conflict_count = int(np.sum(conflict_mask))

    if policy == "error_on_conflict" and conflict_count > 0:
        raise InvalidArrayShapeError(f"Mask conflict detected at {conflict_count} voxels")

    priority_map = {mid: prio for mid, prio in zip(materials.ids, materials.priority, strict=True)}
    sorted_ids = sorted(ids, key=lambda mid: (priority_map.get(mid, -10_000), -mid), reverse=True)
    for material_id in sorted_ids:
        current = masks_by_id[material_id].astype(bool, copy=False)
        fill_mask = current & (merged == void_id)
        merged[fill_mask] = material_id

    return merged, conflict_count


def apply_remap_ids(labels_zyx: np.ndarray, remap_ids: dict[int, int]) -> tuple[np.ndarray, bool]:
    if not remap_ids:
        return labels_zyx, False

    output = labels_zyx.copy()
    changed = False
    for src, dst in remap_ids.items():
        src_mask = output == int(src)
        if np.any(src_mask):
            output[src_mask] = int(dst)
            changed = True
    return output, changed


def apply_unknown_policy(
    labels_zyx: np.ndarray,
    materials: MaterialSpec,
    *,
    policy: UnknownLabelPolicy,
    unknown_to_void_id: int,
) -> tuple[np.ndarray, int, list[int]]:
    known_mask = np.isin(labels_zyx, np.asarray(materials.ids, dtype=np.int64))
    unknown_mask = ~known_mask
    unknown_count = int(np.sum(unknown_mask))
    if unknown_count == 0:
        return labels_zyx, 0, []

    unknown_values = sorted(int(v) for v in np.unique(labels_zyx[unknown_mask]).tolist())
    if policy == "error":
        raise UnknownMaterialIdError(
            f"Unknown material ids detected: values={unknown_values}, count={unknown_count}"
        )

    if unknown_to_void_id not in materials.ids:
        raise UnknownMaterialIdError(
            "unknown_to_void_id must exist in MaterialSpec.ids when map_to_void policy is used"
        )

    output = labels_zyx.copy()
    output[unknown_mask] = int(unknown_to_void_id)
    return output, unknown_count, unknown_values


def build_gridspec_meta(
    raw: RawVtiImage,
    material_id_zyx: np.ndarray,
    config: LabelNormalizeConfig,
    *,
    source_path: str,
    source_name: str,
    source_location: str,
    converted_from_point: bool,
    remap_applied: bool,
    unknown_policy: UnknownLabelPolicy,
    mask_conflict_count: int,
    input_hash: str,
    generator_version: str,
    git_commit: str,
    created_at: str | None,
    materials: MaterialSpec,
) -> tuple[GridSpec, Meta]:
    spacing_xyz = config.spacing_override_xyz or raw.spacing_xyz
    origin_xyz = config.origin_override_xyz or raw.origin_xyz

    for idx, value in enumerate(spacing_xyz):
        if not isfinite(value) or value <= 0.0:
            raise InvalidGridMetaError(f"spacing_xyz[{idx}] must be finite and >0, got {value}")
    for idx, value in enumerate(origin_xyz):
        if not isfinite(value):
            raise InvalidGridMetaError(f"origin_xyz[{idx}] must be finite, got {value}")

    sx, sy, sz = (float(v) for v in spacing_xyz)
    ox, oy, oz = (float(v) for v in origin_xyz)

    # Canonical output is cell-centered regardless of input location.
    origin_cell_xyz = (ox + 0.5 * sx, oy + 0.5 * sy, oz + 0.5 * sz)
    spacing_zyx = (sz, sy, sx)
    origin_zyx = (origin_cell_xyz[2], origin_cell_xyz[1], origin_cell_xyz[0])

    units = config.force_units or raw.vtk_meta.get("units", "nm")
    grid = GridSpec(
        dim=3,
        spacing=spacing_zyx,
        origin=origin_zyx,
        axis_order="ZYX",
        sample_location="cell_center",
        units=units,
    )

    config_hash = hash_config(
        {
            "normalize": asdict(config),
            "materials": {
                "ids": materials.ids,
                "void_id": materials.void_id,
            },
        }
    )

    meta = Meta(
        schema_version=config.schema_version,
        profile_id=config.profile_id,
        config_hash=config_hash,
        generator_version=generator_version,
        git_commit=git_commit,
        input_hash=input_hash,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        extra={
            "source_path": source_path,
            "label_array_name": source_name,
            "label_was_point_data": str(source_location == "point"),
            "converted_from_point": str(converted_from_point),
            "remap_applied": str(remap_applied),
            "unknown_policy": unknown_policy,
            "is_2d": str(material_id_zyx.shape[0] == 1),
            "mask_conflict_count": str(mask_conflict_count),
        },
    )
    return grid, meta


def normalize_raw_to_label(
    raw: RawVtiImage,
    materials: MaterialSpec,
    config: LabelNormalizeConfig,
    *,
    source_path: str,
    input_hash: str,
    generator_version: str,
    git_commit: str,
    created_at: str | None = None,
) -> tuple[LabelVolume, LabelQA]:
    source = select_label_source(raw, materials, config)
    notes: list[str] = []
    resolved_flat_layout: FlatArrayLayout = (
        infer_flat_array_layout(raw)
        if config.flat_array_layout == "auto"
        else config.flat_array_layout
    )
    notes.append(f"flat_layout={resolved_flat_layout}")

    converted_from_point = False
    mask_conflict_count = 0

    if source.kind == "scalar":
        assert source.name is not None
        location = raw.array_location[source.name]
        labels_zyx = to_zyx(
            raw.arrays[source.name],
            raw.dims_xyz,
            location,
            flat_layout=resolved_flat_layout,
        )
        labels_zyx = _coerce_scalar_labels(labels_zyx)
        if location == "point":
            converted_from_point = True
            labels_zyx = point_to_cell(
                labels_zyx,
                policy=config.point_to_cell_policy,
                materials=materials,
            )
    else:
        masks_by_id: dict[int, np.ndarray] = {}
        for material_id, mask_name in source.mask_names.items():
            location = raw.array_location[mask_name]
            mask_zyx = to_zyx(
                raw.arrays[mask_name],
                raw.dims_xyz,
                location,
                flat_layout=resolved_flat_layout,
            )
            mask_zyx = mask_zyx.astype(bool, copy=False)
            if location == "point":
                converted_from_point = True
                mask_zyx = point_mask_to_cell(mask_zyx, policy=config.point_to_cell_policy)
            masks_by_id[material_id] = mask_zyx

        labels_zyx, mask_conflict_count = merge_material_masks(
            masks_by_id,
            materials,
            policy=config.mask_merge_policy,
        )
        if mask_conflict_count > 0:
            notes.append(f"mask_conflict_count={mask_conflict_count}")

    remapped_zyx, remap_applied = apply_remap_ids(labels_zyx, config.remap_ids)
    if remap_applied:
        notes.append("remap_ids_applied")

    labels_known_zyx, unknown_count, unknown_values = apply_unknown_policy(
        remapped_zyx,
        materials,
        policy=config.unknown_label_policy,
        unknown_to_void_id=config.unknown_to_void_id,
    )
    if unknown_count > 0:
        notes.append(f"unknown_count={unknown_count}")

    target_dtype = np.uint8 if max(materials.ids) <= 255 else np.uint16
    material_id_zyx = labels_known_zyx.astype(target_dtype, copy=False)

    source_name = source.name or "mask_bundle"
    source_location: str
    if source.kind == "scalar" and source.name is not None:
        source_location = raw.array_location[source.name]
    else:
        source_location = "mixed"

    grid, meta = build_gridspec_meta(
        raw,
        material_id_zyx,
        config,
        source_path=source_path,
        source_name=source_name,
        source_location=source_location,
        converted_from_point=converted_from_point,
        remap_applied=remap_applied,
        unknown_policy=config.unknown_label_policy,
        mask_conflict_count=mask_conflict_count,
        input_hash=input_hash,
        generator_version=generator_version,
        git_commit=git_commit,
        created_at=created_at,
        materials=materials,
    )

    label = LabelVolume(
        grid=grid,
        material=materials,
        material_id=material_id_zyx,
        meta=meta,
    )

    qa = compute_label_qa(
        material_id_zyx,
        grid,
        materials,
        unknown_count=unknown_count,
        unknown_values=unknown_values,
        converted_from_point=converted_from_point,
        mask_conflict_count=mask_conflict_count,
        notes=notes,
    )
    return label, qa


def _coerce_to_zyx(
    array: np.ndarray,
    candidate_xyz_shapes: list[tuple[int, int, int]],
    *,
    flat_layout: FlatArrayLayout = "auto",
) -> np.ndarray:
    resolved_layout: FlatArrayLayout = (
        "legacy_xyz_transpose" if flat_layout == "auto" else flat_layout
    )
    if array.ndim == 1:
        for shape_xyz in candidate_xyz_shapes:
            if int(np.prod(shape_xyz)) == int(array.size):
                if resolved_layout == "vtk_x_fastest":
                    shape_zyx = (shape_xyz[2], shape_xyz[1], shape_xyz[0])
                    return np.reshape(array, shape_zyx, order="C")
                return np.reshape(array, shape_xyz, order="C").transpose(2, 1, 0)
        raise InvalidArrayShapeError(
            f"Flat array size={array.size} does not match expected shapes={candidate_xyz_shapes}"
        )

    if array.ndim == 3:
        for shape_xyz in candidate_xyz_shapes:
            shape_zyx = (shape_xyz[2], shape_xyz[1], shape_xyz[0])
            if tuple(array.shape) == shape_xyz:
                return array.transpose(2, 1, 0)
            if tuple(array.shape) == shape_zyx:
                return array
        for shape_xyz in candidate_xyz_shapes:
            if int(np.prod(shape_xyz)) == int(array.size):
                if resolved_layout == "vtk_x_fastest":
                    shape_zyx = (shape_xyz[2], shape_xyz[1], shape_xyz[0])
                    return np.reshape(array, shape_zyx, order="C")
                reshaped = np.reshape(array, shape_xyz, order="C")
                return reshaped.transpose(2, 1, 0)
        raise InvalidArrayShapeError(
            f"3D array shape={array.shape} does not match expected shapes={candidate_xyz_shapes}"
        )

    raise InvalidArrayShapeError(
        f"Array ndim must be 1 or 3, got ndim={array.ndim} shape={array.shape}"
    )


def _coerce_scalar_labels(labels: np.ndarray) -> np.ndarray:
    if np.issubdtype(labels.dtype, np.integer):
        return labels.astype(np.int64, copy=False)

    if np.issubdtype(labels.dtype, np.floating):
        rounded = np.rint(labels)
        if np.allclose(labels, rounded, atol=0.0):
            return rounded.astype(np.int64)

    raise InvalidArrayShapeError(
        f"Scalar label array must be integer-valued, got dtype={labels.dtype}"
    )


def point_mask_to_cell(point_mask_zyx: np.ndarray, *, policy: PointToCellPolicy) -> np.ndarray:
    if policy == "error":
        raise PointToCellConversionError("PointData input requires point_to_cell_policy != 'error'")

    point_mask = point_mask_zyx.astype(bool, copy=False)
    z_size, y_size, x_size = point_mask.shape
    cell_shape = (max(z_size - 1, 1), max(y_size - 1, 1), max(x_size - 1, 1))

    if policy == "nearest":
        return point_mask[: cell_shape[0], : cell_shape[1], : cell_shape[2]].copy()

    output = np.zeros(cell_shape, dtype=bool)
    for z in range(cell_shape[0]):
        z_index = [0] if z_size == 1 else [z, z + 1]
        for y in range(cell_shape[1]):
            y_index = [0] if y_size == 1 else [y, y + 1]
            for x in range(cell_shape[2]):
                x_index = [0] if x_size == 1 else [x, x + 1]
                true_count = 0
                total = 0
                for zz in z_index:
                    for yy in y_index:
                        for xx in x_index:
                            total += 1
                            if point_mask[zz, yy, xx]:
                                true_count += 1
                if policy == "majority_nearest_tie" and true_count * 2 == total:
                    output[z, y, x] = bool(point_mask[z_index[0], y_index[0], x_index[0]])
                else:
                    output[z, y, x] = true_count * 2 >= total
    return output
