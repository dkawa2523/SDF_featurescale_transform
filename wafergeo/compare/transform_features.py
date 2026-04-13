from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wafergeo.compare.sdf_helpers import tsdf_from_sdf_nm
from wafergeo.core.types import LabelVolume
from wafergeo.sdf.edt import signed_distance_from_mask

TSDF_VIEW_CLIP_NM = (10.0, 30.0, 100.0)
PROCESS_DELTA_NAMES = ("changed", "etched", "deposited", "material_changed")


def _array_stats(array: np.ndarray) -> dict[str, object]:
    values = np.asarray(array)
    finite = values[np.isfinite(values)]
    stats: dict[str, object] = {
        "shape": [int(v) for v in values.shape],
        "dtype": str(values.dtype),
        "nan_count": int(np.isnan(values).sum()) if np.issubdtype(values.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(values).sum()) if np.issubdtype(values.dtype, np.floating) else 0,
    }
    if finite.size:
        stats.update(
            {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean": float(finite.mean()),
                "std": float(finite.std()),
            }
        )
    return stats


def _file_size_mb(path: Path) -> float:
    return float(path.stat().st_size) / (1024.0 * 1024.0)


def _non_void_sdf_raw(label: LabelVolume) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(label.material_id)
    non_void = labels != int(label.material.void_id)
    spacing_zyx = (
        float(label.grid.spacing[0]),
        float(label.grid.spacing[1]),
        float(label.grid.spacing[2]),
    )
    if not np.any(non_void):
        sdf_nm = np.full(labels.shape, 1e6, dtype=np.float32)
    elif np.all(non_void):
        sdf_nm = np.full(labels.shape, -1e6, dtype=np.float32)
    else:
        sdf_nm = signed_distance_from_mask(non_void, spacing_zyx, backend="scipy")
    return sdf_nm.astype(np.float32, copy=False), non_void


def _material_ids_without_void(label: LabelVolume) -> list[int]:
    void_id = int(label.material.void_id)
    return [int(v) for v in label.material.ids if int(v) != void_id]


def _sdf_from_mask(mask: np.ndarray, spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    if not np.any(mask):
        return np.full(mask.shape, 1e6, dtype=np.float32)
    if np.all(mask):
        return np.full(mask.shape, -1e6, dtype=np.float32)
    return signed_distance_from_mask(mask, spacing_zyx, backend="scipy").astype(
        np.float32,
        copy=False,
    )


def _material_sdf_stack(
    label: LabelVolume,
    *,
    include_void: bool,
) -> tuple[np.ndarray, list[int], list[int]]:
    labels = np.asarray(label.material_id)
    spacing_zyx = (
        float(label.grid.spacing[0]),
        float(label.grid.spacing[1]),
        float(label.grid.spacing[2]),
    )
    material_ids = (
        [int(v) for v in label.material.ids]
        if include_void
        else _material_ids_without_void(label)
    )
    sdf_stack = np.empty((len(material_ids), *labels.shape), dtype=np.float32)
    voxel_counts: list[int] = []
    for index, material_id in enumerate(material_ids):
        mask = labels == material_id
        voxel_counts.append(int(np.count_nonzero(mask)))
        sdf_stack[index] = _sdf_from_mask(mask, spacing_zyx)
    return sdf_stack, material_ids, voxel_counts


def _save_spatial_metadata(
    *,
    label: LabelVolume,
    material_ids: list[int],
    voxel_counts: list[int] | None = None,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "material_ids": np.asarray(material_ids, dtype=np.int32),
        "spacing_zyx_nm": np.asarray(label.grid.spacing, dtype=np.float32),
        "origin_zyx_nm": np.asarray(label.grid.origin, dtype=np.float32),
        "void_id": np.asarray(int(label.material.void_id), dtype=np.int32),
    }
    if voxel_counts is not None:
        arrays["voxel_counts"] = np.asarray(voxel_counts, dtype=np.int64)
    return arrays


def _tsdf_view_arrays(sdf_nm: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "tsdf_10nm": tsdf_from_sdf_nm(sdf_nm, clip_nm=10.0),
        "tsdf_30nm": tsdf_from_sdf_nm(sdf_nm, clip_nm=30.0),
        "tsdf_100nm": tsdf_from_sdf_nm(sdf_nm, clip_nm=100.0),
        "log_abs_sdf": np.log1p(np.abs(sdf_nm)).astype(np.float32),
        "clip_nm": np.asarray(TSDF_VIEW_CLIP_NM, dtype=np.float32),
    }


def _top2_abs_indices(
    abs_stack: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nearest_distance = np.full(abs_stack.shape[1:], np.inf, dtype=np.float32)
    second_distance = np.full(abs_stack.shape[1:], np.inf, dtype=np.float32)
    nearest_index = np.full(abs_stack.shape[1:], -1, dtype=np.int16)
    second_index = np.full(abs_stack.shape[1:], -1, dtype=np.int16)
    for index in range(abs_stack.shape[0]):
        current = np.asarray(abs_stack[index], dtype=np.float32)
        better = current < nearest_distance
        second_distance[better] = nearest_distance[better]
        second_index[better] = nearest_index[better]
        nearest_distance[better] = current[better]
        nearest_index[better] = index

        candidate_second = (~better) & (current < second_distance)
        second_distance[candidate_second] = current[candidate_second]
        second_index[candidate_second] = index
    return nearest_distance, second_distance, nearest_index, second_index


def _pair_code_arrays(
    *,
    nearest_index: np.ndarray,
    second_index: np.ndarray,
    material_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    codebook_rows: list[tuple[int, int, int]] = []
    pair_code = np.full(nearest_index.shape, -1, dtype=np.int32)
    code = 0
    for left_index in range(len(material_ids)):
        for right_index in range(left_index + 1, len(material_ids)):
            left_id = int(material_ids[left_index])
            right_id = int(material_ids[right_index])
            codebook_rows.append((code, left_id, right_id))
            mask = (
                ((nearest_index == left_index) & (second_index == right_index))
                | ((nearest_index == right_index) & (second_index == left_index))
            )
            pair_code[mask] = code
            code += 1
    return pair_code, np.asarray(codebook_rows, dtype=np.int32)


def write_material_interface_relation_feature(label: LabelVolume, output_dir: Path) -> str:
    sdf_stack, material_ids, _ = _material_sdf_stack(label, include_void=True)
    abs_stack = np.abs(sdf_stack)
    nearest_distance, second_distance, nearest_index, second_index = _top2_abs_indices(abs_stack)
    material_id_array = np.asarray(material_ids, dtype=np.int32)
    nearest_material_id = material_id_array[np.maximum(nearest_index, 0)]
    second_material_id = material_id_array[np.maximum(second_index, 0)]
    pair_code, pair_codebook = _pair_code_arrays(
        nearest_index=nearest_index,
        second_index=second_index,
        material_ids=material_ids,
    )
    interface_distance_nm = nearest_distance.astype(np.float32, copy=False)
    distance_gap_nm = np.where(
        np.isfinite(second_distance),
        np.maximum(second_distance - nearest_distance, 0.0),
        0.0,
    ).astype(np.float32, copy=False)

    path = output_dir / "material_interface_relation.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "interface_distance_nm": interface_distance_nm,
        "nearest_material_id": nearest_material_id.astype(np.int32, copy=False),
        "second_material_id": second_material_id.astype(np.int32, copy=False),
        "pair_code": pair_code,
        "pair_codebook": pair_codebook,
        "distance_gap_nm": distance_gap_nm,
        "interface_band_10nm": (interface_distance_nm <= 10.0).astype(np.uint8),
        "interface_band_30nm": (interface_distance_nm <= 30.0).astype(np.uint8),
        "interface_band_100nm": (interface_distance_nm <= 100.0).astype(np.uint8),
        **_save_spatial_metadata(label=label, material_ids=material_ids),
    }
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return path.name


def write_label_sdf_raw_feature(label: LabelVolume, output_dir: Path) -> str:
    spacing_zyx = (
        float(label.grid.spacing[0]),
        float(label.grid.spacing[1]),
        float(label.grid.spacing[2]),
    )
    sdf_nm, non_void = _non_void_sdf_raw(label)
    path = output_dir / "sdf_raw.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(  # type: ignore[arg-type]
        path,
        sdf_nm=sdf_nm.astype(np.float32, copy=False),
        mask=non_void.astype(np.uint8, copy=False),
        spacing_zyx_nm=np.asarray(spacing_zyx, dtype=np.float32),
        origin_zyx_nm=np.asarray(label.grid.origin, dtype=np.float32),
        material_ids=np.asarray(label.material.ids, dtype=np.int32),
        void_id=np.asarray(int(label.material.void_id), dtype=np.int32),
    )
    return str(path.name)


def write_tsdf_views_feature(label: LabelVolume, output_dir: Path) -> str:
    sdf_nm, non_void = _non_void_sdf_raw(label)
    path = output_dir / "tsdf_views.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "sdf_nm": sdf_nm,
        "mask": non_void.astype(np.uint8, copy=False),
        **_tsdf_view_arrays(sdf_nm),
        "spacing_zyx_nm": np.asarray(label.grid.spacing, dtype=np.float32),
        "origin_zyx_nm": np.asarray(label.grid.origin, dtype=np.float32),
        "material_ids": np.asarray(label.material.ids, dtype=np.int32),
        "void_id": np.asarray(int(label.material.void_id), dtype=np.int32),
    }
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return str(path.name)


def write_label_udf_feature(label: LabelVolume, output_dir: Path) -> str:
    sdf_nm, non_void = _non_void_sdf_raw(label)
    udf_nm = np.abs(sdf_nm).astype(np.float32, copy=False)
    path = output_dir / "udf.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(  # type: ignore[arg-type]
        path,
        udf_nm=udf_nm,
        mask=non_void.astype(np.uint8, copy=False),
        spacing_zyx_nm=np.asarray(label.grid.spacing, dtype=np.float32),
        origin_zyx_nm=np.asarray(label.grid.origin, dtype=np.float32),
        material_ids=np.asarray(label.material.ids, dtype=np.int32),
        void_id=np.asarray(int(label.material.void_id), dtype=np.int32),
    )
    return str(path.name)


def write_material_sdf_feature(label: LabelVolume, output_dir: Path) -> str:
    sdf_stack, material_ids, voxel_counts = _material_sdf_stack(label, include_void=False)
    path = output_dir / "material_sdf.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "sdf_nm": sdf_stack,
        **_save_spatial_metadata(
            label=label,
            material_ids=material_ids,
            voxel_counts=voxel_counts,
        ),
    }
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return str(path.name)


def write_material_tsdf_views_feature(label: LabelVolume, output_dir: Path) -> str:
    sdf_stack, material_ids, voxel_counts = _material_sdf_stack(label, include_void=False)
    path = output_dir / "material_tsdf_views.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "sdf_nm": sdf_stack,
        **_tsdf_view_arrays(sdf_stack),
        **_save_spatial_metadata(
            label=label,
            material_ids=material_ids,
            voxel_counts=voxel_counts,
        ),
    }
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return path.name


def write_material_udf_feature(label: LabelVolume, output_dir: Path) -> str:
    sdf_stack, material_ids, voxel_counts = _material_sdf_stack(label, include_void=False)
    path = output_dir / "material_udf.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "udf_nm": np.abs(sdf_stack).astype(np.float32, copy=False),
        **_save_spatial_metadata(
            label=label,
            material_ids=material_ids,
            voxel_counts=voxel_counts,
        ),
    }
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return path.name


def _delta_sdf_from_mask(mask: np.ndarray, spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    return _sdf_from_mask(mask, spacing_zyx)


def _process_delta_masks(
    *,
    reference_label: LabelVolume,
    final_label: LabelVolume,
) -> dict[str, np.ndarray]:
    initial = np.asarray(reference_label.material_id)
    final = np.asarray(final_label.material_id)
    initial_void = initial == int(reference_label.material.void_id)
    final_void = final == int(final_label.material.void_id)
    changed = initial != final
    return {
        "changed": changed,
        "etched": np.logical_and(~initial_void, final_void),
        "deposited": np.logical_and(initial_void, ~final_void),
        "material_changed": np.logical_and.reduce((~initial_void, ~final_void, changed)),
    }


def _process_delta_sdf_stack(
    *,
    reference_label: LabelVolume,
    final_label: LabelVolume,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    masks = _process_delta_masks(reference_label=reference_label, final_label=final_label)
    spacing_zyx = (
        float(final_label.grid.spacing[0]),
        float(final_label.grid.spacing[1]),
        float(final_label.grid.spacing[2]),
    )
    sdf_stack = np.stack(
        [_delta_sdf_from_mask(masks[name], spacing_zyx) for name in PROCESS_DELTA_NAMES],
        axis=0,
    ).astype(np.float32, copy=False)
    return sdf_stack, masks


def _process_delta_metadata(
    *,
    reference_label: LabelVolume,
    final_label: LabelVolume,
) -> dict[str, np.ndarray]:
    return {
        "delta_names": np.asarray(PROCESS_DELTA_NAMES),
        "spacing_zyx_nm": np.asarray(final_label.grid.spacing, dtype=np.float32),
        "origin_zyx_nm": np.asarray(final_label.grid.origin, dtype=np.float32),
        "initial_void_id": np.asarray(int(reference_label.material.void_id), dtype=np.int32),
        "final_void_id": np.asarray(int(final_label.material.void_id), dtype=np.int32),
    }


def _transition_code_arrays(
    *,
    reference_label: LabelVolume,
    final_label: LabelVolume,
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.asarray(reference_label.material_id)
    final = np.asarray(final_label.material_id)
    changed = initial != final
    transition_code = np.full(initial.shape, -1, dtype=np.int32)
    codebook_rows: list[tuple[int, int, int]] = []
    for code, (initial_id, final_id) in enumerate(
        sorted({(int(a), int(b)) for a, b in zip(initial[changed], final[changed], strict=True)})
    ):
        transition_code[(initial == initial_id) & (final == final_id)] = code
        codebook_rows.append((code, initial_id, final_id))
    return transition_code, np.asarray(codebook_rows, dtype=np.int32)


def write_process_delta_sdf_feature(
    *,
    reference_label: LabelVolume,
    final_label: LabelVolume,
    output_dir: Path,
) -> dict[str, str]:
    sdf_stack, masks = _process_delta_sdf_stack(
        reference_label=reference_label,
        final_label=final_label,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "process_delta_sdf.npz"
    arrays = _process_delta_metadata(reference_label=reference_label, final_label=final_label)
    for idx, name in enumerate(PROCESS_DELTA_NAMES):
        mask = masks[name]
        arrays[f"{name}_mask"] = mask.astype(np.uint8, copy=False)
        arrays[f"{name}_sdf_nm"] = sdf_stack[idx]

    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return {"process_delta_sdf": path.name}


def write_process_delta_tsdf_views_feature(
    *,
    reference_label: LabelVolume,
    final_label: LabelVolume,
    output_dir: Path,
) -> str:
    sdf_stack, masks = _process_delta_sdf_stack(
        reference_label=reference_label,
        final_label=final_label,
    )
    path = output_dir / "process_delta_tsdf_views.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "sdf_nm": sdf_stack,
        "changed_mask": masks["changed"].astype(np.uint8, copy=False),
        **_tsdf_view_arrays(sdf_stack),
        **_process_delta_metadata(reference_label=reference_label, final_label=final_label),
    }
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return path.name


def write_process_delta_udf_feature(
    *,
    reference_label: LabelVolume,
    final_label: LabelVolume,
    output_dir: Path,
) -> str:
    sdf_stack, masks = _process_delta_sdf_stack(
        reference_label=reference_label,
        final_label=final_label,
    )
    path = output_dir / "process_delta_udf.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "udf_nm": np.abs(sdf_stack).astype(np.float32, copy=False),
        "changed_mask": masks["changed"].astype(np.uint8, copy=False),
        **_process_delta_metadata(reference_label=reference_label, final_label=final_label),
    }
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return path.name


def write_process_transition_relation_feature(
    *,
    reference_label: LabelVolume,
    final_label: LabelVolume,
    output_dir: Path,
) -> str:
    sdf_stack, masks = _process_delta_sdf_stack(
        reference_label=reference_label,
        final_label=final_label,
    )
    transition_code, transition_codebook = _transition_code_arrays(
        reference_label=reference_label,
        final_label=final_label,
    )
    transition_distance_nm = np.abs(sdf_stack[0]).astype(np.float32, copy=False)
    path = output_dir / "process_transition_relation.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "transition_code": transition_code,
        "transition_codebook": transition_codebook,
        "changed_mask": masks["changed"].astype(np.uint8, copy=False),
        "transition_distance_nm": transition_distance_nm,
        "transition_band_10nm": (transition_distance_nm <= 10.0).astype(np.uint8),
        "transition_band_30nm": (transition_distance_nm <= 30.0).astype(np.uint8),
        "transition_band_100nm": (transition_distance_nm <= 100.0).astype(np.uint8),
        **_process_delta_metadata(reference_label=reference_label, final_label=final_label),
    }
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return path.name


def write_transform_feature_summary(
    *,
    label: LabelVolume,
    output_dir: Path,
    written: dict[str, str],
) -> str:
    features: list[dict[str, object]] = []
    for name, filename in sorted(written.items()):
        if (
            name.endswith("_summary") or name.endswith("_legend") or name.endswith("_change_map")
        ):
            continue
        path = output_dir / filename
        row: dict[str, object] = {
            "name": name,
            "path": filename,
            "size_mb": _file_size_mb(path) if path.exists() else None,
        }
        if name == "sdf_raw" and path.exists():
            with np.load(path, allow_pickle=False) as data:
                sdf_nm = np.asarray(data["sdf_nm"], dtype=np.float32)
            row.update(
                {
                    "semantics": "signed_distance",
                    "units": label.grid.units,
                    "axis_order_internal": "ZYX",
                    "axis_order_user": ["x", "y", "z"],
                    "spacing_zyx_nm": [float(v) for v in label.grid.spacing],
                    "origin_zyx_nm": [float(v) for v in label.grid.origin],
                    "void_id": int(label.material.void_id),
                    "material_ids": [int(v) for v in label.material.ids],
                    "inside_sign": "negative",
                    "outside_sign": "positive",
                    "source_region": "non_void_union",
                    "array": _array_stats(sdf_nm),
                }
            )
        elif name == "tsdf_views" and path.exists():
            with np.load(path, allow_pickle=False) as data:
                sdf_nm = np.asarray(data["sdf_nm"], dtype=np.float32)
                clip_nm = [float(v) for v in np.asarray(data["clip_nm"]).tolist()]
            row.update(
                {
                    "semantics": "derived_tsdf_views",
                    "units": label.grid.units,
                    "axis_order_internal": "ZYX",
                    "axis_order_user": ["x", "y", "z"],
                    "spacing_zyx_nm": [float(v) for v in label.grid.spacing],
                    "origin_zyx_nm": [float(v) for v in label.grid.origin],
                    "void_id": int(label.material.void_id),
                    "material_ids": [int(v) for v in label.material.ids],
                    "source_feature": "sdf_raw",
                    "source_region": "non_void_union",
                    "clip_nm": clip_nm,
                    "array": _array_stats(sdf_nm),
                }
            )
        elif name == "udf" and path.exists():
            with np.load(path, allow_pickle=False) as data:
                udf_nm = np.asarray(data["udf_nm"], dtype=np.float32)
            row.update(
                {
                    "semantics": "unsigned_distance",
                    "units": label.grid.units,
                    "axis_order_internal": "ZYX",
                    "axis_order_user": ["x", "y", "z"],
                    "spacing_zyx_nm": [float(v) for v in label.grid.spacing],
                    "origin_zyx_nm": [float(v) for v in label.grid.origin],
                    "void_id": int(label.material.void_id),
                    "material_ids": [int(v) for v in label.material.ids],
                    "source_feature": "sdf_raw",
                    "source_region": "non_void_union_boundary",
                    "array": _array_stats(udf_nm),
                }
            )
        elif name in {"material_sdf", "material_tsdf_views"} and path.exists():
            with np.load(path, allow_pickle=False) as data:
                sdf_nm = np.asarray(data["sdf_nm"], dtype=np.float32)
                material_ids = [int(v) for v in np.asarray(data["material_ids"]).tolist()]
                voxel_counts = [int(v) for v in np.asarray(data["voxel_counts"]).tolist()]
                material_clip_nm: list[float] | None = (
                    [float(v) for v in np.asarray(data["clip_nm"]).tolist()]
                    if "clip_nm" in data.files
                    else None
                )
            row.update(
                {
                    "semantics": (
                        "per_material_multi_scale_tsdf"
                        if name == "material_tsdf_views"
                        else "per_material_signed_distance"
                    ),
                    "units": label.grid.units,
                    "axis_order_internal": "MZYX",
                    "axis_order_user": ["material", "x", "y", "z"],
                    "spacing_zyx_nm": [float(v) for v in label.grid.spacing],
                    "origin_zyx_nm": [float(v) for v in label.grid.origin],
                    "void_id": int(label.material.void_id),
                    "material_ids": material_ids,
                    "voxel_counts": voxel_counts,
                    "inside_sign": "negative",
                    "outside_sign": "positive",
                    "source_region": "per_material_mask",
                    "array": _array_stats(sdf_nm),
                }
            )
            if material_clip_nm is not None:
                row["clip_nm"] = material_clip_nm
        elif name == "material_udf" and path.exists():
            with np.load(path, allow_pickle=False) as data:
                udf_nm = np.asarray(data["udf_nm"], dtype=np.float32)
                material_ids = [int(v) for v in np.asarray(data["material_ids"]).tolist()]
                voxel_counts = [int(v) for v in np.asarray(data["voxel_counts"]).tolist()]
            row.update(
                {
                    "semantics": "per_material_unsigned_distance",
                    "units": label.grid.units,
                    "axis_order_internal": "MZYX",
                    "axis_order_user": ["material", "x", "y", "z"],
                    "spacing_zyx_nm": [float(v) for v in label.grid.spacing],
                    "origin_zyx_nm": [float(v) for v in label.grid.origin],
                    "void_id": int(label.material.void_id),
                    "material_ids": material_ids,
                    "voxel_counts": voxel_counts,
                    "source_region": "per_material_boundary",
                    "array": _array_stats(udf_nm),
                }
            )
        elif name == "material_interface_relation" and path.exists():
            with np.load(path, allow_pickle=False) as data:
                interface_distance_nm = np.asarray(
                    data["interface_distance_nm"],
                    dtype=np.float32,
                )
                material_ids = [int(v) for v in np.asarray(data["material_ids"]).tolist()]
            row.update(
                {
                    "semantics": "material_interface_relation",
                    "units": label.grid.units,
                    "axis_order_internal": "ZYX",
                    "axis_order_user": ["x", "y", "z"],
                    "spacing_zyx_nm": [float(v) for v in label.grid.spacing],
                    "origin_zyx_nm": [float(v) for v in label.grid.origin],
                    "void_id": int(label.material.void_id),
                    "material_ids": material_ids,
                    "source_feature": "material_sdf",
                    "source_region": "nearest_material_interface",
                    "array": _array_stats(interface_distance_nm),
                }
            )
        elif name == "process_delta_sdf" and path.exists():
            with np.load(path, allow_pickle=False) as data:
                changed_sdf_nm = np.asarray(data["changed_sdf_nm"], dtype=np.float32)
                changed_mask = np.asarray(data["changed_mask"], dtype=bool)
            row.update(
                {
                    "semantics": "process_delta_signed_distance",
                    "units": label.grid.units,
                    "axis_order_internal": "ZYX",
                    "axis_order_user": ["x", "y", "z"],
                    "spacing_zyx_nm": [float(v) for v in label.grid.spacing],
                    "origin_zyx_nm": [float(v) for v in label.grid.origin],
                    "void_id": int(label.material.void_id),
                    "source_region": "changed_material_regions",
                    "inside_sign": "negative",
                    "outside_sign": "positive",
                    "changed_voxels": int(np.count_nonzero(changed_mask)),
                    "changed_fraction": (
                        float(np.count_nonzero(changed_mask) / changed_mask.size)
                        if changed_mask.size
                        else 0.0
                    ),
                    "array": _array_stats(changed_sdf_nm),
                }
            )
        elif name == "process_delta_tsdf_views" and path.exists():
            with np.load(path, allow_pickle=False) as data:
                sdf_nm = np.asarray(data["sdf_nm"], dtype=np.float32)
                changed_mask = np.asarray(data["changed_mask"], dtype=bool)
                clip_nm = [float(v) for v in np.asarray(data["clip_nm"]).tolist()]
            row.update(
                {
                    "semantics": "process_delta_multi_scale_tsdf",
                    "units": label.grid.units,
                    "axis_order_internal": "DZYX",
                    "axis_order_user": ["delta", "x", "y", "z"],
                    "spacing_zyx_nm": [float(v) for v in label.grid.spacing],
                    "origin_zyx_nm": [float(v) for v in label.grid.origin],
                    "void_id": int(label.material.void_id),
                    "source_region": "changed_material_regions",
                    "clip_nm": clip_nm,
                    "changed_voxels": int(np.count_nonzero(changed_mask)),
                    "changed_fraction": (
                        float(np.count_nonzero(changed_mask) / changed_mask.size)
                        if changed_mask.size
                        else 0.0
                    ),
                    "array": _array_stats(sdf_nm),
                }
            )
        elif name == "process_delta_udf" and path.exists():
            with np.load(path, allow_pickle=False) as data:
                udf_nm = np.asarray(data["udf_nm"], dtype=np.float32)
                changed_mask = np.asarray(data["changed_mask"], dtype=bool)
            row.update(
                {
                    "semantics": "process_delta_unsigned_distance",
                    "units": label.grid.units,
                    "axis_order_internal": "DZYX",
                    "axis_order_user": ["delta", "x", "y", "z"],
                    "spacing_zyx_nm": [float(v) for v in label.grid.spacing],
                    "origin_zyx_nm": [float(v) for v in label.grid.origin],
                    "void_id": int(label.material.void_id),
                    "source_region": "changed_material_boundary",
                    "changed_voxels": int(np.count_nonzero(changed_mask)),
                    "changed_fraction": (
                        float(np.count_nonzero(changed_mask) / changed_mask.size)
                        if changed_mask.size
                        else 0.0
                    ),
                    "array": _array_stats(udf_nm),
                }
            )
        elif name == "process_transition_relation" and path.exists():
            with np.load(path, allow_pickle=False) as data:
                transition_distance_nm = np.asarray(
                    data["transition_distance_nm"],
                    dtype=np.float32,
                )
                changed_mask = np.asarray(data["changed_mask"], dtype=bool)
            row.update(
                {
                    "semantics": "process_transition_relation",
                    "units": label.grid.units,
                    "axis_order_internal": "ZYX",
                    "axis_order_user": ["x", "y", "z"],
                    "spacing_zyx_nm": [float(v) for v in label.grid.spacing],
                    "origin_zyx_nm": [float(v) for v in label.grid.origin],
                    "void_id": int(label.material.void_id),
                    "source_feature": "process_delta_sdf",
                    "source_region": "reference_to_final_material_transition",
                    "changed_voxels": int(np.count_nonzero(changed_mask)),
                    "changed_fraction": (
                        float(np.count_nonzero(changed_mask) / changed_mask.size)
                        if changed_mask.size
                        else 0.0
                    ),
                    "array": _array_stats(transition_distance_nm),
                }
            )
        features.append(row)

    summary_path = output_dir.parent / "feature_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "feature_summary/v1",
                "features": features,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return summary_path.name
