from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from wafergeo.compare.render import write_rgb_png
from wafergeo.compare.sdf_helpers import tsdf_from_sdf_nm
from wafergeo.core.types import LabelVolume, TSDFVolume
from wafergeo.mesh.build import build_mesh_from_tsdf
from wafergeo.mesh.config import MeshBuildConfig
from wafergeo.sdf.edt import signed_distance_from_mask
from wafergeo.sdf.full_material import build_full_material_sdf

TSDF_VIEW_CLIP_NM = (10.0, 30.0, 100.0)
PROCESS_DELTA_PREVIEW_COLORS: dict[str, tuple[int, int, int]] = {
    "unchanged": (245, 245, 245),
    "etched": (220, 70, 70),
    "deposited": (70, 160, 80),
    "material_changed": (70, 110, 220),
}


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


def _label_to_tsdf_volume(label: LabelVolume, *, mu_nm: float) -> TSDFVolume:
    material_ids = [int(v) for v in label.material.ids]
    spacing_zyx = (
        float(label.grid.spacing[0]),
        float(label.grid.spacing[1]),
        float(label.grid.spacing[2]),
    )
    tsdf = build_full_material_sdf(
        label_zyx=label.material_id,
        material_ids=material_ids,
        spacing_zyx=spacing_zyx,
        mu_nm=float(mu_nm),
    )
    return TSDFVolume(
        grid=label.grid,
        material=label.material,
        mu_nm=float(mu_nm),
        tsdf=tsdf,
        meta=label.meta,
    )


def write_label_sdf_feature(label: LabelVolume, output_dir: Path, *, mu_nm: float = 20.0) -> str:
    tsdf_volume = _label_to_tsdf_volume(label, mu_nm=mu_nm)
    material_ids = [int(v) for v in label.material.ids]
    path = output_dir / "sdf.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        tsdf=tsdf_volume.tsdf,
        material_ids=np.asarray(material_ids, dtype=np.int32),
        mu_nm=mu_nm,
    )
    return str(path.name)


def write_label_sdf_raw_feature(label: LabelVolume, output_dir: Path) -> str:
    spacing_zyx = (
        float(label.grid.spacing[0]),
        float(label.grid.spacing[1]),
        float(label.grid.spacing[2]),
    )
    sdf_nm, non_void = _non_void_sdf_raw(label)
    path = output_dir / "sdf_raw.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
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
    np.savez_compressed(
        path,
        sdf_nm=sdf_nm,
        tsdf_10nm=tsdf_from_sdf_nm(sdf_nm, clip_nm=10.0),
        tsdf_30nm=tsdf_from_sdf_nm(sdf_nm, clip_nm=30.0),
        tsdf_100nm=tsdf_from_sdf_nm(sdf_nm, clip_nm=100.0),
        log_abs_sdf=np.log1p(np.abs(sdf_nm)).astype(np.float32),
        mask=non_void.astype(np.uint8, copy=False),
        clip_nm=np.asarray(TSDF_VIEW_CLIP_NM, dtype=np.float32),
        spacing_zyx_nm=np.asarray(label.grid.spacing, dtype=np.float32),
        origin_zyx_nm=np.asarray(label.grid.origin, dtype=np.float32),
        material_ids=np.asarray(label.material.ids, dtype=np.int32),
        void_id=np.asarray(int(label.material.void_id), dtype=np.int32),
    )
    return str(path.name)


def write_label_udf_feature(label: LabelVolume, output_dir: Path) -> str:
    sdf_nm, non_void = _non_void_sdf_raw(label)
    udf_nm = np.abs(sdf_nm).astype(np.float32, copy=False)
    path = output_dir / "udf.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
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
    labels = np.asarray(label.material_id)
    spacing_zyx = tuple(float(v) for v in label.grid.spacing)
    material_ids = _material_ids_without_void(label)
    sdf_stack = np.empty((len(material_ids), *labels.shape), dtype=np.float32)
    voxel_counts: list[int] = []
    for index, material_id in enumerate(material_ids):
        mask = labels == material_id
        voxel_counts.append(int(np.count_nonzero(mask)))
        if not np.any(mask):
            sdf_stack[index].fill(1e6)
        elif np.all(mask):
            sdf_stack[index].fill(-1e6)
        else:
            sdf_stack[index] = signed_distance_from_mask(
                mask,
                spacing_zyx,
                backend="scipy",
            ).astype(np.float32, copy=False)

    path = output_dir / "material_sdf.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sdf_nm=sdf_stack,
        material_ids=np.asarray(material_ids, dtype=np.int32),
        voxel_counts=np.asarray(voxel_counts, dtype=np.int64),
        spacing_zyx_nm=np.asarray(label.grid.spacing, dtype=np.float32),
        origin_zyx_nm=np.asarray(label.grid.origin, dtype=np.float32),
        void_id=np.asarray(int(label.material.void_id), dtype=np.int32),
    )
    return str(path.name)


def _voxel_bbox_nm(mask: np.ndarray, label: LabelVolume) -> dict[str, float | None]:
    if not np.any(mask):
        return {
            "bbox_min_z_nm": None,
            "bbox_min_y_nm": None,
            "bbox_min_x_nm": None,
            "bbox_max_z_nm": None,
            "bbox_max_y_nm": None,
            "bbox_max_x_nm": None,
        }
    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    origin = tuple(float(v) for v in label.grid.origin)
    spacing = tuple(float(v) for v in label.grid.spacing)
    return {
        "bbox_min_z_nm": origin[0] + float(mins[0]) * spacing[0],
        "bbox_min_y_nm": origin[1] + float(mins[1]) * spacing[1],
        "bbox_min_x_nm": origin[2] + float(mins[2]) * spacing[2],
        "bbox_max_z_nm": origin[0] + float(maxs[0]) * spacing[0],
        "bbox_max_y_nm": origin[1] + float(maxs[1]) * spacing[1],
        "bbox_max_x_nm": origin[2] + float(maxs[2]) * spacing[2],
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_material_profile_feature(label: LabelVolume, output_dir: Path) -> dict[str, str]:
    labels = np.asarray(label.material_id)
    total_voxels = int(labels.size)
    void_id = int(label.material.void_id)
    material_names = {
        int(material_id): str(name)
        for material_id, name in zip(label.material.ids, label.material.names, strict=True)
    }
    present_ids = sorted(int(v) for v in np.unique(labels).tolist())
    spacing_zyx = tuple(float(v) for v in label.grid.spacing)
    origin_zyx = tuple(float(v) for v in label.grid.origin)

    material_rows: list[dict[str, object]] = []
    z_rows: list[dict[str, object]] = []
    material_summaries: list[dict[str, object]] = []

    for material_id in present_ids:
        mask = labels == material_id
        voxel_count = int(np.count_nonzero(mask))
        voxel_fraction = float(voxel_count / total_voxels) if total_voxels else 0.0
        bbox = _voxel_bbox_nm(mask, label)
        material_name = material_names.get(material_id, f"material_{material_id}")
        row: dict[str, object] = {
            "material_id": material_id,
            "material_name": material_name,
            "is_void": str(material_id == void_id).lower(),
            "voxel_count": voxel_count,
            "voxel_fraction": voxel_fraction,
            **bbox,
        }
        material_rows.append(row)
        material_summaries.append(
            {
                **row,
                "is_void": material_id == void_id,
            }
        )

        z_counts = np.count_nonzero(mask, axis=(1, 2))
        slice_voxels = int(labels.shape[1] * labels.shape[2])
        for z_index, z_count in enumerate(z_counts.tolist()):
            z_rows.append(
                {
                    "z_index": int(z_index),
                    "z_nm": origin_zyx[0] + float(z_index) * spacing_zyx[0],
                    "material_id": material_id,
                    "material_name": material_name,
                    "is_void": str(material_id == void_id).lower(),
                    "voxel_count": int(z_count),
                    "slice_fraction": float(z_count / slice_voxels) if slice_voxels else 0.0,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "material_profile.csv"
    z_profile_path = output_dir / "material_profile_z_profile.csv"
    summary_path = output_dir / "material_profile_summary.json"
    _write_csv(
        profile_path,
        material_rows,
        [
            "material_id",
            "material_name",
            "is_void",
            "voxel_count",
            "voxel_fraction",
            "bbox_min_z_nm",
            "bbox_min_y_nm",
            "bbox_min_x_nm",
            "bbox_max_z_nm",
            "bbox_max_y_nm",
            "bbox_max_x_nm",
        ],
    )
    _write_csv(
        z_profile_path,
        z_rows,
        [
            "z_index",
            "z_nm",
            "material_id",
            "material_name",
            "is_void",
            "voxel_count",
            "slice_fraction",
        ],
    )
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "material_profile/v1",
                "feature": "material_profile",
                "axis_order_internal": "ZYX",
                "units": label.grid.units,
                "shape_zyx": [int(v) for v in labels.shape],
                "spacing_zyx_nm": [float(v) for v in spacing_zyx],
                "origin_zyx_nm": [float(v) for v in origin_zyx],
                "void_id": void_id,
                "total_voxels": total_voxels,
                "material_count": len(present_ids),
                "material_ids": present_ids,
                "materials": material_summaries,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "material_profile": profile_path.name,
        "material_profile_z_profile": z_profile_path.name,
        "material_profile_summary": summary_path.name,
    }


def _material_name_map(label: LabelVolume) -> dict[int, str]:
    return {
        int(material_id): str(name)
        for material_id, name in zip(label.material.ids, label.material.names, strict=True)
    }


def _change_type(
    *,
    initial_material_id: int,
    final_material_id: int,
    initial_void_id: int,
    final_void_id: int,
) -> str:
    initial_is_void = initial_material_id == initial_void_id
    final_is_void = final_material_id == final_void_id
    if not initial_is_void and final_is_void:
        return "etched"
    if initial_is_void and not final_is_void:
        return "deposited"
    return "material_changed"


def write_process_delta_profile_feature(
    *,
    reference_label: LabelVolume,
    final_label: LabelVolume,
    output_dir: Path,
) -> dict[str, str]:
    initial = np.asarray(reference_label.material_id)
    final = np.asarray(final_label.material_id)
    total_voxels = int(final.size)
    initial_void_id = int(reference_label.material.void_id)
    final_void_id = int(final_label.material.void_id)
    initial_names = _material_name_map(reference_label)
    final_names = _material_name_map(final_label)
    changed = initial != final
    transition_rows: list[dict[str, object]] = []
    z_rows: list[dict[str, object]] = []
    transition_summaries: list[dict[str, object]] = []

    changed_pairs = sorted(
        {
            (int(initial_id), int(final_id))
            for initial_id, final_id in zip(initial[changed], final[changed], strict=True)
        }
    )
    for initial_id, final_id in changed_pairs:
        mask = np.logical_and(initial == initial_id, final == final_id)
        voxel_count = int(np.count_nonzero(mask))
        voxel_fraction = float(voxel_count / total_voxels) if total_voxels else 0.0
        transition_key = f"{initial_id}_to_{final_id}"
        change_type = _change_type(
            initial_material_id=initial_id,
            final_material_id=final_id,
            initial_void_id=initial_void_id,
            final_void_id=final_void_id,
        )
        bbox = _voxel_bbox_nm(mask, final_label)
        row: dict[str, object] = {
            "transition_key": transition_key,
            "change_type": change_type,
            "initial_material_id": initial_id,
            "initial_material_name": initial_names.get(initial_id, f"material_{initial_id}"),
            "final_material_id": final_id,
            "final_material_name": final_names.get(final_id, f"material_{final_id}"),
            "voxel_count": voxel_count,
            "voxel_fraction": voxel_fraction,
            **bbox,
        }
        transition_rows.append(row)
        transition_summaries.append(row)

        z_counts = np.count_nonzero(mask, axis=(1, 2))
        slice_voxels = int(final.shape[1] * final.shape[2])
        for z_index, z_count in enumerate(z_counts.tolist()):
            z_rows.append(
                {
                    "z_index": int(z_index),
                    "z_nm": float(final_label.grid.origin[0])
                    + float(z_index) * float(final_label.grid.spacing[0]),
                    "transition_key": transition_key,
                    "change_type": change_type,
                    "initial_material_id": initial_id,
                    "final_material_id": final_id,
                    "voxel_count": int(z_count),
                    "slice_fraction": float(z_count / slice_voxels) if slice_voxels else 0.0,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "process_delta_profile.csv"
    z_profile_path = output_dir / "process_delta_z_profile.csv"
    summary_path = output_dir / "process_delta_summary.json"
    _write_csv(
        profile_path,
        transition_rows,
        [
            "transition_key",
            "change_type",
            "initial_material_id",
            "initial_material_name",
            "final_material_id",
            "final_material_name",
            "voxel_count",
            "voxel_fraction",
            "bbox_min_z_nm",
            "bbox_min_y_nm",
            "bbox_min_x_nm",
            "bbox_max_z_nm",
            "bbox_max_y_nm",
            "bbox_max_x_nm",
        ],
    )
    _write_csv(
        z_profile_path,
        z_rows,
        [
            "z_index",
            "z_nm",
            "transition_key",
            "change_type",
            "initial_material_id",
            "final_material_id",
            "voxel_count",
            "slice_fraction",
        ],
    )
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "process_delta_profile/v1",
                "feature": "process_delta_profile",
                "axis_order_internal": "ZYX",
                "units": final_label.grid.units,
                "shape_zyx": [int(v) for v in final.shape],
                "spacing_zyx_nm": [float(v) for v in final_label.grid.spacing],
                "origin_zyx_nm": [float(v) for v in final_label.grid.origin],
                "initial_void_id": initial_void_id,
                "final_void_id": final_void_id,
                "total_voxels": total_voxels,
                "changed_voxels": int(np.count_nonzero(changed)),
                "changed_fraction": (
                    float(np.count_nonzero(changed) / total_voxels) if total_voxels else 0.0
                ),
                "transition_count": len(transition_rows),
                "transitions": transition_summaries,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "process_delta_profile": profile_path.name,
        "process_delta_z_profile": z_profile_path.name,
        "process_delta_summary": summary_path.name,
    }


def _delta_sdf_from_mask(mask: np.ndarray, spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    if not np.any(mask):
        return np.full(mask.shape, 1e6, dtype=np.float32)
    if np.all(mask):
        return np.full(mask.shape, -1e6, dtype=np.float32)
    return signed_distance_from_mask(mask, spacing_zyx, backend="scipy").astype(
        np.float32,
        copy=False,
    )


def _preview_scale(shape_yx: tuple[int, int], *, min_size: int = 384, max_scale: int = 8) -> int:
    smallest = max(min(shape_yx), 1)
    if smallest >= min_size:
        return 1
    return max(1, min(max_scale, int(np.ceil(min_size / smallest))))


def _process_delta_preview_y_index(changed_mask: np.ndarray) -> int:
    if changed_mask.ndim != 3:
        raise ValueError("process delta masks must be 3D ZYX arrays")
    changed_per_y = np.count_nonzero(changed_mask, axis=(0, 2))
    if np.any(changed_per_y):
        return int(np.argmax(changed_per_y))
    return int(changed_mask.shape[1] // 2)


def _write_process_delta_preview(
    *,
    output_dir: Path,
    masks: dict[str, np.ndarray],
    final_label: LabelVolume,
) -> dict[str, object]:
    y_index = _process_delta_preview_y_index(masks["changed"])
    z_x_shape = masks["changed"][:, y_index, :].shape
    rgb = np.full(
        z_x_shape + (3,),
        PROCESS_DELTA_PREVIEW_COLORS["unchanged"],
        dtype=np.uint8,
    )
    for name in ("etched", "deposited", "material_changed"):
        rgb[masks[name][:, y_index, :]] = np.asarray(
            PROCESS_DELTA_PREVIEW_COLORS[name],
            dtype=np.uint8,
        )
    rgb = np.flipud(rgb)
    display_scale = _preview_scale((int(z_x_shape[0]), int(z_x_shape[1])))
    if display_scale > 1:
        rgb = np.repeat(np.repeat(rgb, display_scale, axis=0), display_scale, axis=1)

    preview_path = output_dir / "process_delta_sdf_preview.png"
    write_rgb_png(preview_path, rgb)
    y_nm = float(final_label.grid.origin[1]) + float(y_index) * float(final_label.grid.spacing[1])
    return {
        "preview": preview_path.name,
        "view": "xz",
        "y_index": y_index,
        "y_nm": y_nm,
        "display_scale": display_scale,
        "colors": {
            name: {"rgb": list(rgb_value)}
            for name, rgb_value in PROCESS_DELTA_PREVIEW_COLORS.items()
        },
    }


def write_process_delta_sdf_feature(
    *,
    reference_label: LabelVolume,
    final_label: LabelVolume,
    output_dir: Path,
) -> dict[str, str]:
    initial = np.asarray(reference_label.material_id)
    final = np.asarray(final_label.material_id)
    initial_void_id = int(reference_label.material.void_id)
    final_void_id = int(final_label.material.void_id)
    spacing_zyx = (
        float(final_label.grid.spacing[0]),
        float(final_label.grid.spacing[1]),
        float(final_label.grid.spacing[2]),
    )

    changed = initial != final
    initial_void = initial == initial_void_id
    final_void = final == final_void_id
    etched = np.logical_and(~initial_void, final_void)
    deposited = np.logical_and(initial_void, ~final_void)
    material_changed = np.logical_and.reduce((~initial_void, ~final_void, changed))
    masks = {
        "changed": changed,
        "etched": etched,
        "deposited": deposited,
        "material_changed": material_changed,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "process_delta_sdf.npz"
    arrays: dict[str, np.ndarray] = {
        "spacing_zyx_nm": np.asarray(final_label.grid.spacing, dtype=np.float32),
        "origin_zyx_nm": np.asarray(final_label.grid.origin, dtype=np.float32),
        "initial_void_id": np.asarray(initial_void_id, dtype=np.int32),
        "final_void_id": np.asarray(final_void_id, dtype=np.int32),
    }
    summary_masks: dict[str, dict[str, object]] = {}
    for name, mask in masks.items():
        arrays[f"{name}_mask"] = mask.astype(np.uint8, copy=False)
        arrays[f"{name}_sdf_nm"] = _delta_sdf_from_mask(mask, spacing_zyx)
        voxel_count = int(np.count_nonzero(mask))
        summary_masks[name] = {
            "voxel_count": voxel_count,
            "voxel_fraction": float(voxel_count / final.size) if final.size else 0.0,
            **_voxel_bbox_nm(mask, final_label),
        }

    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]

    preview_info = _write_process_delta_preview(
        output_dir=output_dir,
        masks=masks,
        final_label=final_label,
    )
    legend_path = output_dir / "process_delta_sdf_legend.json"
    legend_path.write_text(
        json.dumps(
            {
                "schema_version": "process_delta_sdf_legend/v1",
                "feature": "process_delta_sdf",
                "preview": preview_info,
                "category_meaning": {
                    "unchanged": "same material id before and after process",
                    "etched": "initial non-void material became final void",
                    "deposited": "initial void became final non-void material",
                    "material_changed": (
                        "initial and final are both non-void, but material id changed"
                    ),
                },
                "category_priority": ["etched", "deposited", "material_changed"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    summary_path = output_dir / "process_delta_sdf_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "process_delta_sdf/v1",
                "feature": "process_delta_sdf",
                "axis_order_internal": "ZYX",
                "units": final_label.grid.units,
                "shape_zyx": [int(v) for v in final.shape],
                "spacing_zyx_nm": [float(v) for v in final_label.grid.spacing],
                "origin_zyx_nm": [float(v) for v in final_label.grid.origin],
                "initial_void_id": initial_void_id,
                "final_void_id": final_void_id,
                "total_voxels": int(final.size),
                "changed_voxels": int(np.count_nonzero(changed)),
                "changed_fraction": (
                    float(np.count_nonzero(changed) / final.size) if final.size else 0.0
                ),
                "masks": summary_masks,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "process_delta_sdf": path.name,
        "process_delta_sdf_legend": legend_path.name,
        "process_delta_sdf_preview": str(preview_info["preview"]),
        "process_delta_sdf_summary": summary_path.name,
    }


def write_transform_feature_summary(
    *,
    label: LabelVolume,
    output_dir: Path,
    written: dict[str, str],
) -> str:
    features: list[dict[str, object]] = []
    for name, filename in sorted(written.items()):
        if (
            name.endswith("_summary")
            or name.endswith("_z_profile")
            or name.endswith("_legend")
            or name.endswith("_preview")
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
        elif name == "material_sdf" and path.exists():
            with np.load(path, allow_pickle=False) as data:
                sdf_nm = np.asarray(data["sdf_nm"], dtype=np.float32)
                material_ids = [int(v) for v in np.asarray(data["material_ids"]).tolist()]
                voxel_counts = [int(v) for v in np.asarray(data["voxel_counts"]).tolist()]
            row.update(
                {
                    "semantics": "per_material_signed_distance",
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
        elif name == "material_profile" and path.exists():
            summary_path = output_dir / "material_profile_summary.json"
            material_summary: dict[str, object] = {}
            if summary_path.exists():
                material_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row.update(
                {
                    "semantics": "per_material_profile",
                    "units": label.grid.units,
                    "axis_order_internal": "ZYX",
                    "axis_order_user": ["x", "y", "z"],
                    "spacing_zyx_nm": [float(v) for v in label.grid.spacing],
                    "origin_zyx_nm": [float(v) for v in label.grid.origin],
                    "void_id": int(label.material.void_id),
                    "material_ids": material_summary.get(
                        "material_ids",
                        [int(v) for v in label.material.ids],
                    ),
                    "material_count": material_summary.get("material_count"),
                    "source_region": "label_material_ids",
                    "outputs": {
                        "profile": "material_profile.csv",
                        "z_profile": "material_profile_z_profile.csv",
                        "summary": "material_profile_summary.json",
                    },
                }
            )
        elif name == "process_delta_profile" and path.exists():
            summary_path = output_dir / "process_delta_summary.json"
            delta_summary: dict[str, object] = {}
            if summary_path.exists():
                delta_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row.update(
                {
                    "semantics": "process_delta_profile",
                    "units": label.grid.units,
                    "axis_order_internal": "ZYX",
                    "axis_order_user": ["x", "y", "z"],
                    "spacing_zyx_nm": [float(v) for v in label.grid.spacing],
                    "origin_zyx_nm": [float(v) for v in label.grid.origin],
                    "void_id": int(label.material.void_id),
                    "source_region": "changed_material_transitions",
                    "changed_voxels": delta_summary.get("changed_voxels"),
                    "changed_fraction": delta_summary.get("changed_fraction"),
                    "transition_count": delta_summary.get("transition_count"),
                    "outputs": {
                        "profile": "process_delta_profile.csv",
                        "z_profile": "process_delta_z_profile.csv",
                        "summary": "process_delta_summary.json",
                    },
                }
            )
        elif name == "process_delta_sdf" and path.exists():
            with np.load(path, allow_pickle=False) as data:
                changed_sdf_nm = np.asarray(data["changed_sdf_nm"], dtype=np.float32)
            summary_path = output_dir / "process_delta_sdf_summary.json"
            process_summary: dict[str, object] = {}
            if summary_path.exists():
                process_summary = json.loads(summary_path.read_text(encoding="utf-8"))
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
                    "changed_voxels": process_summary.get("changed_voxels"),
                    "changed_fraction": process_summary.get("changed_fraction"),
                    "outputs": {
                        "legend": "process_delta_sdf_legend.json",
                        "preview": "process_delta_sdf_preview.png",
                        "summary": "process_delta_sdf_summary.json",
                    },
                    "array": _array_stats(changed_sdf_nm),
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


def write_mesh_feature(
    label: LabelVolume,
    output_dir: Path,
    *,
    mu_nm: float = 20.0,
) -> dict[str, str]:
    tsdf_volume = _label_to_tsdf_volume(label, mu_nm=mu_nm)
    material_ids = [int(v) for v in label.material.ids]
    cfg = MeshBuildConfig(
        backend="naive_interface",
        mode="interface_mesh",
        channel_material_ids=material_ids,
        sample_points_n=128,
        sample_seed=0,
    )
    mesh, point_cloud, qa = build_mesh_from_tsdf(tsdf_volume, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = output_dir / "mesh.npz"
    np.savez(
        mesh_path,
        vertices=mesh.vertices,
        faces=mesh.faces,
        face_mat_in=mesh.face_mat_in,
        face_mat_out=mesh.face_mat_out,
        face_is_exposed=mesh.face_is_exposed,
        sample_points=point_cloud.points,
        sample_normals=point_cloud.normals,
        sample_pair_code=point_cloud.pair_code,
        sample_is_exposed=point_cloud.point_is_exposed,
    )

    summary_path = output_dir / "mesh_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "OK",
                "backend": cfg.backend,
                "mode": cfg.mode,
                "mesh_file": mesh_path.name,
                "qa": asdict(qa),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"mesh": mesh_path.name, "mesh_summary": summary_path.name}
