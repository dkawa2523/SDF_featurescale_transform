from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from wafergeo.compare.features import ViewFeature
from wafergeo.compare.sdf_helpers import tsdf_from_sdf_nm
from wafergeo.core.types import LabelVolume, TSDFVolume
from wafergeo.mesh.build import build_mesh_from_tsdf
from wafergeo.mesh.config import MeshBuildConfig
from wafergeo.sdf.edt import signed_distance_from_mask
from wafergeo.sdf.full_material import build_full_material_sdf


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


def write_sdf_views_feature(view_feature: ViewFeature, output_dir: Path) -> str:
    sdf_nm = np.asarray(view_feature.sdf_nm, dtype=np.float32)
    path = output_dir / "sdf_views.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        sdf_nm=sdf_nm,
        tsdf_10nm=tsdf_from_sdf_nm(sdf_nm, clip_nm=10.0),
        tsdf_50nm=tsdf_from_sdf_nm(sdf_nm, clip_nm=50.0),
        log_abs_sdf=np.log1p(np.abs(sdf_nm)).astype(np.float32),
        mask=view_feature.mask.astype(np.uint8),
        spacing=np.asarray(view_feature.grid2d.spacing, dtype=np.float32),
        origin=np.asarray(view_feature.grid2d.origin, dtype=np.float32),
    )
    return str(path.name)


def write_transform_feature_summary(
    *,
    label: LabelVolume,
    output_dir: Path,
    written: dict[str, str],
) -> str:
    features: list[dict[str, object]] = []
    for name, filename in sorted(written.items()):
        if name.endswith("_summary"):
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
