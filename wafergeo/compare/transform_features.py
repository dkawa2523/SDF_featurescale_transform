from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from wafergeo.core.types import LabelVolume, TSDFVolume
from wafergeo.mesh.build import build_mesh_from_tsdf
from wafergeo.mesh.config import MeshBuildConfig
from wafergeo.sdf.full_material import build_full_material_sdf


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
