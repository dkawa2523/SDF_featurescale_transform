from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import numpy as np

from wafergeo.core.hashing import hash_config
from wafergeo.core.meta import Meta
from wafergeo.core.types import MeshGeom, PointCloud, TSDFVolume
from wafergeo.mesh.attrib import annotate_faces_from_tsdf, apply_mesh_mode
from wafergeo.mesh.config import MeshBuildConfig
from wafergeo.mesh.errors import ChannelMaterialMappingError, MeshBackendUnavailableError
from wafergeo.mesh.extractors.registry import get_mesh_extractor
from wafergeo.mesh.extractors.vtk_interface import apply_vtk_visual_postprocess
from wafergeo.mesh.qa import MeshQA, compute_mesh_qa
from wafergeo.mesh.sampling import sample_pointcloud


def _parse_selected_material_ids(raw: str) -> list[int]:
    items = [v.strip() for v in raw.split(",") if v.strip()]
    if not items:
        raise ValueError("selected_material_ids is empty")
    return [int(v) for v in items]


def resolve_channel_material_ids(tsdf: TSDFVolume, cfg: MeshBuildConfig) -> list[int]:
    channels = tsdf.tsdf.shape[0]

    if tsdf.meta is not None:
        selected_raw = tsdf.meta.extra.get("selected_material_ids")
        if selected_raw:
            parsed = _parse_selected_material_ids(selected_raw)
            if len(parsed) == channels:
                return parsed

    if cfg.channel_material_ids is not None:
        if len(cfg.channel_material_ids) != channels:
            raise ChannelMaterialMappingError(
                "MeshBuildConfig.channel_material_ids length must match TSDF channel count"
            )
        return list(cfg.channel_material_ids)

    raise ChannelMaterialMappingError(
        "channel->material mapping is ambiguous. Provide meta.extra['selected_material_ids'] "
        "or MeshBuildConfig.channel_material_ids"
    )


def _build_meta(
    tsdf: TSDFVolume,
    cfg: MeshBuildConfig,
    *,
    extractor_name: str,
    extractor_version: str,
) -> Meta:
    parent = tsdf.meta
    generator_version = parent.generator_version if parent is not None else "0.1.0"
    git_commit = parent.git_commit if parent is not None else "unknown"
    input_hash = parent.input_hash if parent is not None else "unknown"
    base_extra = dict(parent.extra) if parent is not None else {}

    base_extra.update(
        {
            "mesh_backend": cfg.backend,
            "mesh_mode": cfg.mode,
            "mesh_extractor_name": extractor_name,
            "mesh_extractor_version": extractor_version,
            "mesh_sample_points_n": str(cfg.sample_points_n),
            "mesh_sample_seed": str(cfg.sample_seed),
        }
    )

    return Meta(
        schema_version=cfg.schema_version,
        profile_id=cfg.profile_id,
        config_hash=hash_config(cfg.to_hash_payload()),
        generator_version=generator_version,
        git_commit=git_commit,
        input_hash=input_hash,
        created_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        extra=base_extra,
    )


def build_mesh_from_tsdf(
    tsdf: TSDFVolume,
    cfg: MeshBuildConfig,
) -> tuple[MeshGeom, PointCloud, MeshQA]:
    if tsdf.tsdf.ndim != 4:
        raise ValueError(f"tsdf must be shape (M,Z,Y,X), got ndim={tsdf.tsdf.ndim}")
    if not np.isfinite(tsdf.tsdf).all():
        raise ValueError("tsdf contains NaN/Inf")

    channel_material_ids = resolve_channel_material_ids(tsdf, cfg)

    try:
        extractor = get_mesh_extractor(cfg.backend)
    except KeyError as exc:
        raise MeshBackendUnavailableError(f"unknown mesh backend: {cfg.backend}") from exc

    raw = extractor.extract_from_tsdf(tsdf, cfg, channel_material_ids)
    if raw.faces.shape[0] == 0:
        raise ValueError("mesh extraction produced no faces")

    postprocess_notes: list[str] = []
    postprocess_status: Literal["OK", "WARN", "FAIL"] | None = None
    post_bbox_shift_nm: float | None = None
    post_area_rel_error: float | None = None
    if cfg.backend == "vtk":
        vertices_pp, faces_pp, post_metrics = apply_vtk_visual_postprocess(
            raw.vertices,
            raw.faces,
            cfg,
        )
        raw = type(raw)(vertices=vertices_pp, faces=faces_pp)
        post_bbox_shift_nm = float(post_metrics["bbox_shift_nm"])
        post_area_rel_error = float(post_metrics["area_rel_error"])
        postprocess_status = "OK"
        exceeds = (
            post_bbox_shift_nm > float(cfg.qa_max_bbox_shift_nm)
            or post_area_rel_error > float(cfg.qa_max_area_rel_error)
        )
        if exceeds:
            postprocess_status = "WARN"
            msg = (
                "vtk postprocess exceeds thresholds: "
                f"bbox_shift_nm={post_bbox_shift_nm:.6f} "
                f"(max={cfg.qa_max_bbox_shift_nm}), "
                f"area_rel_error={post_area_rel_error:.6f} "
                f"(max={cfg.qa_max_area_rel_error})"
            )
            if cfg.qa_postprocess_on_exceed == "fail":
                raise ValueError(msg)
            postprocess_notes.append(msg)

    attrs = annotate_faces_from_tsdf(raw.vertices, raw.faces, tsdf, channel_material_ids)
    vertices, faces, attrs = apply_mesh_mode(raw.vertices, raw.faces, attrs, mode=cfg.mode)

    if faces.shape[0] == 0:
        raise ValueError("mesh mode filtering produced no faces")

    meta = _build_meta(
        tsdf,
        cfg,
        extractor_name=extractor.name,
        extractor_version=extractor.version,
    )

    mesh = MeshGeom(
        vertices=vertices.astype(np.float32, copy=False),
        faces=faces.astype(np.int32, copy=False),
        face_mat_in=attrs.face_mat_in.astype(np.int32, copy=False),
        face_mat_out=attrs.face_mat_out.astype(np.int32, copy=False),
        face_is_exposed=attrs.face_is_exposed.astype(bool, copy=False),
        grid=tsdf.grid,
        material=tsdf.material,
        meta=meta,
    )

    if cfg.backend == "vtk":
        mesh.meta.extra["vtk_postprocess_enabled"] = str(cfg.vtk_viz_postprocess_enabled)
        mesh.meta.extra["vtk_post_bbox_shift_nm"] = (
            f"{post_bbox_shift_nm:.10f}" if post_bbox_shift_nm is not None else ""
        )
        mesh.meta.extra["vtk_post_area_rel_error"] = (
            f"{post_area_rel_error:.10f}" if post_area_rel_error is not None else ""
        )
        mesh.meta.extra["vtk_postprocess_status"] = postprocess_status or ""

    point_cloud = sample_pointcloud(
        vertices=mesh.vertices,
        faces=mesh.faces,
        face_normals=attrs.face_normals,
        face_pair_code=attrs.face_pair_code,
        face_is_exposed=mesh.face_is_exposed,
        n_points=cfg.sample_points_n,
        seed=cfg.sample_seed,
        meta=meta,
    )

    qa = compute_mesh_qa(
        vertices=mesh.vertices,
        faces=mesh.faces,
        face_mat_in=mesh.face_mat_in,
        face_mat_out=mesh.face_mat_out,
        face_is_exposed=mesh.face_is_exposed,
        degenerate_area_eps=cfg.qa_degenerate_area_eps,
        post_bbox_shift_nm=post_bbox_shift_nm,
        post_area_rel_error=post_area_rel_error,
        postprocess_status=postprocess_status,
        postprocess_notes=postprocess_notes,
    )

    return mesh, point_cloud, qa
