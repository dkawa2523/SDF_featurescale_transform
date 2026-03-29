from __future__ import annotations

from typing import Any, cast

import numpy as np

from wafergeo.core.grid import AxisOrder, GridSpec, SampleLocation
from wafergeo.core.meta import Meta
from wafergeo.core.types import (
    ContourLoop,
    LabelVolume,
    MaterialSpec,
    MeshGeom,
    Obs2D,
    PointCloud,
    TSDFVolume,
)
from wafergeo.io.artifact_store import ArtifactStore
from wafergeo.sem.artifact import obs2d_from_sem_obs_payload


def _to_grid3d(raw: dict[str, Any]) -> GridSpec:
    spacing = tuple(float(v) for v in raw["spacing"])
    origin = tuple(float(v) for v in raw["origin"])
    return GridSpec(
        dim=int(raw["dim"]),
        spacing=cast(tuple[float, ...], spacing),
        origin=cast(tuple[float, ...], origin),
        axis_order=cast(AxisOrder, str(raw["axis_order"])),
        sample_location=cast(SampleLocation, str(raw["sample_location"])),
        units=str(raw["units"]),
    )


def _to_grid2d(raw: dict[str, Any]) -> GridSpec:
    return GridSpec(
        dim=int(raw["dim"]),
        spacing=(float(raw["spacing"][0]), float(raw["spacing"][1])),
        origin=(float(raw["origin"][0]), float(raw["origin"][1])),
        axis_order=cast(AxisOrder, str(raw["axis_order"])),
        sample_location=cast(SampleLocation, str(raw["sample_location"])),
        units=str(raw["units"]),
    )


def _to_material(raw: dict[str, Any]) -> MaterialSpec:
    return MaterialSpec(
        ids=[int(v) for v in list(raw["ids"])],
        names=[str(v) for v in list(raw["names"])],
        void_id=int(raw["void_id"]),
        priority=[int(v) for v in list(raw["priority"])],
        ignore_in_exposure=[bool(v) for v in list(raw["ignore_in_exposure"])],
    )


def load_label_from_artifact(store: ArtifactStore, artifact_id: str) -> LabelVolume:
    payload = store.load(artifact_id)
    if not isinstance(payload, dict):
        raise ValueError("label artifact payload must be mapping")

    if "material_id" not in payload:
        raise ValueError("label payload missing 'material_id'")

    grid_key = "grid" if "grid" in payload else "grid3d"
    mat_key = "materials" if "materials" in payload else "material"
    meta_key = "meta"

    return LabelVolume(
        grid=_to_grid3d(dict(payload[grid_key])),
        material=_to_material(dict(payload[mat_key])),
        material_id=np.asarray(payload["material_id"], dtype=np.uint8),
        meta=Meta.from_dict(dict(payload[meta_key])),
    )


def load_tsdf_from_artifact(store: ArtifactStore, artifact_id: str) -> TSDFVolume:
    payload = store.load(artifact_id)
    if not isinstance(payload, dict):
        raise ValueError("tsdf artifact payload must be mapping")

    tsdf = np.asarray(payload["tsdf"], dtype=np.float32)
    d_boundary_raw = payload.get("d_boundary")
    pair_code_raw = payload.get("pair_code")
    present_mask_raw = payload.get("present_mask")

    return TSDFVolume(
        grid=_to_grid3d(dict(payload["grid"])),
        material=_to_material(dict(payload["materials"])),
        mu_nm=float(payload["mu_nm"]),
        tsdf=tsdf,
        d_boundary=(
            None if d_boundary_raw is None else np.asarray(d_boundary_raw, dtype=np.float32)
        ),
        pair_code=None if pair_code_raw is None else np.asarray(pair_code_raw, dtype=np.uint8),
        present_mask=(
            None if present_mask_raw is None else np.asarray(present_mask_raw, dtype=bool)
        ),
        meta=(
            None
            if payload.get("meta") is None
            else Meta.from_dict(dict(cast(dict[str, Any], payload["meta"])))
        ),
    )


def load_mesh_from_artifact(
    store: ArtifactStore,
    artifact_id: str,
) -> tuple[MeshGeom | None, PointCloud | None]:
    payload = store.load(artifact_id)
    if not isinstance(payload, dict):
        raise ValueError("mesh artifact payload must be mapping")

    mesh: MeshGeom | None = None
    point_cloud: PointCloud | None = None

    grid_raw = payload.get("grid")
    materials_raw = payload.get("materials")
    meta_raw = payload.get("meta")
    if grid_raw is None or materials_raw is None or meta_raw is None:
        raise ValueError("mesh payload requires grid/materials/meta")

    grid = _to_grid3d(dict(grid_raw))
    material = _to_material(dict(materials_raw))
    meta = Meta.from_dict(dict(meta_raw))

    if "vertices" in payload and "faces" in payload:
        mesh = MeshGeom(
            vertices=np.asarray(payload["vertices"], dtype=np.float32),
            faces=np.asarray(payload["faces"], dtype=np.int32),
            face_mat_in=np.asarray(payload["face_mat_in"], dtype=np.int32),
            face_mat_out=np.asarray(payload["face_mat_out"], dtype=np.int32),
            face_is_exposed=np.asarray(payload["face_is_exposed"], dtype=bool),
            grid=grid,
            material=material,
            meta=meta,
        )

    if "pointcloud" in payload and isinstance(payload["pointcloud"], dict):
        pc_raw = cast(dict[str, Any], payload["pointcloud"])
        point_cloud = PointCloud(
            points=np.asarray(pc_raw["points"], dtype=np.float32),
            normals=np.asarray(pc_raw["normals"], dtype=np.float32),
            pair_code=np.asarray(pc_raw["pair_code"], dtype=np.uint8),
            point_is_exposed=np.asarray(pc_raw["point_is_exposed"], dtype=bool),
            meta=meta,
        )
    elif "pc_points" in payload:
        point_cloud = PointCloud(
            points=np.asarray(payload["pc_points"], dtype=np.float32),
            normals=np.asarray(payload["pc_normals"], dtype=np.float32),
            pair_code=np.asarray(payload["pc_pair_code"], dtype=np.uint8),
            point_is_exposed=np.asarray(payload["pc_is_exposed"], dtype=bool),
            meta=meta,
        )

    return mesh, point_cloud


def _to_loops(raw: list[dict[str, Any]]) -> list[ContourLoop]:
    loops: list[ContourLoop] = []
    for row in raw:
        loops.append(
            ContourLoop(
                points_xy=np.asarray(row["points_xy"], dtype=np.float32),
                is_hole=bool(row.get("is_hole", False)),
                label=None if row.get("label") is None else str(row["label"]),
                meta={str(k): v for k, v in dict(row.get("meta", {})).items()},
            )
        )
    return loops


def load_obs2d_from_artifact(store: ArtifactStore, artifact_id: str) -> Obs2D:
    payload = store.load(artifact_id)
    if not isinstance(payload, dict):
        raise ValueError("obs2d artifact payload must be mapping")

    if "contours" in payload and "grid2d" in payload:
        return obs2d_from_sem_obs_payload(payload)

    if "mask" not in payload or "tsdf" not in payload:
        raise ValueError("obs2d payload missing required keys: mask/tsdf")

    loops_raw = payload.get("loops", payload.get("contours", []))
    if not isinstance(loops_raw, list):
        raise ValueError("obs2d loops/contours must be list")

    grid_raw = payload.get("grid2d")
    if grid_raw is None:
        raise ValueError("obs2d payload missing grid2d")

    transform = payload.get("transform")
    weight = payload.get("weight")
    debug_maps_raw = payload.get("debug_maps", {})

    return Obs2D(
        grid2d=_to_grid2d(dict(grid_raw)),
        mask=np.asarray(payload["mask"], dtype=np.uint8),
        tsdf=np.asarray(payload["tsdf"], dtype=np.float32),
        loops=_to_loops([dict(v) for v in loops_raw]),
        weight=None if weight is None else np.asarray(weight, dtype=np.float32),
        transform=None if transform is None else dict(transform),
        debug_maps={str(k): np.asarray(v) for k, v in dict(debug_maps_raw).items()},
        meta=Meta.from_dict(dict(payload["meta"])),
    )
