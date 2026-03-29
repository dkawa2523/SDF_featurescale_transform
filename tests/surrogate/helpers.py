from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wafergeo.core.grid import GridSpec
from wafergeo.core.meta import Meta
from wafergeo.core.types import ContourLoop, MaterialSpec, Obs2D
from wafergeo.io.artifact_store import LocalDiskArtifactStore
from wafergeo.sem.artifact import write_sem_obs_artifact
from wafergeo.sem.qa import SEMQA


def make_material() -> MaterialSpec:
    return MaterialSpec(
        ids=[0, 1],
        names=["void", "resist"],
        void_id=0,
        priority=[0, 10],
        ignore_in_exposure=[True, False],
    )


def make_grid3d() -> GridSpec:
    return GridSpec(
        dim=3,
        spacing=(10.0, 10.0, 10.0),
        origin=(0.0, 0.0, 0.0),
        axis_order="ZYX",
        sample_location="cell_center",
        units="nm",
    )


def make_grid2d() -> GridSpec:
    return GridSpec(
        dim=2,
        spacing=(10.0, 10.0),
        origin=(0.0, 0.0),
        axis_order="YX",
        sample_location="cell_center",
        units="nm",
    )


def make_meta(schema_version: str, profile_id: str, input_hash: str) -> Meta:
    return Meta(
        schema_version=schema_version,
        profile_id=profile_id,
        config_hash="cfg_hash",
        generator_version="0.1.0",
        git_commit="deadbeef",
        input_hash=input_hash,
        created_at=datetime.now(UTC).isoformat(),
        extra={},
    )


def _build_tsdf_payload(
    *,
    sample_idx: int,
    missing_material: bool,
) -> dict[str, object]:
    material = make_material()
    grid = make_grid3d()

    tsdf = np.ones((2, 1, 8, 8), dtype=np.float32)
    tsdf[1, :, 2:6, 2:6] = -0.5
    if missing_material:
        tsdf[1, ...] = 1.0

    pair_code = np.full((1, 8, 8), 255, dtype=np.uint8)
    pair_code[:, 2:6, 2:6] = 0
    d_boundary = np.ones((1, 8, 8), dtype=np.float32)
    d_boundary[:, 2:6, 2:6] = 0.25
    present_mask = np.array([True, not missing_material], dtype=bool)

    return {
        "grid": {
            "dim": grid.dim,
            "spacing": list(grid.spacing),
            "origin": list(grid.origin),
            "axis_order": grid.axis_order,
            "sample_location": grid.sample_location,
            "units": grid.units,
        },
        "materials": {
            "ids": list(material.ids),
            "names": list(material.names),
            "void_id": material.void_id,
            "priority": list(material.priority),
            "ignore_in_exposure": list(material.ignore_in_exposure),
        },
        "mu_nm": 30.0,
        "tsdf": tsdf,
        "pair_code": pair_code,
        "d_boundary": d_boundary,
        "present_mask": present_mask,
        "meta": {
            "schema_version": "tsdf/v1",
            "profile_id": "sdf_build",
            "config_hash": "cfg_hash",
            "generator_version": "0.1.0",
            "git_commit": "deadbeef",
            "input_hash": f"tsdf_input_{sample_idx}",
            "created_at": datetime.now(UTC).isoformat(),
            "extra": {
                "selected_material_ids": "0,1",
                "sample_idx": str(sample_idx),
            },
        },
    }


def write_tsdf_artifact(
    store: LocalDiskArtifactStore,
    *,
    sample_idx: int,
    missing_material: bool = False,
) -> str:
    payload = _build_tsdf_payload(sample_idx=sample_idx, missing_material=missing_material)
    meta = make_meta("tsdf/v1", "sdf_build", f"tsdf_input_{sample_idx}")
    return store.write("tsdf", payload, meta)


def write_mesh_artifact(store: LocalDiskArtifactStore, *, sample_idx: int) -> str:
    material = make_material()
    grid = make_grid3d()
    meta = make_meta("mesh/v1", "mesh_build", f"mesh_input_{sample_idx}")

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [0.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    n_points = 16
    rng = np.random.default_rng(seed=sample_idx)
    points = rng.uniform(low=0.0, high=10.0, size=(n_points, 3)).astype(np.float32)
    normals = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (n_points, 1))
    pair_code = np.zeros((n_points,), dtype=np.uint8)
    point_is_exposed = np.array([(idx % 2) == 0 for idx in range(n_points)], dtype=bool)

    payload = {
        "grid": {
            "dim": grid.dim,
            "spacing": list(grid.spacing),
            "origin": list(grid.origin),
            "axis_order": grid.axis_order,
            "sample_location": grid.sample_location,
            "units": grid.units,
        },
        "materials": {
            "ids": list(material.ids),
            "names": list(material.names),
            "void_id": material.void_id,
            "priority": list(material.priority),
            "ignore_in_exposure": list(material.ignore_in_exposure),
        },
        "meta": meta.to_dict(),
        "vertices": vertices,
        "faces": faces,
        "face_mat_in": np.array([1, 1], dtype=np.int32),
        "face_mat_out": np.array([0, 0], dtype=np.int32),
        "face_is_exposed": np.array([True, True], dtype=bool),
        "pc_points": points,
        "pc_normals": normals,
        "pc_pair_code": pair_code,
        "pc_is_exposed": point_is_exposed,
    }
    return store.write("mesh", payload, meta)


def write_obs2d_artifact(store: LocalDiskArtifactStore, *, sample_idx: int) -> str:
    grid2d = make_grid2d()
    meta = make_meta("sem_obs/v1", "sem_prepare", f"obs_input_{sample_idx}")

    mask = np.zeros((16, 16), dtype=np.uint8)
    x0 = 3 + sample_idx
    x1 = min(x0 + 6, mask.shape[1] - 1)
    mask[4:12, x0:x1] = 1
    tsdf = np.where(mask > 0, -0.5, 1.0).astype(np.float32)

    loop = ContourLoop(
        points_xy=np.array(
            [[30.0, 40.0], [90.0, 40.0], [90.0, 100.0], [30.0, 100.0], [30.0, 40.0]],
            dtype=np.float32,
        ),
        is_hole=False,
        label="outer",
        meta={},
    )

    obs = Obs2D(
        grid2d=grid2d,
        mask=mask,
        tsdf=tsdf,
        loops=[loop],
        weight=None,
        transform={"type": "identity"},
        debug_maps={},
        meta=meta,
    )
    return write_sem_obs_artifact(store, obs, SEMQA(status="OK"), extra_payload={})


def write_input_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"samples": rows}, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def make_rows_for_build(
    store: LocalDiskArtifactStore,
    *,
    n_samples: int = 3,
    include_tsdf: bool = True,
    include_mesh: bool = True,
    include_obs: bool = True,
    missing_material_at: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(n_samples):
        tsdf_id = None
        mesh_id = None
        obs_id = None

        if include_tsdf:
            tsdf_id = write_tsdf_artifact(
                store,
                sample_idx=idx,
                missing_material=(missing_material_at == idx),
            )
        if include_mesh:
            mesh_id = write_mesh_artifact(store, sample_idx=idx)
        if include_obs:
            obs_id = write_obs2d_artifact(store, sample_idx=idx)

        row: dict[str, object] = {
            "sample_id": f"sample_{idx}",
            "group_id": f"group_{idx}",
            "recipe_params": {"dose": float(idx), "focus": float(idx) * 0.1},
            "param_vector": [float(idx), float(idx) * 0.1],
            "obs2d_sim_ids": {},
        }
        if tsdf_id is not None:
            row["tsdf_artifact_id"] = tsdf_id
        if mesh_id is not None:
            row["mesh_artifact_id"] = mesh_id
        if obs_id is not None:
            row["obs2d_sim_ids"] = {"topdown": obs_id}
        rows.append(row)
    return rows
