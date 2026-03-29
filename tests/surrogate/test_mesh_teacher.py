from __future__ import annotations

import numpy as np

from tests.surrogate.helpers import make_rows_for_build, write_input_manifest
from wafergeo.io.artifact_store import LocalDiskArtifactStore
from wafergeo.surrogate.builder import build_dataset_package
from wafergeo.surrogate.spec import DatasetBuildSpecV1, DatasetQASpec, GroupSplitSpec


def test_mesh_teacher_pointcloud_is_packed_with_fixed_arrays(tmp_path) -> None:
    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    rows = make_rows_for_build(
        store,
        n_samples=3,
        include_tsdf=False,
        include_mesh=True,
        include_obs=True,
    )
    in_manifest = write_input_manifest(tmp_path / "input_manifest.json", rows)

    spec = DatasetBuildSpecV1(
        schema_version="surrogate_build/v1",
        profile_id="surrogate_mesh_v1",
        dataset_id_prefix="mesh_ds",
        task_kind="mesh",
        storage_mode="packed",
        include_sdf_features={"pair_code": False, "d_boundary": False, "present_mask": False},
        include_mesh=True,
        include_obs2d_pack=False,
        obs_targets={"topdown": "observer_hash_topdown"},
        param_spec_hash="param_hash_mesh",
        split=GroupSplitSpec(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=3),
        qa=DatasetQASpec(),
        input_manifest_path=str(in_manifest),
    )

    out_dir = tmp_path / "dataset"
    manifest = build_dataset_package(spec, store, output_dir=out_dir)
    sample0 = manifest.samples[0]

    points = np.load(out_dir / sample0.packed_paths["mesh.pc_points"], allow_pickle=False)
    normals = np.load(out_dir / sample0.packed_paths["mesh.pc_normals"], allow_pickle=False)
    pair_code = np.load(out_dir / sample0.packed_paths["mesh.pc_pair_code"], allow_pickle=False)
    exposed = np.load(out_dir / sample0.packed_paths["mesh.pc_is_exposed"], allow_pickle=False)

    assert points.shape == normals.shape
    assert points.ndim == 2 and points.shape[1] == 3
    assert pair_code.shape[0] == points.shape[0]
    assert exposed.shape[0] == points.shape[0]
