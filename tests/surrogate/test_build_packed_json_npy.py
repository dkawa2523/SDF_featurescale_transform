from __future__ import annotations

from pathlib import Path

from tests.surrogate.helpers import make_rows_for_build, write_input_manifest
from wafergeo.io.artifact_store import LocalDiskArtifactStore
from wafergeo.surrogate.builder import build_dataset_package
from wafergeo.surrogate.spec import DatasetBuildSpecV1, DatasetQASpec, GroupSplitSpec


def test_build_dataset_packed_writes_json_npy_layout(tmp_path) -> None:
    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    rows = make_rows_for_build(
        store,
        n_samples=3,
        include_tsdf=True,
        include_mesh=True,
        include_obs=True,
    )
    in_manifest = write_input_manifest(tmp_path / "input_manifest.json", rows)

    spec = DatasetBuildSpecV1(
        schema_version="surrogate_build/v1",
        profile_id="surrogate_hybrid_v1",
        dataset_id_prefix="hybrid_ds",
        task_kind="hybrid",
        storage_mode="packed",
        include_sdf_features={"pair_code": True, "d_boundary": True, "present_mask": True},
        include_mesh=True,
        include_obs2d_pack=True,
        obs_targets={"topdown": "observer_hash_topdown"},
        param_spec_hash="param_hash_packed",
        split=GroupSplitSpec(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=1),
        qa=DatasetQASpec(),
        input_manifest_path=str(in_manifest),
    )

    out_dir = tmp_path / "dataset_packed"
    manifest = build_dataset_package(spec, store, output_dir=out_dir)

    sample0 = manifest.samples[0]
    assert "sdf.tsdf" in sample0.packed_paths
    assert "mesh.pc_points" in sample0.packed_paths
    assert "obs2d.topdown.mask" in sample0.packed_paths

    for rel in sample0.packed_paths.values():
        assert (out_dir / Path(rel)).exists()
