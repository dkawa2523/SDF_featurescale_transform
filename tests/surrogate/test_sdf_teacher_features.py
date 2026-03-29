from __future__ import annotations

from tests.surrogate.helpers import make_rows_for_build, write_input_manifest
from wafergeo.io.artifact_store import LocalDiskArtifactStore
from wafergeo.surrogate.builder import build_dataset_package
from wafergeo.surrogate.spec import DatasetBuildSpecV1, DatasetQASpec, GroupSplitSpec


def test_sdf_optional_features_are_collected_when_present(tmp_path) -> None:
    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    rows = make_rows_for_build(
        store,
        n_samples=3,
        include_tsdf=True,
        include_mesh=False,
        include_obs=True,
    )
    in_manifest = write_input_manifest(tmp_path / "input_manifest.json", rows)

    spec = DatasetBuildSpecV1(
        schema_version="surrogate_build/v1",
        profile_id="surrogate_sdf_v1",
        dataset_id_prefix="sdf_ds",
        task_kind="sdf",
        storage_mode="packed",
        include_sdf_features={"pair_code": True, "d_boundary": True, "present_mask": True},
        include_mesh=False,
        include_obs2d_pack=False,
        obs_targets={"topdown": "observer_hash_topdown"},
        param_spec_hash="param_hash_sdf_features",
        split=GroupSplitSpec(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=2),
        qa=DatasetQASpec(),
        input_manifest_path=str(in_manifest),
    )

    manifest = build_dataset_package(spec, store, output_dir=tmp_path / "dataset")
    sample0 = manifest.samples[0]

    assert "sdf.pair_code" in sample0.packed_paths
    assert "sdf.d_boundary" in sample0.packed_paths
    assert "sdf.present_mask" in sample0.packed_paths
