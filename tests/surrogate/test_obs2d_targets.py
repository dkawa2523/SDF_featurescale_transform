from __future__ import annotations

from tests.surrogate.helpers import make_rows_for_build, write_input_manifest
from wafergeo.io.artifact_store import LocalDiskArtifactStore
from wafergeo.surrogate.builder import build_dataset_package
from wafergeo.surrogate.spec import DatasetBuildSpecV1, DatasetQASpec, GroupSplitSpec


def test_obs2d_target_ids_are_kept_in_linked_mode(tmp_path) -> None:
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
        profile_id="surrogate_obs_linked",
        dataset_id_prefix="obs_linked",
        task_kind="sdf",
        storage_mode="linked",
        include_sdf_features={"pair_code": False, "d_boundary": False, "present_mask": False},
        include_mesh=False,
        include_obs2d_pack=False,
        obs_targets={"topdown": "observer_hash_topdown"},
        param_spec_hash="param_hash_obs_linked",
        split=GroupSplitSpec(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=4),
        qa=DatasetQASpec(),
        input_manifest_path=str(in_manifest),
    )

    manifest = build_dataset_package(spec, store, output_dir=tmp_path / "dataset")
    assert manifest.samples[0].obs2d_sim_ids.get("topdown")
    assert all("obs2d.topdown.mask" not in sample.packed_paths for sample in manifest.samples)



def test_obs2d_mask_tsdf_can_be_packed(tmp_path) -> None:
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
        profile_id="surrogate_obs_packed",
        dataset_id_prefix="obs_packed",
        task_kind="sdf",
        storage_mode="packed",
        include_sdf_features={"pair_code": False, "d_boundary": False, "present_mask": False},
        include_mesh=False,
        include_obs2d_pack=True,
        obs_targets={"topdown": "observer_hash_topdown"},
        param_spec_hash="param_hash_obs_packed",
        split=GroupSplitSpec(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=5),
        qa=DatasetQASpec(),
        input_manifest_path=str(in_manifest),
    )

    manifest = build_dataset_package(spec, store, output_dir=tmp_path / "dataset")
    assert "obs2d.topdown.mask" in manifest.samples[0].packed_paths
    assert "obs2d.topdown.tsdf" in manifest.samples[0].packed_paths
