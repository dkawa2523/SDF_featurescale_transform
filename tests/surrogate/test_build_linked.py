from __future__ import annotations

from tests.surrogate.helpers import make_rows_for_build, write_input_manifest
from wafergeo.io.artifact_store import LocalDiskArtifactStore
from wafergeo.surrogate.builder import build_dataset_package
from wafergeo.surrogate.manifest_io import read_dataset_manifest
from wafergeo.surrogate.spec import DatasetBuildSpecV1, DatasetQASpec, GroupSplitSpec


def test_build_dataset_linked_from_dummy_manifest(tmp_path) -> None:
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
        storage_mode="linked",
        include_sdf_features={"pair_code": True, "d_boundary": True, "present_mask": True},
        include_mesh=False,
        include_obs2d_pack=False,
        obs_targets={"topdown": "observer_hash_topdown"},
        param_spec_hash="param_hash_linked",
        split=GroupSplitSpec(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0),
        qa=DatasetQASpec(),
        input_manifest_path=str(in_manifest),
    )

    out_dir = tmp_path / "dataset_linked"
    manifest = build_dataset_package(spec, store, output_dir=out_dir)

    assert manifest.storage_mode == "linked"
    assert manifest.build_spec_hash
    assert manifest.param_spec_hash
    assert manifest.observer_spec_hashes["topdown"] == "observer_hash_topdown"
    assert all(len(sample.packed_paths) == 0 for sample in manifest.samples)

    reloaded = read_dataset_manifest(out_dir / "dataset_manifest.json")
    assert reloaded.dataset_id == manifest.dataset_id
    assert reloaded.profile_id == "surrogate_sdf_v1"
