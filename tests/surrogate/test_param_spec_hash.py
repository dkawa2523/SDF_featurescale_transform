from __future__ import annotations

from tests.surrogate.helpers import make_rows_for_build, write_input_manifest
from wafergeo.io.artifact_store import LocalDiskArtifactStore
from wafergeo.surrogate.builder import build_dataset_package
from wafergeo.surrogate.spec import DatasetBuildSpecV1, DatasetQASpec, GroupSplitSpec


def _build_spec(input_manifest_path: str, *, param_spec_hash: str) -> DatasetBuildSpecV1:
    return DatasetBuildSpecV1(
        schema_version="surrogate_build/v1",
        profile_id="surrogate_sdf_v1",
        dataset_id_prefix="sdf_ds",
        task_kind="sdf",
        storage_mode="linked",
        include_sdf_features={"pair_code": True, "d_boundary": True, "present_mask": True},
        include_mesh=False,
        include_obs2d_pack=False,
        obs_targets={"topdown": "observer_hash_topdown"},
        split=GroupSplitSpec(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0),
        qa=DatasetQASpec(),
        input_manifest_path=input_manifest_path,
        param_spec_hash=param_spec_hash,
    )


def test_param_spec_hash_uses_explicit_value(tmp_path) -> None:
    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    rows = make_rows_for_build(
        store,
        n_samples=3,
        include_tsdf=True,
        include_mesh=False,
        include_obs=True,
    )
    in_manifest = write_input_manifest(tmp_path / "input_manifest.json", rows)

    manifest = build_dataset_package(
        _build_spec(str(in_manifest), param_spec_hash="param_hash_v1"),
        store,
        output_dir=tmp_path / "dataset_a",
    )
    assert manifest.param_spec_hash == "param_hash_v1"


def test_param_spec_hash_changes_with_explicit_override(tmp_path) -> None:
    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    rows = make_rows_for_build(
        store,
        n_samples=3,
        include_tsdf=True,
        include_mesh=False,
        include_obs=True,
    )
    in_manifest = write_input_manifest(tmp_path / "input_manifest.json", rows)

    manifest_a = build_dataset_package(
        _build_spec(str(in_manifest), param_spec_hash="param_hash_a"),
        store,
        output_dir=tmp_path / "dataset_a",
    )
    manifest_b = build_dataset_package(
        _build_spec(str(in_manifest), param_spec_hash="param_hash_b"),
        store,
        output_dir=tmp_path / "dataset_b",
    )
    assert manifest_a.param_spec_hash != manifest_b.param_spec_hash
