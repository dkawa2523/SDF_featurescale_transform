from __future__ import annotations

from tests.surrogate.helpers import make_rows_for_build, write_input_manifest
from wafergeo.io.artifact_store import LocalDiskArtifactStore
from wafergeo.surrogate.builder import build_dataset_package
from wafergeo.surrogate.spec import DatasetBuildSpecV1, DatasetQASpec, GroupSplitSpec


def test_dataset_stats_and_qa_summary_are_computed(tmp_path) -> None:
    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    rows = make_rows_for_build(
        store,
        n_samples=4,
        include_tsdf=True,
        include_mesh=False,
        include_obs=True,
        missing_material_at=1,
    )
    in_manifest = write_input_manifest(tmp_path / "input_manifest.json", rows)

    spec = DatasetBuildSpecV1(
        schema_version="surrogate_build/v1",
        profile_id="surrogate_qa",
        dataset_id_prefix="qa_ds",
        task_kind="sdf",
        storage_mode="linked",
        include_sdf_features={"pair_code": True, "d_boundary": True, "present_mask": True},
        include_mesh=False,
        include_obs2d_pack=False,
        obs_targets={"topdown": "observer_hash_topdown"},
        param_spec_hash="param_hash_qa",
        split=GroupSplitSpec(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=6),
        qa=DatasetQASpec(warn_missing_material_rate_gt=0.0),
        input_manifest_path=str(in_manifest),
    )

    manifest = build_dataset_package(spec, store, output_dir=tmp_path / "dataset")
    assert "param_distribution" in manifest.stats
    assert "missing_material_rate_mean" in manifest.stats
    assert "interface_frequency_mean" in manifest.stats

    assert manifest.qa_summary["status"] in {"OK", "WARN", "FAIL"}
    assert "missing_material_rate_mean" in manifest.qa_summary


def test_dataset_build_can_fail_on_qa_status(tmp_path) -> None:
    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    rows = make_rows_for_build(
        store,
        n_samples=3,
        include_tsdf=True,
        include_mesh=False,
        include_obs=True,
    )
    in_manifest = write_input_manifest(tmp_path / "input_manifest.json", rows)

    fail_spec = DatasetBuildSpecV1(
        schema_version="surrogate_build/v1",
        profile_id="surrogate_qa_fail",
        dataset_id_prefix="qa_fail_ds",
        task_kind="sdf",
        storage_mode="linked",
        include_sdf_features={"pair_code": True, "d_boundary": True, "present_mask": True},
        include_mesh=False,
        include_obs2d_pack=False,
        obs_targets={"topdown": "observer_hash_topdown"},
        param_spec_hash="param_hash_qa_fail",
        split=GroupSplitSpec(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=6),
        qa=DatasetQASpec(require_material_count_leq=1),
        input_manifest_path=str(in_manifest),
        fail_on_qa_status=True,
    )

    import pytest

    with pytest.raises(ValueError, match="fail_on_qa_status=true"):
        build_dataset_package(fail_spec, store, output_dir=tmp_path / "dataset_fail")
