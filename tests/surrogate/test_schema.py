from __future__ import annotations

from datetime import UTC, datetime

from wafergeo.surrogate.schema import DatasetManifest, SampleRecord


def test_sample_record_roundtrip() -> None:
    sample = SampleRecord(
        sample_id="s1",
        group_id="g1",
        recipe_params={"dose": 1.2},
        param_vector=[1.2, 0.3],
        tsdf_artifact_id="a_tsdf",
        obs2d_sim_ids={"topdown": "a_obs"},
        packed_paths={"sdf.tsdf": "packed/s1/sdf/tsdf.npy"},
        qa={"status": "OK"},
        meta={"k": "v"},
    )

    loaded = SampleRecord.from_dict(sample.to_dict())
    assert loaded.sample_id == sample.sample_id
    assert loaded.obs2d_sim_ids == {"topdown": "a_obs"}
    assert loaded.packed_paths["sdf.tsdf"].endswith("tsdf.npy")


def test_dataset_manifest_roundtrip() -> None:
    now = datetime.now(UTC).isoformat()
    manifest = DatasetManifest(
        schema_version="surrogate_dataset/v3",
        dataset_id="ds_1",
        profile_id="p1",
        created_at=now,
        generator_version="0.1.0",
        git_commit="deadbeef",
        storage_mode="linked",
        task_kind="sdf",
        materials={"ids": [0, 1], "void_id": 0},
        grid3d={"dim": 3, "spacing": [1.0, 1.0, 1.0]},
        observers=["topdown"],
        param_spec_hash="phash",
        observer_spec_hashes={"topdown": "ohash"},
        build_spec_hash="bhash",
        samples=[SampleRecord(sample_id="s1", group_id="g1")],
        splits={"train": ["s1"], "val": ["s2"], "test": ["s3"]},
        stats={"num_samples": 1},
        qa_summary={"status": "OK"},
    )

    loaded = DatasetManifest.from_dict(manifest.to_dict())
    assert loaded.dataset_id == "ds_1"
    assert loaded.storage_mode == "linked"
    assert loaded.samples[0].sample_id == "s1"
