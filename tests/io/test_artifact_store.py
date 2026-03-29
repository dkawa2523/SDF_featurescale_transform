from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest

from wafergeo.core.hashing import make_artifact_id
from wafergeo.core.meta import Meta
from wafergeo.io.artifact_store import LocalDiskArtifactStore


def _meta() -> Meta:
    return Meta(
        schema_version="schema/v1",
        profile_id="phase0",
        config_hash="cfg_hash",
        generator_version="0.1.0",
        git_commit="deadbeef",
        input_hash="input_hash",
        created_at=datetime.now(UTC).isoformat(),
        extra={"note": "roundtrip"},
    )


def test_local_disk_artifact_store_roundtrip_and_layout(tmp_path) -> None:
    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    payload = {
        "name": "sample",
        "array": np.arange(6, dtype=np.float32).reshape(2, 3),
        "nested": {"flag": True, "value": 3},
    }

    artifact_id = store.write("label", payload, _meta())
    assert store.exists(artifact_id)

    meta = store.read_meta(artifact_id)
    assert meta.schema_version == "schema/v1"
    assert meta.profile_id == "phase0"
    assert meta.config_hash == "cfg_hash"
    assert meta.generator_version == "0.1.0"
    assert meta.input_hash == "input_hash"

    loaded = store.load(artifact_id)
    assert loaded["name"] == "sample"
    assert loaded["nested"]["flag"] is True
    assert np.array_equal(loaded["array"], payload["array"])

    manifest_path = tmp_path / "artifacts" / "label" / artifact_id / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["payload_format"] == "json+npy"
    assert manifest["payload_format_version"] == 1
    assert manifest["schema_version"] == "schema/v1"
    assert manifest["profile_id"] == "phase0"
    assert manifest["config_hash"] == "cfg_hash"
    assert manifest["generator_version"] == "0.1.0"
    assert manifest["input_hash"] == "input_hash"


def test_local_disk_artifact_store_write_is_atomic_on_failure(tmp_path) -> None:
    class _FailingStore(LocalDiskArtifactStore):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.fail_on_array = False

        def _encode_payload(self, value, payload_dir, counter):
            out = super()._encode_payload(value, payload_dir, counter)
            if self.fail_on_array and isinstance(value, np.ndarray):
                self.fail_on_array = False
                raise RuntimeError("forced encode failure")
            return out

    store = _FailingStore(root=tmp_path / "artifacts")
    good_payload = {"name": "good", "array": np.arange(4, dtype=np.float32)}
    good_id = store.write("label", good_payload, _meta())
    loaded_good = store.load(good_id)
    assert loaded_good["name"] == "good"

    bad_meta = Meta(
        schema_version="schema/v1",
        profile_id="phase0",
        config_hash="cfg_hash_2",
        generator_version="0.1.0",
        git_commit="deadbeef",
        input_hash="input_hash_2",
        created_at=datetime.now(UTC).isoformat(),
        extra={"note": "atomic"},
    )
    bad_payload = {"name": "bad", "array": np.arange(8, dtype=np.float32)}
    store.fail_on_array = True
    bad_id = make_artifact_id(
        input_hash=bad_meta.input_hash,
        profile_id=bad_meta.profile_id,
        config_hash=bad_meta.config_hash,
        generator_version=bad_meta.generator_version,
    )

    with pytest.raises(RuntimeError, match="forced encode failure"):
        store.write("label", bad_payload, bad_meta)

    assert store.exists(good_id)
    reloaded_good = store.load(good_id)
    assert reloaded_good["name"] == "good"
    assert store.exists(bad_id) is False

    label_dir = tmp_path / "artifacts" / "label"
    temp_dirs = [p for p in label_dir.glob(".tmp_*") if p.is_dir()]
    assert temp_dirs == []
