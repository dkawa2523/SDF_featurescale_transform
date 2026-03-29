from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from wafergeo.core.types import MaterialSpec
from wafergeo.io.artifact_store import LocalDiskArtifactStore
from wafergeo.io.vti_reader import RawVtiImage
from wafergeo.label.artifact import build_label_artifact_payload, write_label_artifact
from wafergeo.label.normalize import LabelNormalizeConfig, normalize_raw_to_label


def _material() -> MaterialSpec:
    return MaterialSpec(
        ids=[0, 1, 2],
        names=["void", "resist", "oxide"],
        void_id=0,
        priority=[0, 10, 20],
        ignore_in_exposure=[True, False, False],
    )


def test_write_label_artifact_contains_qa(tmp_path) -> None:
    raw = RawVtiImage(
        spacing_xyz=(10.0, 20.0, 30.0),
        origin_xyz=(100.0, 200.0, 300.0),
        dims_xyz=(3, 2, 1),
        arrays={"material_id": np.array([[[0], [1]], [[1], [2]], [[0], [2]]], dtype=np.int32)},
        array_location={"material_id": "cell"},
        vtk_meta={},
    )

    label, qa = normalize_raw_to_label(
        raw,
        _material(),
        LabelNormalizeConfig(),
        source_path="synthetic.vti",
        input_hash="input-hash",
        generator_version="0.1.0",
        git_commit="deadbeef",
        created_at=datetime.now(UTC).isoformat(),
    )

    payload = build_label_artifact_payload(label, qa)
    assert "qa" in payload
    assert "material_id" in payload

    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    artifact_id = write_label_artifact(store, label, qa)
    loaded = store.load(artifact_id)

    assert "qa" in loaded
    assert "unknown_count" in loaded["qa"]
