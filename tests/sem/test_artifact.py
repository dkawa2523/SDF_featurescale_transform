from __future__ import annotations

import numpy as np

from tests.sdf.helpers import register_bruteforce_engine
from tests.sem.helpers import build_sem_spec
from wafergeo.io.artifact_store import LocalDiskArtifactStore
from wafergeo.sem.artifact import (
    build_sem_obs_artifact_payload,
    read_sem_obs_artifact,
    write_sem_obs_artifact,
)
from wafergeo.sem.build_obs import build_sem_obs2d
from wafergeo.sem.normalize import build_transform_chain, normalize_contours
from wafergeo.sem.readers import RawContourLoop, RawContourSet


def test_write_sem_obs_artifact_contains_qa_transform_and_overlay(tmp_path) -> None:
    backend = register_bruteforce_engine("brute_sem_artifact")
    spec = build_sem_spec(
        coord_system="nm",
        units="nm",
        tsdf_mode="signed_region",
        distance_backend=backend,
        overlay_enable=True,
    )
    chain = build_transform_chain(spec, image_shape=None)
    raw = RawContourSet(
        coord_system="nm",
        units="nm",
        loops_raw=[
            RawContourLoop(
                loop_id="outer_0",
                role="outer",
                points_xy=np.array(
                    [[20.0, 20.0], [60.0, 20.0], [60.0, 60.0], [20.0, 60.0], [20.0, 20.0]],
                    dtype=np.float32,
                ),
                is_closed_hint=True,
            )
        ],
    )
    normalized = normalize_contours(raw, spec, chain)
    obs, qa, extra_payload = build_sem_obs2d(
        normalized,
        spec,
        source_contour_path="contours.json",
        source_image_path=None,
        image_raw=None,
        transform_chain=chain,
        input_hash="sem_input_hash",
        generator_version="0.1.0",
        git_commit="deadbeef",
    )

    payload = build_sem_obs_artifact_payload(obs, qa, extra_payload)
    assert "qa" in payload
    assert "transform_chain" in payload
    assert "overlay" in payload

    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    artifact_id = write_sem_obs_artifact(store, obs, qa, extra_payload)
    loaded = store.load(artifact_id)
    assert "qa" in loaded
    assert "transform_chain" in loaded
    assert "overlay" in loaded

    loaded_obs, loaded_payload = read_sem_obs_artifact(store, artifact_id)
    assert loaded_obs.mask.shape == obs.mask.shape
    assert loaded_obs.tsdf.shape == obs.tsdf.shape
    assert "qa" in loaded_payload
