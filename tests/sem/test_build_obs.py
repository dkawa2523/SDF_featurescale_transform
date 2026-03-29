from __future__ import annotations

import numpy as np

from tests.sdf.helpers import register_bruteforce_engine
from tests.sem.helpers import build_sem_spec
from wafergeo.sem.build_obs import build_sem_obs2d
from wafergeo.sem.normalize import build_transform_chain, normalize_contours
from wafergeo.sem.readers import RawContourLoop, RawContourSet


def _closed_square_raw() -> RawContourSet:
    return RawContourSet(
        coord_system="nm",
        units="nm",
        loops_raw=[
            RawContourLoop(
                loop_id="outer_0",
                role="outer",
                points_xy=np.array(
                    [[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0], [20.0, 20.0]],
                    dtype=np.float32,
                ),
                is_closed_hint=True,
            )
        ],
    )


def _open_line_raw() -> RawContourSet:
    return RawContourSet(
        coord_system="nm",
        units="nm",
        loops_raw=[
            RawContourLoop(
                loop_id="open_0",
                role="open_curve",
                points_xy=np.array([[20.0, 30.0], [60.0, 30.0], [100.0, 40.0]], dtype=np.float32),
                is_closed_hint=False,
            )
        ],
    )


def test_build_closed_contour_obs2d_signed_tsdf() -> None:
    backend = register_bruteforce_engine("brute_sem_closed")
    spec = build_sem_spec(
        coord_system="nm",
        units="nm",
        tsdf_mode="signed_region",
        distance_backend=backend,
    )
    chain = build_transform_chain(spec, image_shape=None)
    normalized = normalize_contours(_closed_square_raw(), spec, chain)

    obs, qa, extra = build_sem_obs2d(
        normalized,
        spec,
        source_contour_path="contours.json",
        source_image_path=None,
        image_raw=None,
        transform_chain=chain,
        input_hash="input_hash",
        generator_version="0.1.0",
        git_commit="deadbeef",
    )

    assert obs.mask.ndim == 2
    assert obs.mask.shape == obs.tsdf.shape
    assert float(obs.tsdf.min()) >= -1.0001
    assert float(obs.tsdf.max()) <= 1.0001
    assert int(np.sum(obs.mask)) > 0
    assert qa.status in {"OK", "WARN"}
    assert extra["distance_type"] == "signed_region"


def test_build_open_contour_obs2d_unsigned_curve() -> None:
    spec = build_sem_spec(coord_system="nm", units="nm", tsdf_mode="unsigned_curve")
    chain = build_transform_chain(spec, image_shape=None)
    normalized = normalize_contours(_open_line_raw(), spec, chain)

    obs, qa, extra = build_sem_obs2d(
        normalized,
        spec,
        source_contour_path="contours.csv",
        source_image_path=None,
        image_raw=None,
        transform_chain=chain,
        input_hash="input_hash_open",
        generator_version="0.1.0",
        git_commit="deadbeef",
    )

    assert float(obs.tsdf.min()) >= -1e-6
    assert float(obs.tsdf.max()) <= 1.0001
    assert int(np.sum(obs.mask)) > 0
    assert qa.open_contour_count >= 1
    assert extra["distance_type"] == "unsigned_curve"
