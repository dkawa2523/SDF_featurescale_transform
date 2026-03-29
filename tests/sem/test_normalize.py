from __future__ import annotations

import numpy as np

from tests.sem.helpers import build_sem_spec
from wafergeo.sem.normalize import build_transform_chain, normalize_contours
from wafergeo.sem.readers import RawContourLoop, RawContourSet


def _signed_area(points_xy: np.ndarray) -> float:
    pts = points_xy
    if not np.allclose(pts[0], pts[-1], rtol=0.0, atol=1e-6):
        pts = np.vstack([pts, pts[0]])
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def test_build_transform_chain_flip_y() -> None:
    spec = build_sem_spec(
        coord_system="pixel",
        units="px",
        pixel_size_nm=2.0,
        pixel_y_policy="flip_y",
    )
    chain = build_transform_chain(spec, image_shape=(4, 5))
    expected = np.array([[2.0, 0.0, 0.0], [0.0, -2.0, 6.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    assert np.allclose(chain.T_px_to_sem_nm, expected)


def test_normalize_contours_enforces_orientation_and_resamples() -> None:
    spec = build_sem_spec(coord_system="nm", units="nm")
    raw = RawContourSet(
        coord_system="nm",
        units="nm",
        loops_raw=[
            RawContourLoop(
                loop_id="outer0",
                role="outer",
                # clockwise order on purpose
                points_xy=np.array(
                    [[10.0, 10.0], [10.0, 40.0], [40.0, 40.0], [40.0, 10.0], [10.0, 10.0]],
                    dtype=np.float32,
                ),
                is_closed_hint=True,
            ),
            RawContourLoop(
                loop_id="line0",
                role="open_curve",
                points_xy=np.array([[5.0, 5.0], [20.0, 25.0], [35.0, 35.0]], dtype=np.float32),
                is_closed_hint=False,
            ),
        ],
    )
    chain = build_transform_chain(spec, image_shape=None)
    loops = normalize_contours(raw, spec, chain)
    assert len(loops) == 2

    closed = loops[0]
    open_curve = loops[1]
    assert closed.is_closed is True
    assert open_curve.is_closed is False
    assert closed.points_sim_nm.shape[0] == spec.normalize.resample_points_closed
    assert open_curve.points_sim_nm.shape[0] == spec.normalize.resample_points_open
    assert _signed_area(closed.points_sim_nm) > 0.0
