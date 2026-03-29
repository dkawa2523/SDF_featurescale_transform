from __future__ import annotations

import numpy as np

from tests.sdf.helpers import register_bruteforce_engine
from wafergeo.core.grid import GridSpec
from wafergeo.observe.tsdf2d import tsdf2d_from_mask


def test_tsdf2d_from_mask_value_range_and_sign() -> None:
    backend = register_bruteforce_engine("brute_obs_tsdf2d")
    grid2d = GridSpec(
        dim=2,
        spacing=(10.0, 10.0),
        origin=(0.0, 0.0),
        axis_order="YX",
        sample_location="cell_center",
        units="nm",
    )
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1

    tsdf = tsdf2d_from_mask(mask, grid2d, mu_nm=20.0, backend=backend)

    assert tsdf.shape == (5, 5)
    assert float(tsdf.min()) >= -1.0
    assert float(tsdf.max()) <= 1.0
    assert float(tsdf[2, 2]) <= 0.0
    assert float(tsdf[0, 0]) >= 0.0
