from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from tests.sdf.helpers import register_bruteforce_engine
from wafergeo.core.grid import GridSpec
from wafergeo.observe.contour_extract import extract_contours_from_tsdf
from wafergeo.observe.errors import ObserverOptionalDependencyError
from wafergeo.observe.tsdf2d import tsdf2d_from_mask


def _grid2d() -> GridSpec:
    return GridSpec(
        dim=2,
        spacing=(10.0, 10.0),
        origin=(0.0, 0.0),
        axis_order="YX",
        sample_location="cell_center",
        units="nm",
    )


def test_extract_contours_missing_skimage_raises_helpful_error() -> None:
    if importlib.util.find_spec("skimage") is not None:
        pytest.skip("scikit-image is installed; missing-dependency path is not applicable")

    with pytest.raises(ObserverOptionalDependencyError) as exc_info:
        extract_contours_from_tsdf(
            np.zeros((4, 4), dtype=np.float32),
            _grid2d(),
            level=0.0,
            resample_points=16,
            backend="skimage",
        )

    assert "wafergeo[observe]" in str(exc_info.value)


def test_extract_contours_resample_point_count_when_skimage_available() -> None:
    pytest.importorskip("skimage")
    backend = register_bruteforce_engine("brute_obs_contour")

    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1
    tsdf = tsdf2d_from_mask(mask, _grid2d(), mu_nm=30.0, backend=backend)

    loops = extract_contours_from_tsdf(
        tsdf,
        _grid2d(),
        level=0.0,
        resample_points=32,
        backend="skimage",
    )

    assert len(loops) >= 1
    assert loops[0].points_xy.shape[0] == 32
