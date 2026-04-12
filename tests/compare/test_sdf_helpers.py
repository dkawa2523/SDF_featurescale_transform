from __future__ import annotations

import numpy as np
import pytest

from wafergeo.compare.sdf_helpers import (
    clipped_signed_distance_from_mask_2d,
    distance_transform_2d,
    signed_distance_from_mask_2d,
    tsdf_from_sdf_nm,
    unsigned_distance_from_mask_2d,
)


def test_signed_and_unsigned_distance_helpers_share_expected_convention() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    signed = signed_distance_from_mask_2d(mask, (1.0, 1.0))
    unsigned = unsigned_distance_from_mask_2d(mask, (1.0, 1.0))

    assert signed.shape == mask.shape
    assert unsigned.shape == mask.shape
    assert signed[2, 2] < 0.0
    assert signed[0, 0] > 0.0
    assert unsigned[2, 2] == pytest.approx(0.0)
    assert unsigned[0, 0] > 0.0


def test_clipped_signed_distance_and_tsdf_are_bounded() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    clipped = clipped_signed_distance_from_mask_2d(mask, (1.0, 1.0), clip_nm=1.0)
    tsdf = tsdf_from_sdf_nm(clipped, clip_nm=1.0)

    assert float(np.max(clipped)) <= 1.0
    assert float(np.min(clipped)) >= -1.0
    assert float(np.max(tsdf)) <= 1.0
    assert float(np.min(tsdf)) >= -1.0


def test_tsdf_rejects_invalid_clip() -> None:
    with pytest.raises(ValueError, match="clip_nm"):
        tsdf_from_sdf_nm(np.zeros((2, 2), dtype=np.float32), clip_nm=0.0)


def test_distance_helpers_reject_non_2d_masks_and_invalid_spacing() -> None:
    with pytest.raises(ValueError, match="2D"):
        distance_transform_2d(np.zeros((1, 2, 2), dtype=bool), (1.0, 1.0))

    with pytest.raises(ValueError, match="spacing_yx"):
        signed_distance_from_mask_2d(np.zeros((2, 2), dtype=bool), (1.0, 0.0))


def test_clipped_distance_rejects_invalid_clip() -> None:
    with pytest.raises(ValueError, match="clip_nm"):
        clipped_signed_distance_from_mask_2d(
            np.zeros((2, 2), dtype=bool),
            (1.0, 1.0),
            clip_nm=-1.0,
        )
