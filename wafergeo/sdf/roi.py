from __future__ import annotations

from math import ceil

import numpy as np

from wafergeo.sdf.errors import ShapeMismatchError


def expand_roi_with_margin(
    roi_zyx: tuple[slice, slice, slice],
    shape_zyx: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
    margin_nm: float,
) -> tuple[slice, slice, slice]:
    if margin_nm < 0:
        raise ValueError(f"margin_nm must be >= 0, got {margin_nm}")

    expanded: list[slice] = []
    for axis, (roi_axis, axis_size, spacing) in enumerate(
        zip(roi_zyx, shape_zyx, spacing_zyx, strict=True)
    ):
        if roi_axis.step not in (None, 1):
            raise ValueError("roi slices with step are not supported")
        start = 0 if roi_axis.start is None else int(roi_axis.start)
        stop = axis_size if roi_axis.stop is None else int(roi_axis.stop)
        if start < 0 or stop > axis_size or start >= stop:
            raise ValueError(f"invalid roi_zyx axis={axis}: start={start}, stop={stop}")
        margin_vox = int(ceil(float(margin_nm) / float(spacing)))
        expanded.append(slice(max(0, start - margin_vox), min(axis_size, stop + margin_vox), 1))

    return expanded[0], expanded[1], expanded[2]


def place_subvolume_back(
    full_shape: tuple[int, ...],
    roi: tuple[slice, slice, slice],
    sub: np.ndarray,
    fill_value: float | int,
) -> np.ndarray:
    full = np.full(full_shape, fill_value, dtype=sub.dtype)
    if len(full_shape) == 3:
        expected = tuple(s.stop - s.start for s in roi)
        if sub.shape != expected:
            raise ShapeMismatchError(f"sub shape {sub.shape} does not match ROI shape {expected}")
        full[roi[0], roi[1], roi[2]] = sub
        return full

    if len(full_shape) == 4:
        expected = tuple(s.stop - s.start for s in roi)
        if sub.shape[1:] != expected:
            raise ShapeMismatchError(
                "sub shape "
                f"{sub.shape} does not match channel+ROI shape "
                f"(*,{expected[0]},{expected[1]},{expected[2]})"
            )
        if sub.shape[0] != full_shape[0]:
            raise ShapeMismatchError(
                "sub channel count "
                f"{sub.shape[0]} does not match full channel count {full_shape[0]}"
            )
        full[:, roi[0], roi[1], roi[2]] = sub
        return full

    raise ShapeMismatchError(f"full_shape must be rank 3 or 4, got rank={len(full_shape)}")
