from __future__ import annotations

import numpy as np


def _as_2d_bool_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2D [Y,X], got ndim={binary.ndim}")
    return binary


def _validate_spacing_yx(spacing_yx: tuple[float, float]) -> tuple[float, float]:
    if len(spacing_yx) != 2:
        raise ValueError(f"spacing_yx must have length 2, got {len(spacing_yx)}")
    spacing = (float(spacing_yx[0]), float(spacing_yx[1]))
    for i, value in enumerate(spacing):
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError(f"spacing_yx[{i}] must be finite and > 0, got {value}")
    return spacing


def _validate_positive_nm(name: str, value: float) -> float:
    checked = float(value)
    if checked <= 0.0 or not np.isfinite(checked):
        raise ValueError(f"{name} must be finite and > 0, got {value}")
    return checked


def distance_transform_2d(mask: np.ndarray, sampling_yx: tuple[float, float]) -> np.ndarray:
    binary = _as_2d_bool_mask(mask)
    sampling = _validate_spacing_yx(sampling_yx)
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        yy, xx = np.meshgrid(
            np.arange(binary.shape[0], dtype=np.float32),
            np.arange(binary.shape[1], dtype=np.float32),
            indexing="ij",
        )
        true_rows, true_cols = np.nonzero(binary)
        true_points = np.column_stack([true_rows, true_cols]).astype(np.float32, copy=False)
        if true_points.size == 0:
            return np.full(binary.shape, np.inf, dtype=np.float32)
        dy = (yy[..., None] - true_points[:, 0]) * sampling[0]
        dx = (xx[..., None] - true_points[:, 1]) * sampling[1]
        return np.sqrt(np.min(dy * dy + dx * dx, axis=-1)).astype(np.float32)
    return distance_transform_edt(binary, sampling=sampling).astype(np.float32)


def signed_distance_from_mask_2d(mask: np.ndarray, spacing_yx: tuple[float, float]) -> np.ndarray:
    binary = _as_2d_bool_mask(mask)
    spacing = _validate_spacing_yx(spacing_yx)
    if not np.any(binary):
        return np.full(binary.shape, 1e6, dtype=np.float32)
    if np.all(binary):
        return np.full(binary.shape, -1e6, dtype=np.float32)
    outside = distance_transform_2d(~binary, spacing)
    inside = distance_transform_2d(binary, spacing)
    return (outside - inside).astype(np.float32)


def unsigned_distance_from_mask_2d(mask: np.ndarray, spacing_yx: tuple[float, float]) -> np.ndarray:
    binary = _as_2d_bool_mask(mask)
    spacing = _validate_spacing_yx(spacing_yx)
    if not np.any(binary):
        return np.full(binary.shape, np.inf, dtype=np.float32)
    return distance_transform_2d(~binary, spacing).astype(np.float32, copy=False)


def clipped_signed_distance_from_mask_2d(
    mask: np.ndarray,
    spacing_yx: tuple[float, float],
    *,
    clip_nm: float,
) -> np.ndarray:
    clip = _validate_positive_nm("clip_nm", clip_nm)
    sdf = signed_distance_from_mask_2d(mask, spacing_yx)
    return np.clip(sdf, -clip, clip).astype(np.float32, copy=False)


def tsdf_from_sdf_nm(sdf_nm: np.ndarray, *, clip_nm: float) -> np.ndarray:
    clip = _validate_positive_nm("clip_nm", clip_nm)
    return np.clip(
        np.asarray(sdf_nm, dtype=np.float32) / np.float32(clip),
        -1.0,
        1.0,
    ).astype(np.float32, copy=False)
