from __future__ import annotations

from collections.abc import Callable

import numpy as np

from wafergeo.sdf.engines.registry import (
    get_sdf_engine,
    list_sdf_engines,
    register_default_sdf_engines,
)
from wafergeo.sdf.errors import EDTBackendUnavailableError, ShapeMismatchError

DistanceFn = Callable[[np.ndarray, tuple[float, ...]], np.ndarray]


def register_default_backends() -> None:
    """Backward-compatible name for default engine registration."""

    register_default_sdf_engines()


def get_edt_backend(name: str) -> DistanceFn:
    register_default_backends()
    try:
        engine = get_sdf_engine(name)
    except KeyError as exc:
        known = ", ".join(list_sdf_engines())
        raise EDTBackendUnavailableError(
            f"Unknown EDT backend '{name}'. Available: [{known}]"
        ) from exc
    return engine.distance


def _signed_distance_from_distance_fn(
    mask: np.ndarray,
    sampling_zyx: tuple[float, ...],
    distance_fn: DistanceFn,
) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 3:
        raise ShapeMismatchError(f"mask must be shape (Z,Y,X), got ndim={mask_bool.ndim}")
    if len(sampling_zyx) != 3:
        raise ShapeMismatchError(
            f"sampling_zyx must be length 3 for (Z,Y,X), got {len(sampling_zyx)}"
        )

    if mask_bool.shape[0] == 1:
        mask2d = mask_bool[0]
        sampling2d = (sampling_zyx[1], sampling_zyx[2])
        dist_in = np.asarray(distance_fn(mask2d, sampling2d), dtype=np.float32)
        dist_out = np.asarray(distance_fn(~mask2d, sampling2d), dtype=np.float32)
        phi2d = dist_out - dist_in
        return phi2d[np.newaxis, ...].astype(np.float32, copy=False)

    dist_in = np.asarray(distance_fn(mask_bool, sampling_zyx), dtype=np.float32)
    dist_out = np.asarray(distance_fn(~mask_bool, sampling_zyx), dtype=np.float32)
    return (dist_out - dist_in).astype(np.float32, copy=False)


def edt_distance(mask: np.ndarray, sampling_zyx: tuple[float, ...], backend: str) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim not in (2, 3):
        raise ShapeMismatchError(f"mask must be 2D or 3D for EDT, got ndim={mask_bool.ndim}")
    if len(sampling_zyx) != mask_bool.ndim:
        raise ShapeMismatchError(
            f"sampling_zyx length must match ndim: len={len(sampling_zyx)} ndim={mask_bool.ndim}"
        )
    distance_fn = get_edt_backend(backend)
    return np.asarray(distance_fn(mask_bool, sampling_zyx), dtype=np.float32)


def signed_distance_from_mask(
    mask: np.ndarray,
    sampling_zyx: tuple[float, ...],
    backend: str,
) -> np.ndarray:
    distance_fn = get_edt_backend(backend)
    return _signed_distance_from_distance_fn(mask, sampling_zyx, distance_fn)


def signed_distance_from_mask_with_distance_fn(
    mask: np.ndarray,
    sampling_zyx: tuple[float, ...],
    distance_fn: DistanceFn,
) -> np.ndarray:
    return _signed_distance_from_distance_fn(mask, sampling_zyx, distance_fn)
