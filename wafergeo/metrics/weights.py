from __future__ import annotations

import numpy as np

from wafergeo.core.types import Obs2D


def _boundary_emphasis(obs: Obs2D, gamma: float) -> np.ndarray:
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    return np.exp(-gamma * np.abs(np.asarray(obs.tsdf, dtype=np.float32)))


def _coerce_float(value: object, default: float) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    return float(default)


def build_weight_map(
    obs: Obs2D,
    entry_params: dict[str, object],
    band_mask: np.ndarray,
) -> np.ndarray:
    params = dict(entry_params)
    mode = str(params.get("weight_mode", "uniform")).lower()

    if mode == "uniform":
        weight = np.ones(obs.mask.shape, dtype=np.float32)
    elif mode == "sem_weight":
        if obs.weight is None:
            weight = np.ones(obs.mask.shape, dtype=np.float32)
        else:
            weight = np.asarray(obs.weight, dtype=np.float32).copy()
            weight[~np.isfinite(weight)] = 0.0
            weight = np.clip(weight, 0.0, None)
    elif mode == "boundary_emphasis":
        gamma = _coerce_float(params.get("gamma", 2.0), 2.0)
        weight = _boundary_emphasis(obs, gamma)
    else:
        raise ValueError(f"unsupported weight_mode: {mode}")

    spatial = params.get("spatial_weight")
    if isinstance(spatial, dict):
        spatial_kind = str(spatial.get("type", "")).lower()
        if spatial_kind == "boundary_emphasis":
            gamma = _coerce_float(spatial.get("gamma", 2.0), 2.0)
            weight *= _boundary_emphasis(obs, gamma)

    mask = np.asarray(band_mask, dtype=bool)
    if mask.shape != obs.mask.shape:
        raise ValueError("band_mask shape must match obs mask shape")
    weight = np.where(mask, weight, 0.0)

    weight = np.asarray(weight, dtype=np.float32)
    weight[~np.isfinite(weight)] = 0.0
    weight = np.clip(weight, 0.0, None)
    return weight
