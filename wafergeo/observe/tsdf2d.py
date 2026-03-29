from __future__ import annotations

import numpy as np

from wafergeo.core.grid import GridSpec
from wafergeo.sdf.edt import edt_distance


def tsdf2d_from_mask(
    mask: np.ndarray,
    grid2d: GridSpec,
    mu_nm: float,
    backend: str = "scipy",
) -> np.ndarray:
    if grid2d.dim != 2:
        raise ValueError("grid2d.dim must be 2")
    if mu_nm <= 0.0 or not np.isfinite(mu_nm):
        raise ValueError(f"mu_nm must be finite and > 0, got {mu_nm}")

    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2:
        raise ValueError(f"mask must be 2D, got ndim={mask_bool.ndim}")

    if np.all(mask_bool):
        return np.full(mask_bool.shape, -1.0, dtype=np.float32)
    if not np.any(mask_bool):
        return np.full(mask_bool.shape, +1.0, dtype=np.float32)

    sampling_yx = (float(grid2d.spacing[0]), float(grid2d.spacing[1]))
    dist_in = edt_distance(mask_bool, sampling_yx, backend)
    dist_out = edt_distance(~mask_bool, sampling_yx, backend)
    phi = np.asarray(dist_out - dist_in, dtype=np.float32)

    tsdf = np.clip(phi, -float(mu_nm), float(mu_nm)) / float(mu_nm)
    return tsdf.astype(np.float32, copy=False)
