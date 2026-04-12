from __future__ import annotations

import numpy as np


def _require_scipy_distance_transform_edt():
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "scipy is required for full-material SDF. "
            "Install with: pip install 'wafergeo[scipy]'"
        ) from exc
    return distance_transform_edt


def build_full_material_sdf(
    label_zyx: np.ndarray,
    material_ids: list[int],
    spacing_zyx: tuple[float, float, float],
    mu_nm: float,
) -> np.ndarray:
    """Build a TSDF stack for all requested materials.

    Output shape is ``(M, Z, Y, X)`` with values in ``[-1, 1]``.
    """

    if label_zyx.ndim != 3:
        raise ValueError(f"label_zyx must be 3D, got ndim={label_zyx.ndim}")
    if not material_ids:
        raise ValueError("material_ids must be non-empty")
    if mu_nm <= 0.0 or not np.isfinite(mu_nm):
        raise ValueError(f"mu_nm must be finite and >0, got {mu_nm}")

    for i, value in enumerate(spacing_zyx):
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError(f"spacing_zyx[{i}] must be finite and >0, got {value}")

    edt = _require_scipy_distance_transform_edt()

    zyx = label_zyx.shape
    out = np.empty((len(material_ids),) + zyx, dtype=np.float32)
    sampling = tuple(float(v) for v in spacing_zyx)

    for channel, material_id in enumerate(material_ids):
        mask = label_zyx == int(material_id)
        if np.all(mask):
            out[channel].fill(-1.0)
            continue
        if not np.any(mask):
            out[channel].fill(1.0)
            continue

        # Signed distance in nm: inside negative, outside positive.
        outside_nm = edt(~mask, sampling=sampling).astype(np.float32, copy=False)
        inside_nm = edt(mask, sampling=sampling).astype(np.float32, copy=False)
        phi_nm = outside_nm - inside_nm
        out[channel] = np.clip(phi_nm, -mu_nm, mu_nm) / float(mu_nm)

    return out.astype(np.float32, copy=False)
