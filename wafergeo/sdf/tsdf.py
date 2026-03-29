from __future__ import annotations

import numpy as np

from wafergeo.core.types import MaterialSpec
from wafergeo.sdf.errors import InvalidMuError, ShapeMismatchError


def to_tsdf(
    phi_nm: np.ndarray,
    mu_nm: float,
    out_dtype: np.dtype | type[np.floating] | str = np.float16,
) -> np.ndarray:
    if not np.isfinite(mu_nm) or mu_nm <= 0.0:
        raise InvalidMuError(f"mu_nm must be finite and > 0, got {mu_nm}")
    tsdf = np.clip(np.asarray(phi_nm, dtype=np.float32), -mu_nm, mu_nm) / float(mu_nm)
    return tsdf.astype(out_dtype, copy=False)


def from_tsdf(tsdf: np.ndarray, mu_nm: float) -> np.ndarray:
    if not np.isfinite(mu_nm) or mu_nm <= 0.0:
        raise InvalidMuError(f"mu_nm must be finite and > 0, got {mu_nm}")
    return np.asarray(tsdf, dtype=np.float32) * float(mu_nm)


def label_from_tsdf(
    tsdf_stack: np.ndarray,
    material: MaterialSpec,
    *,
    void_index: int = 0,
    tie_break: str = "priority",
    selected_material_ids: list[int] | None = None,
) -> np.ndarray:
    """Convert TSDF stack to deterministic material-id labels.

    Rules:
    - choose channel with minimum TSDF (most inside)
    - if all channels are positive, return void material id
    - ties are broken by MaterialSpec.priority
    """

    tsdf = np.asarray(tsdf_stack)
    if tsdf.ndim != 4:
        raise ShapeMismatchError(f"tsdf_stack must be shape (M,Z,Y,X), got ndim={tsdf.ndim}")

    channel_ids = selected_material_ids or list(material.ids)
    m = tsdf.shape[0]
    if len(channel_ids) != m:
        raise ShapeMismatchError(
            f"selected_material_ids length ({len(channel_ids)}) must match channel count ({m})"
        )
    if tie_break != "priority":
        raise ValueError(f"unsupported tie_break={tie_break}")

    id_to_priority = {
        material_id: priority
        for material_id, priority in zip(material.ids, material.priority, strict=True)
    }
    priorities = np.asarray([id_to_priority[mid] for mid in channel_ids], dtype=np.int32)

    best_value = np.asarray(tsdf[0], dtype=np.float32)
    best_index = np.zeros(tsdf.shape[1:], dtype=np.int16)
    best_priority = np.full(tsdf.shape[1:], priorities[0], dtype=np.int32)

    for channel in range(1, m):
        value = np.asarray(tsdf[channel], dtype=np.float32)
        prio = int(priorities[channel])
        better = value < best_value
        tie = np.isclose(value, best_value, rtol=0.0, atol=1e-7)
        better_tie = tie & (
            (prio > best_priority) | ((prio == best_priority) & (channel < best_index))
        )
        choose = better | better_tie
        best_value[choose] = value[choose]
        best_index[choose] = channel
        best_priority[choose] = prio

    channel_ids_arr = np.asarray(channel_ids, dtype=np.int32)
    labels = channel_ids_arr[best_index].astype(np.int32, copy=False)

    if void_index < 0 or void_index >= len(material.ids):
        raise ValueError(f"void_index out of range: {void_index}")
    outside_void_id = int(material.ids[void_index])
    outside = np.all(tsdf > 0.0, axis=0)
    labels[outside] = outside_void_id

    max_id = int(max(material.ids))
    dtype = np.uint8 if max_id <= 255 else np.uint16
    return labels.astype(dtype, copy=False)
