from __future__ import annotations

import numpy as np

from wafergeo.sdf.errors import InvalidMuError, ShapeMismatchError


def pair_codebook(num_channels: int) -> dict[tuple[int, int], int]:
    if num_channels < 2:
        return {}
    code = 0
    codebook: dict[tuple[int, int], int] = {}
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            codebook[(i, j)] = code
            code += 1
    return codebook


def encode_pair_indices(
    i: np.ndarray,
    j: np.ndarray,
    codebook: dict[tuple[int, int], int],
    outside_code: int = 255,
) -> np.ndarray:
    if i.shape != j.shape:
        raise ShapeMismatchError(f"i and j must share shape, got {i.shape} vs {j.shape}")

    pair_code = np.full(i.shape, int(outside_code), dtype=np.uint8)
    for (a, b), code in codebook.items():
        mask = ((i == a) & (j == b)) | ((i == b) & (j == a))
        pair_code[mask] = np.uint8(code)
    return pair_code


def _top2_abs_indices(abs_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = abs_stack.shape[0]
    best_abs = np.full(abs_stack.shape[1:], np.inf, dtype=np.float32)
    best_idx = np.full(abs_stack.shape[1:], -1, dtype=np.int16)
    second_idx = np.full(abs_stack.shape[1:], -1, dtype=np.int16)
    second_abs = np.full(abs_stack.shape[1:], np.inf, dtype=np.float32)

    for channel in range(m):
        current = np.asarray(abs_stack[channel], dtype=np.float32)
        better = (current < best_abs) | ((current == best_abs) & (channel < best_idx))

        second_abs[better] = best_abs[better]
        second_idx[better] = best_idx[better]
        best_abs[better] = current[better]
        best_idx[better] = channel

        remain = ~better
        better_second = remain & (
            (current < second_abs) | ((current == second_abs) & (channel < second_idx))
        )
        second_abs[better_second] = current[better_second]
        second_idx[better_second] = channel

    return best_abs, best_idx, second_idx


def compute_boundary_features_from_phi(
    phi_stack_nm: np.ndarray,
    mu_nm: float,
    outside_code: int,
    band_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(mu_nm) or mu_nm <= 0.0:
        raise InvalidMuError(f"mu_nm must be finite and > 0, got {mu_nm}")

    phi = np.asarray(phi_stack_nm, dtype=np.float32)
    if phi.ndim != 4:
        raise ShapeMismatchError(f"phi_stack_nm must be shape (M,Z,Y,X), got ndim={phi.ndim}")

    abs_stack = np.abs(phi)
    best_abs, best_idx, second_idx = _top2_abs_indices(abs_stack)
    d_boundary = np.clip(best_abs, 0.0, float(mu_nm)) / float(mu_nm)

    codebook = pair_codebook(phi.shape[0])
    pair_code = encode_pair_indices(best_idx, second_idx, codebook, outside_code=outside_code)
    if band_only:
        pair_code[best_abs >= float(mu_nm)] = np.uint8(outside_code)

    return d_boundary.astype(np.float32, copy=False), pair_code
