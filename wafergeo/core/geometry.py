from __future__ import annotations

import numpy as np


def vtk_polys_to_triangles(polys, vtk_to_numpy) -> np.ndarray:
    connectivity = polys.GetConnectivityArray()
    offsets = polys.GetOffsetsArray()
    if connectivity is None or offsets is None:
        return np.zeros((0, 3), dtype=np.int32)
    conn = vtk_to_numpy(connectivity).astype(np.int64, copy=False)
    offs = vtk_to_numpy(offsets).astype(np.int64, copy=False)
    if offs.size <= 1:
        return np.zeros((0, 3), dtype=np.int32)
    faces: list[tuple[int, int, int]] = []
    for start, end in zip(offs[:-1], offs[1:], strict=True):
        ids = conn[int(start) : int(end)]
        if ids.size < 3:
            continue
        base = int(ids[0])
        for i in range(1, int(ids.size) - 1):
            faces.append((base, int(ids[i]), int(ids[i + 1])))
    if not faces:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(faces, dtype=np.int32)


def nearest_neighbor_distances_numpy(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    if src_xyz.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if dst_xyz.shape[0] == 0:
        return np.full((src_xyz.shape[0],), np.inf, dtype=np.float64)

    src = np.asarray(src_xyz, dtype=np.float64)
    dst = np.asarray(dst_xyz, dtype=np.float64)
    out = np.empty((src.shape[0],), dtype=np.float64)
    for start in range(0, src.shape[0], chunk_size):
        stop = min(start + chunk_size, src.shape[0])
        chunk = src[start:stop]
        diff = chunk[:, None, :] - dst[None, :, :]
        dist_sq = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
        out[start:stop] = np.sqrt(np.min(dist_sq, axis=1))
    return out
