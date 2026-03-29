from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wafergeo.core.types import Obs2D
from wafergeo.metrics.spec import MeasurementSpecV1


@dataclass(frozen=True)
class LineScanCacheEntry:
    axis: str
    fixed_index: int
    scan_indices: np.ndarray
    scan_coords_nm: np.ndarray


@dataclass(frozen=True)
class Obs2DPrecomp:
    band_mask_obs: np.ndarray
    contour_points_xy: np.ndarray
    contour_kdtree: Any | None
    line_scan_cache: dict[str, LineScanCacheEntry] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)


def band_mask_from_obs(obs: Obs2D) -> np.ndarray:
    return (np.abs(np.asarray(obs.tsdf, dtype=np.float32)) < 1.0).astype(bool, copy=False)


def contour_points_from_loops(obs: Obs2D, *, use_holes: bool = False) -> np.ndarray:
    points: list[np.ndarray] = []
    for loop in obs.loops:
        if loop.is_hole and not use_holes:
            continue
        pts = np.asarray(loop.points_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
            continue
        points.append(pts)

    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(points, axis=0).astype(np.float32, copy=False)


def build_contour_kdtree(points_xy: np.ndarray) -> Any:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for contour KDTree. "
            "Install: pip install 'wafergeo[scipy]'"
        ) from exc

    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("points_xy must be non-empty shape (N,2)")
    return cKDTree(points)


def _to_index(coord_nm: float, origin_nm: float, spacing_nm: float, max_size: int) -> int:
    idx = int(np.rint((coord_nm - origin_nm) / spacing_nm))
    return int(np.clip(idx, 0, max_size - 1))


def build_line_scan_cache(
    obs: Obs2D,
    measurement: MeasurementSpecV1,
) -> dict[str, LineScanCacheEntry]:
    y_size, x_size = obs.mask.shape
    sy, sx = float(obs.grid2d.spacing[0]), float(obs.grid2d.spacing[1])
    oy, ox = float(obs.grid2d.origin[0]), float(obs.grid2d.origin[1])

    cache: dict[str, LineScanCacheEntry] = {}
    for line in measurement.lines:
        if line.axis == "x":
            y_idx = _to_index(line.coord_nm, oy, sy, y_size)
            x0 = _to_index(line.range_nm[0], ox, sx, x_size)
            x1 = _to_index(line.range_nm[1], ox, sx, x_size)
            x_start, x_stop = min(x0, x1), max(x0, x1)
            scan_indices = np.arange(x_start, x_stop + 1, dtype=np.int32)
            scan_coords = (ox + scan_indices.astype(np.float32) * sx).astype(np.float32)
            entry = LineScanCacheEntry(
                axis="x",
                fixed_index=y_idx,
                scan_indices=scan_indices,
                scan_coords_nm=scan_coords,
            )
        else:
            x_idx = _to_index(line.coord_nm, ox, sx, x_size)
            y0 = _to_index(line.range_nm[0], oy, sy, y_size)
            y1 = _to_index(line.range_nm[1], oy, sy, y_size)
            y_start, y_stop = min(y0, y1), max(y0, y1)
            scan_indices = np.arange(y_start, y_stop + 1, dtype=np.int32)
            scan_coords = (oy + scan_indices.astype(np.float32) * sy).astype(np.float32)
            entry = LineScanCacheEntry(
                axis="y",
                fixed_index=x_idx,
                scan_indices=scan_indices,
                scan_coords_nm=scan_coords,
            )

        if entry.scan_indices.size < 2:
            raise ValueError(f"line '{line.id}' has <2 samples after grid projection")
        cache[line.id] = entry

    return cache
