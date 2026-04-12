from __future__ import annotations

import numpy as np

from wafergeo.compare.features import ViewFeature
from wafergeo.compare.metric_types import MetricComputation, MetricContext


def _resample_polyline(points: np.ndarray, *, closed: bool, max_step_nm: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 2:
        return pts
    start_points = pts if closed else pts[:-1]
    stop_points = np.vstack([pts[1:], pts[:1]]) if closed else pts[1:]
    segments: list[np.ndarray] = []
    for start, stop in zip(start_points, stop_points, strict=True):
        length = float(np.linalg.norm(stop - start))
        steps = max(int(np.ceil(length / max(max_step_nm, 1e-6))), 1)
        t = np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float32)
        segments.append(start[None, :] + t[:, None] * (stop - start)[None, :])
    return np.vstack(segments).astype(np.float32, copy=False)


def _all_points(feature: ViewFeature) -> np.ndarray:
    max_step_nm = float(min(feature.grid2d.spacing))
    points = []
    for contour in feature.contours.contours:
        if contour.meta.get("geometry") == "point_cloud":
            points.append(np.asarray(contour.points_xy_nm, dtype=np.float32))
        else:
            points.append(
                _resample_polyline(
                    contour.points_xy_nm,
                    closed=contour.closed,
                    max_step_nm=max_step_nm,
                )
            )
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(points, axis=0).astype(np.float32, copy=False)


def _nearest_distances(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if lhs.size == 0 or rhs.size == 0:
        return np.full((lhs.shape[0],), np.inf, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        diff = lhs[:, None, :] - rhs[None, :, :]
        return np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1)).astype(np.float32)

    tree = cKDTree(rhs)
    dist, _ = tree.query(lhs, k=1)
    return np.asarray(dist, dtype=np.float32)


def compute_chamfer(
    sim: ViewFeature,
    target: ViewFeature,
    _context: MetricContext,
) -> MetricComputation:
    sim_points = _all_points(sim)
    target_points = _all_points(target)
    sim_to_target = _nearest_distances(sim_points, target_points)
    target_to_sim = _nearest_distances(target_points, sim_points)
    value = float(0.5 * (np.mean(sim_to_target) + np.mean(target_to_sim)))
    return MetricComputation(name="chamfer", loss=value, value=value)
