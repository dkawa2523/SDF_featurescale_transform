from __future__ import annotations

import numpy as np

from wafergeo.core.grid import GridSpec
from wafergeo.core.types import ContourLoop
from wafergeo.observe.errors import ObserverOptionalDependencyError


def _signed_area_xy(points_xy: np.ndarray) -> float:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _resample_closed_loop(points_xy: np.ndarray, n_points: int) -> np.ndarray:
    if n_points <= 0:
        return points_xy.astype(np.float32, copy=False)

    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] < 2:
        return pts.astype(np.float32, copy=False)

    if not np.allclose(pts[0], pts[-1], rtol=0.0, atol=1e-8):
        pts = np.vstack([pts, pts[0]])

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cumulative[-1])
    if total <= 0.0:
        return np.repeat(pts[:1], repeats=max(2, n_points), axis=0).astype(np.float32)

    sample_t = np.linspace(0.0, total, num=max(2, n_points), endpoint=True)
    out = np.empty((sample_t.size, 2), dtype=np.float64)

    idx = 0
    for i, target in enumerate(sample_t):
        while idx < seg.size - 1 and cumulative[idx + 1] < target:
            idx += 1
        left = cumulative[idx]
        right = cumulative[idx + 1]
        if right <= left:
            out[i] = pts[idx]
            continue
        alpha = (target - left) / (right - left)
        out[i] = (1.0 - alpha) * pts[idx] + alpha * pts[idx + 1]

    return out.astype(np.float32, copy=False)


def extract_contours_from_tsdf(
    tsdf2d: np.ndarray,
    grid2d: GridSpec,
    level: float,
    resample_points: int,
    *,
    backend: str = "skimage",
) -> list[ContourLoop]:
    if grid2d.dim != 2:
        raise ValueError("grid2d.dim must be 2")
    if backend != "skimage":
        raise ValueError(f"unsupported contour backend: {backend}")

    try:
        from skimage.measure import find_contours
    except ImportError as exc:
        raise ObserverOptionalDependencyError(
            "contour backend 'skimage' requires scikit-image. "
            "Install with: pip install 'wafergeo[observe]'"
        ) from exc

    field = np.asarray(tsdf2d, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError(f"tsdf2d must be 2D, got ndim={field.ndim}")

    sy, sx = (float(grid2d.spacing[0]), float(grid2d.spacing[1]))
    oy, ox = (float(grid2d.origin[0]), float(grid2d.origin[1]))

    loops: list[ContourLoop] = []
    contours_rc = find_contours(field, level=float(level))
    for contour_rc in contours_rc:
        if contour_rc.shape[0] < 2:
            continue

        y_nm = oy + contour_rc[:, 0] * sy
        x_nm = ox + contour_rc[:, 1] * sx
        points_xy = np.stack([x_nm, y_nm], axis=1).astype(np.float32, copy=False)
        if resample_points > 0:
            points_xy = _resample_closed_loop(points_xy, resample_points)

        if points_xy.shape[0] < 2:
            continue

        if not np.allclose(points_xy[0], points_xy[-1], rtol=0.0, atol=1e-5):
            points_xy = np.vstack([points_xy, points_xy[0]]).astype(np.float32, copy=False)

        is_hole = _signed_area_xy(points_xy) < 0.0
        loops.append(
            ContourLoop(
                points_xy=points_xy,
                is_hole=bool(is_hole),
                label=None,
                meta={"level": float(level)},
            )
        )

    return loops
