from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from tests.sdf.helpers import register_bruteforce_engine
from wafergeo.core.grid import GridSpec
from wafergeo.core.meta import Meta
from wafergeo.core.types import ContourLoop, Obs2D
from wafergeo.observe.tsdf2d import tsdf2d_from_mask


def build_grid2d() -> GridSpec:
    return GridSpec(
        dim=2,
        spacing=(10.0, 10.0),
        origin=(0.0, 0.0),
        axis_order="YX",
        sample_location="cell_center",
        units="nm",
    )


def rect_mask(
    size: int = 32,
    *,
    x0: int = 10,
    x1: int = 22,
    y0: int = 10,
    y1: int = 22,
) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def _bbox_loop(mask: np.ndarray, grid: GridSpec) -> list[ContourLoop]:
    ys, xs = np.nonzero(mask.astype(bool))
    if ys.size == 0:
        return []

    y0, y1 = int(np.min(ys)), int(np.max(ys))
    x0, x1 = int(np.min(xs)), int(np.max(xs))

    sy, sx = float(grid.spacing[0]), float(grid.spacing[1])
    oy, ox = float(grid.origin[0]), float(grid.origin[1])

    pts = np.array(
        [
            [ox + x0 * sx, oy + y0 * sy],
            [ox + x1 * sx, oy + y0 * sy],
            [ox + x1 * sx, oy + y1 * sy],
            [ox + x0 * sx, oy + y1 * sy],
            [ox + x0 * sx, oy + y0 * sy],
        ],
        dtype=np.float32,
    )
    return [ContourLoop(points_xy=pts, is_hole=False)]


def build_obs2d_from_mask(
    mask: np.ndarray,
    *,
    backend_name: str,
    with_weight: bool = False,
    profile_id: str = "metric_test",
) -> Obs2D:
    grid = build_grid2d()
    tsdf = tsdf2d_from_mask(mask, grid, mu_nm=30.0, backend=backend_name)
    loops = _bbox_loop(mask, grid)

    weight = None
    if with_weight:
        weight = np.ones(mask.shape, dtype=np.float32)
        weight[:, : mask.shape[1] // 2] = 0.25

    meta = Meta(
        schema_version="observer/v2",
        profile_id=profile_id,
        config_hash="cfg",
        generator_version="0.1.0",
        git_commit="deadbeef",
        input_hash="input",
        created_at=datetime.now(UTC).isoformat(),
        extra={"observer_kind": "synthetic"},
    )
    return Obs2D(
        grid2d=grid,
        mask=np.asarray(mask, dtype=np.uint8),
        tsdf=np.asarray(tsdf, dtype=np.float32),
        loops=loops,
        weight=weight,
        transform=None,
        debug_maps={},
        meta=meta,
    )


def build_obs_pair_shifted(
    *,
    shift_x: int = 1,
    with_weight: bool = False,
) -> tuple[Obs2D, Obs2D, str]:
    backend = register_bruteforce_engine(f"metric_brute_{shift_x}_{int(with_weight)}")
    base = rect_mask()
    shifted = np.roll(base, shift=shift_x, axis=1)
    obs = build_obs2d_from_mask(base, backend_name=backend, with_weight=with_weight)
    pred = build_obs2d_from_mask(shifted, backend_name=backend, with_weight=with_weight)
    return pred, obs, backend
