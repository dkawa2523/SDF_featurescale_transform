from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wafergeo.core.types import Obs2D
from wafergeo.observe.spec import ObserverSpecV2


@dataclass(frozen=True)
class ObserverQA:
    tsdf_min: float
    tsdf_max: float
    nan_count: int
    inf_count: int
    mask_area_px: int
    mask_fraction: float
    contour_count: int
    open_contour_count: int
    status: str
    notes: list[str] = field(default_factory=list)


def _is_loop_closed(points_xy: np.ndarray, atol: float = 1e-5) -> bool:
    if points_xy.shape[0] < 2:
        return False
    return bool(np.allclose(points_xy[0], points_xy[-1], rtol=0.0, atol=atol))


def run_observer_qa(obs: Obs2D, spec: ObserverSpecV2) -> ObserverQA:
    field = np.asarray(obs.tsdf, dtype=np.float32)
    nan_count = int(np.isnan(field).sum())
    inf_count = int(np.isinf(field).sum())

    contour_count = len(obs.loops)
    open_count = int(sum(0 if _is_loop_closed(loop.points_xy) else 1 for loop in obs.loops))

    total_px = int(obs.mask.size)
    mask_area_px = int(np.sum(np.asarray(obs.mask, dtype=bool)))
    mask_fraction = float(mask_area_px) / float(total_px) if total_px > 0 else 0.0

    notes: list[str] = []
    status = "OK"

    max_open_raw = spec.qa.get("max_open_contours", 0)
    max_open = int(max_open_raw) if isinstance(max_open_raw, (int, float, str)) else 0
    if open_count > max_open:
        status = "WARN"
        notes.append(f"open_contours={open_count} > max_open_contours={max_open}")

    min_area_raw = spec.qa.get("min_area_nm2")
    min_area_nm2: float | None = None
    if isinstance(min_area_raw, (int, float, str)):
        min_area_nm2 = float(min_area_raw)
    if min_area_nm2 is not None:
        pixel_area = float(obs.grid2d.spacing[0]) * float(obs.grid2d.spacing[1])
        area_nm2 = float(mask_area_px) * pixel_area
        if area_nm2 < min_area_nm2:
            status = "WARN"
            notes.append(f"mask_area_nm2={area_nm2:.3f} < min_area_nm2={min_area_nm2:.3f}")

    if nan_count > 0 or inf_count > 0:
        status = "FAIL"
        notes.append("tsdf contains NaN/Inf")

    return ObserverQA(
        tsdf_min=float(np.min(field)) if field.size > 0 else 0.0,
        tsdf_max=float(np.max(field)) if field.size > 0 else 0.0,
        nan_count=nan_count,
        inf_count=inf_count,
        mask_area_px=mask_area_px,
        mask_fraction=mask_fraction,
        contour_count=contour_count,
        open_contour_count=open_count,
        status=status,
        notes=notes,
    )
