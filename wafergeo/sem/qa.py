from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wafergeo.core.types import Obs2D, Status
from wafergeo.sem.normalize import NormalizedContourLoop
from wafergeo.sem.spec import SEMPrepareSpecV1


@dataclass(frozen=True)
class SEMQA:
    status: Status
    notes: list[str] = field(default_factory=list)
    loop_count: int = 0
    open_contour_count: int = 0
    mask_fraction: float = 0.0
    tsdf_min: float = 0.0
    tsdf_max: float = 0.0
    nan_count: int = 0
    inf_count: int = 0


def _combine_status(a: Status, b: Status) -> Status:
    rank: dict[Status, int] = {"OK": 0, "WARN": 1, "FAIL": 2}
    return a if rank[a] >= rank[b] else b


def compute_sem_qa(
    obs: Obs2D,
    loops: list[NormalizedContourLoop],
    spec: SEMPrepareSpecV1,
    *,
    distance_type: str,
    notes: list[str] | None = None,
) -> SEMQA:
    mask = np.asarray(obs.mask, dtype=np.uint8)
    tsdf = np.asarray(obs.tsdf, dtype=np.float32)
    qa_notes = list(notes or [])
    status: Status = "OK"

    open_count = int(sum(not loop.is_closed for loop in loops))
    mask_fraction = float(np.mean(mask > 0)) if mask.size > 0 else 0.0
    nan_count = int(np.isnan(tsdf).sum())
    inf_count = int(np.isinf(tsdf).sum())
    tsdf_min = float(np.nanmin(tsdf)) if tsdf.size else 0.0
    tsdf_max = float(np.nanmax(tsdf)) if tsdf.size else 0.0

    if open_count > spec.qa.max_open_contours:
        status = _combine_status(status, "WARN")
        qa_notes.append(
            f"open_contours={open_count} exceeds max_open_contours={spec.qa.max_open_contours}"
        )
    if mask_fraction < spec.qa.min_mask_fraction:
        status = _combine_status(status, "WARN")
        qa_notes.append(
            "mask_fraction="
            f"{mask_fraction:.6f} below min_mask_fraction={spec.qa.min_mask_fraction:.6f}"
        )
    if nan_count > 0 or inf_count > 0:
        status = _combine_status(status, "FAIL")
        qa_notes.append(f"tsdf has nan_count={nan_count}, inf_count={inf_count}")

    if distance_type == "unsigned_curve":
        if tsdf_min < -1e-4 or tsdf_max > 1.0001:
            status = _combine_status(status, "FAIL")
            qa_notes.append("unsigned_curve tsdf must be in [0,1]")
    else:
        if tsdf_min < -1.0001 or tsdf_max > 1.0001:
            status = _combine_status(status, "FAIL")
            qa_notes.append("signed_region tsdf must be in [-1,1]")

    return SEMQA(
        status=status,
        notes=qa_notes,
        loop_count=len(loops),
        open_contour_count=open_count,
        mask_fraction=mask_fraction,
        tsdf_min=tsdf_min,
        tsdf_max=tsdf_max,
        nan_count=nan_count,
        inf_count=inf_count,
    )
