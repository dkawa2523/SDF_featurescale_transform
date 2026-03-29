from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from wafergeo.sem.readers import RawContourSet
from wafergeo.sem.spec import SEMPrepareSpecV1


@dataclass(frozen=True)
class TransformChain:
    T_px_to_sem_nm: np.ndarray
    T_sem_nm_to_sim_nm: np.ndarray
    T_px_to_sim_nm: np.ndarray
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, mat in {
            "T_px_to_sem_nm": self.T_px_to_sem_nm,
            "T_sem_nm_to_sim_nm": self.T_sem_nm_to_sim_nm,
            "T_px_to_sim_nm": self.T_px_to_sim_nm,
        }.items():
            arr = np.asarray(mat, dtype=np.float64)
            if arr.shape != (3, 3):
                raise ValueError(f"{name} must be shape (3,3)")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class NormalizedContourLoop:
    points_nm: np.ndarray
    points_sim_nm: np.ndarray
    is_closed: bool
    is_hole: bool
    role: str
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.points_nm.ndim != 2 or self.points_nm.shape[1] != 2:
            raise ValueError("points_nm must be shape (N,2)")
        if self.points_sim_nm.shape != self.points_nm.shape:
            raise ValueError("points_sim_nm shape must match points_nm")
        if self.points_nm.shape[0] < 2:
            raise ValueError("points must have at least 2 vertices")


def _identity33() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _unit_scale_to_nm(units: str, pixel_size_nm: float | None) -> float:
    if units == "nm":
        return 1.0
    if units == "um":
        return 1_000.0
    if units == "px":
        if pixel_size_nm is None or not isfinite(pixel_size_nm) or pixel_size_nm <= 0.0:
            raise ValueError("pixel_size_nm must be provided for px units")
        return float(pixel_size_nm)
    raise ValueError(f"unsupported units: {units}")


def build_transform_chain(
    spec: SEMPrepareSpecV1,
    image_shape: tuple[int, int] | None,
    *,
    sem_to_sim_override: np.ndarray | None = None,
) -> TransformChain:
    pixel_size_nm = spec.input.pixel_size_nm
    scale_nm = _unit_scale_to_nm(spec.input.units, pixel_size_nm)
    T_px_to_sem = _identity33()

    if spec.input.coord_system == "pixel":
        if pixel_size_nm is None or pixel_size_nm <= 0.0:
            raise ValueError("pixel_size_nm is required when input.coord_system='pixel'")
        px = float(pixel_size_nm)
        if spec.input.pixel_y_policy == "flip_y":
            if image_shape is None:
                raise ValueError("image_shape is required for pixel_y_policy='flip_y'")
            height = int(image_shape[0])
            T_px_to_sem = np.array(
                [[px, 0.0, 0.0], [0.0, -px, (height - 1) * px], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        else:
            T_px_to_sem = np.array(
                [[px, 0.0, 0.0], [0.0, px, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
    else:
        T_px_to_sem = np.array(
            [[scale_nm, 0.0, 0.0], [0.0, scale_nm, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    if sem_to_sim_override is None:
        T_sem_to_sim = _identity33()
    else:
        T_sem_to_sim = np.asarray(sem_to_sim_override, dtype=np.float64)
        if T_sem_to_sim.shape != (3, 3):
            raise ValueError("sem_to_sim_override must be shape (3,3)")

    T_px_to_sim = T_sem_to_sim @ T_px_to_sem
    return TransformChain(
        T_px_to_sem_nm=T_px_to_sem,
        T_sem_nm_to_sim_nm=T_sem_to_sim,
        T_px_to_sim_nm=T_px_to_sim,
        meta={
            "coord_system": spec.input.coord_system,
            "units": spec.input.units,
            "pixel_y_policy": spec.input.pixel_y_policy,
            "pixel_size_nm": spec.input.pixel_size_nm,
        },
    )


def _dedupe_consecutive(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] <= 1:
        return points_xy
    keep = np.ones(points_xy.shape[0], dtype=bool)
    delta = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    keep[1:] = delta > 1e-8
    out = points_xy[keep]
    if out.shape[0] < 2:
        return points_xy[:2].astype(np.float32, copy=False)
    return out


def _signed_area_closed(points_xy: np.ndarray) -> float:
    pts = points_xy
    if not np.allclose(pts[0], pts[-1], rtol=0.0, atol=1e-6):
        pts = np.vstack([pts, pts[0]])
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _resample_polyline(points_xy: np.ndarray, n_points: int, *, closed: bool) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    if closed and not np.allclose(pts[0], pts[-1], rtol=0.0, atol=1e-8):
        pts = np.vstack([pts, pts[0]])

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cumulative[-1])
    if total <= 0.0:
        return np.repeat(pts[:1], repeats=max(2, n_points), axis=0).astype(np.float32)

    sample_t = np.linspace(0.0, total, num=max(2, n_points), endpoint=True)
    out = np.empty((sample_t.size, 2), dtype=np.float64)
    edge = 0
    for idx, target in enumerate(sample_t):
        while edge < seg.size - 1 and cumulative[edge + 1] < target:
            edge += 1
        left = cumulative[edge]
        right = cumulative[edge + 1]
        if right <= left:
            out[idx] = pts[edge]
            continue
        alpha = (target - left) / (right - left)
        out[idx] = (1.0 - alpha) * pts[edge] + alpha * pts[edge + 1]

    if closed and not np.allclose(out[0], out[-1], rtol=0.0, atol=1e-6):
        out[-1] = out[0]
    return out.astype(np.float32, copy=False)


def _apply_transform(points_xy: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    out = (T @ homo.T).T
    return out[:, :2].astype(np.float32, copy=False)


def normalize_contours(
    raw: RawContourSet,
    spec: SEMPrepareSpecV1,
    chain: TransformChain,
) -> list[NormalizedContourLoop]:
    loops: list[NormalizedContourLoop] = []
    for loop in raw.loops_raw:
        pts = _dedupe_consecutive(np.asarray(loop.points_xy, dtype=np.float32))
        if not np.isfinite(pts).all():
            continue

        pts_nm = _apply_transform(pts, chain.T_px_to_sem_nm)
        close_dist = float(np.linalg.norm(pts_nm[0] - pts_nm[-1]))
        is_closed = bool(loop.is_closed_hint) or (close_dist <= spec.normalize.close_tol_nm)

        role = str(loop.role or "outer").strip().lower()
        is_hole = role == "hole"

        if is_closed and not np.allclose(pts_nm[0], pts_nm[-1], rtol=0.0, atol=1e-6):
            pts_nm = np.vstack([pts_nm, pts_nm[0]]).astype(np.float32, copy=False)

        if is_closed and spec.normalize.enforce_orientation:
            area = _signed_area_closed(pts_nm)
            want_positive = not is_hole
            if (area < 0.0 and want_positive) or (area > 0.0 and not want_positive):
                pts_nm = pts_nm[::-1].copy()
                if is_closed and not np.allclose(pts_nm[0], pts_nm[-1], rtol=0.0, atol=1e-6):
                    pts_nm[-1] = pts_nm[0]

        if is_closed:
            pts_nm = _resample_polyline(
                pts_nm,
                spec.normalize.resample_points_closed,
                closed=True,
            )
        else:
            pts_nm = _resample_polyline(
                pts_nm,
                spec.normalize.resample_points_open,
                closed=False,
            )

        pts_sim = _apply_transform(pts_nm, chain.T_sem_nm_to_sim_nm)
        loops.append(
            NormalizedContourLoop(
                points_nm=pts_nm,
                points_sim_nm=pts_sim,
                is_closed=is_closed,
                is_hole=is_hole,
                role=role,
                meta={
                    "loop_id": loop.loop_id,
                    "is_closed_hint": loop.is_closed_hint,
                    "close_dist_nm": f"{close_dist:.6f}",
                },
            )
        )
    return loops
