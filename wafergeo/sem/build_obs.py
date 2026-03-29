from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from wafergeo.core.hashing import hash_config
from wafergeo.core.meta import Meta
from wafergeo.core.types import ContourLoop, Obs2D
from wafergeo.observe.tsdf2d import tsdf2d_from_mask
from wafergeo.sem.normalize import NormalizedContourLoop, TransformChain
from wafergeo.sem.qa import SEMQA, compute_sem_qa
from wafergeo.sem.readers import SEMImageRaw
from wafergeo.sem.spec import SEMPrepareSpecV1, sem_prepare_spec_hash


def _grid_shape_from_contours(
    contours: list[NormalizedContourLoop],
    spec: SEMPrepareSpecV1,
    image_raw: SEMImageRaw | None,
) -> tuple[int, int]:
    if spec.target_shape_yx is not None:
        return spec.target_shape_yx

    if image_raw is not None:
        return (int(image_raw.image.shape[0]), int(image_raw.image.shape[1]))

    if not contours:
        return (2, 2)

    points = np.concatenate([loop.points_sim_nm for loop in contours], axis=0)
    oy, ox = float(spec.target_grid_2d.origin[0]), float(spec.target_grid_2d.origin[1])
    sy, sx = float(spec.target_grid_2d.spacing[0]), float(spec.target_grid_2d.spacing[1])
    margin = max(float(spec.tsdf.mu_nm), float(spec.tsdf.open_tube_radius_nm))

    max_y = float(np.max(points[:, 1]))
    max_x = float(np.max(points[:, 0]))
    y_size = int(max(2, np.ceil((max(0.0, max_y - oy) + margin) / sy) + 2))
    x_size = int(max(2, np.ceil((max(0.0, max_x - ox) + margin) / sx) + 2))
    return y_size, x_size


def _meshgrid_xy(
    shape_yx: tuple[int, int],
    spec: SEMPrepareSpecV1,
) -> tuple[np.ndarray, np.ndarray]:
    y_size, x_size = shape_yx
    sy, sx = float(spec.target_grid_2d.spacing[0]), float(spec.target_grid_2d.spacing[1])
    oy, ox = float(spec.target_grid_2d.origin[0]), float(spec.target_grid_2d.origin[1])

    ys = oy + np.arange(y_size, dtype=np.float32) * sy
    xs = ox + np.arange(x_size, dtype=np.float32) * sx
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x, grid_y


def _points_in_polygon(
    px: np.ndarray,
    py: np.ndarray,
    polygon_xy: np.ndarray,
) -> np.ndarray:
    poly = np.asarray(polygon_xy, dtype=np.float64)
    if not np.allclose(poly[0], poly[-1], rtol=0.0, atol=1e-6):
        poly = np.vstack([poly, poly[0]])

    inside = np.zeros(px.shape, dtype=bool)
    x = poly[:, 0]
    y = poly[:, 1]
    for i in range(len(poly) - 1):
        x0, y0 = x[i], y[i]
        x1, y1 = x[i + 1], y[i + 1]
        cond = (y0 > py) != (y1 > py)
        cross_x = (x1 - x0) * (py - y0) / ((y1 - y0) + 1e-12) + x0
        inside ^= cond & (px < cross_x)
    return inside


def _rasterize_closed_mask(
    contours: list[NormalizedContourLoop],
    shape_yx: tuple[int, int],
    spec: SEMPrepareSpecV1,
) -> np.ndarray:
    grid_x, grid_y = _meshgrid_xy(shape_yx, spec)
    px = grid_x.reshape(-1)
    py = grid_y.reshape(-1)

    mask = np.zeros(shape_yx, dtype=bool)
    for loop in contours:
        if not loop.is_closed:
            continue
        poly_mask = _points_in_polygon(px, py, loop.points_sim_nm).reshape(shape_yx)
        if loop.is_hole:
            mask &= ~poly_mask
        else:
            mask |= poly_mask
    return mask.astype(np.uint8)


def _distance_to_polyline(query_xy: np.ndarray, poly_xy: np.ndarray) -> np.ndarray:
    p = np.asarray(query_xy, dtype=np.float64)
    line = np.asarray(poly_xy, dtype=np.float64)
    if line.shape[0] < 2:
        return np.full((p.shape[0],), np.inf, dtype=np.float32)

    dist2 = np.full((p.shape[0],), np.inf, dtype=np.float64)
    for idx in range(line.shape[0] - 1):
        a = line[idx]
        b = line[idx + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            candidate = np.sum((p - a) ** 2, axis=1)
            dist2 = np.minimum(dist2, candidate)
            continue
        t = np.sum((p - a) * ab, axis=1) / denom
        t = np.clip(t, 0.0, 1.0)
        proj = a + t[:, None] * ab[None, :]
        candidate = np.sum((p - proj) ** 2, axis=1)
        dist2 = np.minimum(dist2, candidate)
    return np.sqrt(dist2).astype(np.float32, copy=False)


def _distance_map_to_curves(
    contours: list[NormalizedContourLoop],
    shape_yx: tuple[int, int],
    spec: SEMPrepareSpecV1,
) -> np.ndarray:
    grid_x, grid_y = _meshgrid_xy(shape_yx, spec)
    query = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)

    min_dist = np.full((query.shape[0],), np.inf, dtype=np.float32)
    for loop in contours:
        d = _distance_to_polyline(query, loop.points_sim_nm)
        min_dist = np.minimum(min_dist, d)
    return min_dist.reshape(shape_yx)


def _obs_loops(contours: list[NormalizedContourLoop]) -> list[ContourLoop]:
    loops: list[ContourLoop] = []
    for loop in contours:
        loops.append(
            ContourLoop(
                points_xy=np.asarray(loop.points_sim_nm, dtype=np.float32),
                is_hole=bool(loop.is_hole),
                label=loop.role,
                meta={
                    "is_closed": loop.is_closed,
                    "role": loop.role,
                    **dict(loop.meta),
                },
            )
        )
    return loops


def _weight_map(
    shape_yx: tuple[int, int],
    spec: SEMPrepareSpecV1,
    image_raw: SEMImageRaw | None,
    notes: list[str],
) -> np.ndarray | None:
    if spec.weight.mode == "none":
        return None
    if spec.weight.mode == "uniform":
        return np.full(shape_yx, float(spec.weight.default_weight), dtype=np.float32)

    if image_raw is None:
        notes.append("weight_mode=from_image but image missing, fallback to uniform")
        return np.full(shape_yx, float(spec.weight.default_weight), dtype=np.float32)

    img = np.asarray(image_raw.image, dtype=np.float32)
    y_idx = np.linspace(0, img.shape[0] - 1, num=shape_yx[0]).astype(np.int32)
    x_idx = np.linspace(0, img.shape[1] - 1, num=shape_yx[1]).astype(np.int32)
    resized = img[np.ix_(y_idx, x_idx)]
    v_min = float(np.min(resized))
    v_max = float(np.max(resized))
    if v_max <= v_min:
        return np.full(shape_yx, float(spec.weight.default_weight), dtype=np.float32)
    normalized = (resized - v_min) / (v_max - v_min)
    return (normalized.astype(np.float32) * float(spec.weight.default_weight)).astype(
        np.float32,
        copy=False,
    )


def _build_meta(
    spec: SEMPrepareSpecV1,
    *,
    distance_type: str,
    qa: SEMQA,
    input_hash: str,
    generator_version: str,
    git_commit: str,
    created_at: str | None,
    source_contour_path: str,
    source_image_path: str | None,
    transform_chain: TransformChain,
) -> Meta:
    spec_hash = sem_prepare_spec_hash(spec)
    chain_hash = hash_config(
        {
            "T_px_to_sem_nm": transform_chain.T_px_to_sem_nm,
            "T_sem_nm_to_sim_nm": transform_chain.T_sem_nm_to_sim_nm,
            "T_px_to_sim_nm": transform_chain.T_px_to_sim_nm,
        }
    )
    return Meta(
        schema_version="sem_obs/v1",
        profile_id=spec.profile_id,
        config_hash=spec_hash,
        generator_version=generator_version,
        git_commit=git_commit,
        input_hash=input_hash,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        extra={
            "sem_prepare_spec_hash": spec_hash,
            "sem_prepare_spec_version": spec.schema_version,
            "distance_type": distance_type,
            "qa_status": qa.status,
            "qa_notes": " | ".join(qa.notes),
            "source_contour_path": source_contour_path,
            "source_image_path": source_image_path or "",
            "pixel_size_nm": str(spec.input.pixel_size_nm),
            "transform_chain_hash": chain_hash,
        },
    )


def build_sem_obs2d(
    contours: list[NormalizedContourLoop],
    spec: SEMPrepareSpecV1,
    *,
    source_contour_path: str,
    source_image_path: str | None,
    image_raw: SEMImageRaw | None,
    transform_chain: TransformChain,
    input_hash: str,
    generator_version: str,
    git_commit: str,
    created_at: str | None = None,
) -> tuple[Obs2D, SEMQA, dict[str, object]]:
    if not contours:
        raise ValueError("contours must be non-empty")

    shape_yx = _grid_shape_from_contours(contours, spec, image_raw)
    open_loops = [loop for loop in contours if not loop.is_closed]
    closed_loops = [loop for loop in contours if loop.is_closed]

    notes: list[str] = []
    mode = spec.tsdf.mode
    if mode == "auto":
        mode = "unsigned_curve" if len(open_loops) > 0 else "signed_region"

    if mode == "signed_region" and len(open_loops) > 0 and spec.tsdf.mode == "signed_region":
        notes.append("open contours present with signed_region; signed TSDF uses closed loops only")

    if mode == "signed_region":
        mask = _rasterize_closed_mask(closed_loops, shape_yx, spec)
        tsdf = tsdf2d_from_mask(
            mask=mask,
            grid2d=spec.target_grid_2d,
            mu_nm=float(spec.tsdf.mu_nm),
            backend=spec.tsdf.distance_backend,
        ).astype(np.float32, copy=False)
        distance_type = "signed_region"
    else:
        source_loops = open_loops if open_loops else contours
        distance_nm = _distance_map_to_curves(source_loops, shape_yx, spec)
        tsdf = np.clip(distance_nm / float(spec.tsdf.mu_nm), 0.0, 1.0).astype(
            np.float32,
            copy=False,
        )
        mask = (distance_nm <= float(spec.tsdf.open_tube_radius_nm)).astype(np.uint8)
        distance_type = "unsigned_curve"

    weight = _weight_map(shape_yx, spec, image_raw, notes)
    loops_obs = _obs_loops(contours)
    transform_dict = {
        "T_px_to_sem_nm": transform_chain.T_px_to_sem_nm.tolist(),
        "T_sem_nm_to_sim_nm": transform_chain.T_sem_nm_to_sim_nm.tolist(),
        "T_px_to_sim_nm": transform_chain.T_px_to_sim_nm.tolist(),
        "meta": dict(transform_chain.meta),
    }

    provisional_meta = Meta(
        schema_version="sem_obs/v1",
        profile_id=spec.profile_id,
        config_hash=sem_prepare_spec_hash(spec),
        generator_version=generator_version,
        git_commit=git_commit,
        input_hash=input_hash,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        extra={"qa_status": "OK"},
    )
    obs = Obs2D(
        grid2d=spec.target_grid_2d,
        mask=mask.astype(np.uint8, copy=False),
        tsdf=tsdf.astype(np.float32, copy=False),
        loops=loops_obs,
        weight=weight,
        transform=transform_dict,
        debug_maps={"band_mask": (np.abs(tsdf) < 1.0).astype(np.uint8)},
        meta=provisional_meta,
    )
    qa = compute_sem_qa(obs, contours, spec, distance_type=distance_type, notes=notes)
    meta = _build_meta(
        spec,
        distance_type=distance_type,
        qa=qa,
        input_hash=input_hash,
        generator_version=generator_version,
        git_commit=git_commit,
        created_at=created_at,
        source_contour_path=source_contour_path,
        source_image_path=source_image_path,
        transform_chain=transform_chain,
    )

    final_obs = Obs2D(
        grid2d=spec.target_grid_2d,
        mask=obs.mask,
        tsdf=obs.tsdf,
        loops=obs.loops,
        weight=obs.weight,
        transform=obs.transform,
        debug_maps=obs.debug_maps,
        meta=meta,
    )

    extra_payload: dict[str, object] = {
        "transform_chain": {
            "T_px_to_sem_nm": transform_chain.T_px_to_sem_nm,
            "T_sem_nm_to_sim_nm": transform_chain.T_sem_nm_to_sim_nm,
            "T_px_to_sim_nm": transform_chain.T_px_to_sim_nm,
            "meta": dict(transform_chain.meta),
        },
        "distance_type": distance_type,
        "normalized_contours": [
            {
                "points_nm": loop.points_nm,
                "points_sim_nm": loop.points_sim_nm,
                "is_closed": loop.is_closed,
                "is_hole": loop.is_hole,
                "role": loop.role,
                "meta": dict(loop.meta),
            }
            for loop in contours
        ],
        "sem_spec": asdict(spec),
    }

    if spec.overlay.enable:
        overlay: dict[str, object] = {"draw_contours": spec.overlay.draw_contours, "mask": obs.mask}
        if image_raw is not None:
            overlay["image"] = image_raw.image
        if spec.overlay.draw_contours:
            overlay["contour_points_xy"] = [loop.points_sim_nm for loop in contours]
        extra_payload["overlay"] = overlay

    return final_obs, qa, extra_payload
