from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite

import numpy as np

from wafergeo.core.hashing import hash_config
from wafergeo.core.meta import Meta
from wafergeo.core.types import LabelVolume, TSDFVolume
from wafergeo.sdf.boundary_features import compute_boundary_features_from_phi
from wafergeo.sdf.config import SDFBuildConfig
from wafergeo.sdf.edt import signed_distance_from_mask_with_distance_fn
from wafergeo.sdf.engines.registry import get_sdf_engine
from wafergeo.sdf.engines.spec import method_card_to_dict
from wafergeo.sdf.errors import InvalidMuError, InvalidSpacingError
from wafergeo.sdf.qa import SDFQA, compute_sdf_qa
from wafergeo.sdf.roi import expand_roi_with_margin, place_subvolume_back
from wafergeo.sdf.tsdf import to_tsdf


def _full_roi(shape_zyx: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    return (slice(0, shape_zyx[0], 1), slice(0, shape_zyx[1], 1), slice(0, shape_zyx[2], 1))


def _normalize_roi(
    roi_zyx: tuple[slice, slice, slice],
    shape_zyx: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    normalized: list[slice] = []
    for axis, (roi_axis, axis_size) in enumerate(zip(roi_zyx, shape_zyx, strict=True)):
        if roi_axis.step not in (None, 1):
            raise ValueError("roi_zyx step is not supported")
        start = 0 if roi_axis.start is None else int(roi_axis.start)
        stop = axis_size if roi_axis.stop is None else int(roi_axis.stop)
        if start < 0 or stop > axis_size or start >= stop:
            raise ValueError(f"invalid roi_zyx axis={axis}: start={start}, stop={stop}")
        normalized.append(slice(start, stop, 1))
    return normalized[0], normalized[1], normalized[2]


def _local_roi(
    target_roi: tuple[slice, slice, slice],
    compute_roi: tuple[slice, slice, slice],
) -> tuple[slice, slice, slice]:
    z_start = target_roi[0].start - compute_roi[0].start
    z_stop = target_roi[0].stop - compute_roi[0].start
    y_start = target_roi[1].start - compute_roi[1].start
    y_stop = target_roi[1].stop - compute_roi[1].start
    x_start = target_roi[2].start - compute_roi[2].start
    x_stop = target_roi[2].stop - compute_roi[2].start
    return (
        slice(z_start, z_stop, 1),
        slice(y_start, y_stop, 1),
        slice(x_start, x_stop, 1),
    )


def _validate_input(label: LabelVolume, cfg: SDFBuildConfig) -> None:
    if not np.isfinite(cfg.mu_nm) or cfg.mu_nm <= 0.0:
        raise InvalidMuError(f"mu_nm must be finite and > 0, got {cfg.mu_nm}")
    if label.material_id.ndim != 3:
        raise ValueError(f"label.material_id must be 3D, got ndim={label.material_id.ndim}")
    if len(label.material.ids) > 5:
        raise ValueError("material count must be <= 5")

    spacing = label.grid.spacing
    if len(spacing) != 3:
        raise InvalidSpacingError(f"grid spacing must be length 3, got {len(spacing)}")
    for idx, value in enumerate(spacing):
        if not isfinite(value) or value <= 0.0:
            raise InvalidSpacingError(f"spacing[{idx}] must be finite and > 0, got {value}")


def _select_material_ids(label: LabelVolume, include_void_channel: bool) -> list[int]:
    if include_void_channel:
        return list(label.material.ids)
    return [
        material_id
        for material_id in label.material.ids
        if material_id != label.material.void_id
    ]


def build_tsdf_volume(label: LabelVolume, cfg: SDFBuildConfig) -> tuple[TSDFVolume, SDFQA]:
    _validate_input(label, cfg)
    engine = get_sdf_engine(cfg.backend)

    selected_material_ids = _select_material_ids(label, cfg.include_void_channel)
    if not selected_material_ids:
        raise ValueError("selected material list is empty")

    spacing_zyx = (
        float(label.grid.spacing[0]),
        float(label.grid.spacing[1]),
        float(label.grid.spacing[2]),
    )
    full_shape = label.material_id.shape

    target_roi = _full_roi(full_shape)
    compute_roi = target_roi
    roi_applied = False
    if cfg.roi_zyx is not None:
        roi_applied = True
        target_roi = _normalize_roi(cfg.roi_zyx, full_shape)
        margin_nm = float(cfg.roi_margin_nm if cfg.roi_margin_nm is not None else cfg.mu_nm)
        compute_roi = expand_roi_with_margin(target_roi, full_shape, spacing_zyx, margin_nm)

    sub_label = label.material_id[compute_roi]
    m = len(selected_material_ids)
    sub_shape = sub_label.shape

    work_tsdf = np.empty((m,) + sub_shape, dtype=np.float32)
    phi_stack: np.ndarray | None = (
        np.empty((m,) + sub_shape, dtype=np.float32) if cfg.boundary_features else None
    )

    present_mask: np.ndarray | None = None
    if cfg.compute_present_mask:
        present_mask = np.array(
            [np.any(label.material_id == material_id) for material_id in selected_material_ids],
            dtype=bool,
        )

    for channel, material_id in enumerate(selected_material_ids):
        mask = sub_label == material_id

        if np.all(mask):
            work_tsdf[channel].fill(-1.0)
            if phi_stack is not None:
                phi_stack[channel].fill(-float(cfg.mu_nm))
            continue
        if not np.any(mask):
            work_tsdf[channel].fill(+1.0)
            if phi_stack is not None:
                phi_stack[channel].fill(+float(cfg.mu_nm))
            continue

        phi_nm = signed_distance_from_mask_with_distance_fn(mask, spacing_zyx, engine.distance)
        work_tsdf[channel] = to_tsdf(phi_nm, cfg.mu_nm, out_dtype=np.float32)
        if phi_stack is not None:
            phi_stack[channel] = phi_nm

    tsdf_dtype = np.float16 if cfg.tsdf_store_dtype == "float16" else np.float32
    sub_tsdf_store = work_tsdf.astype(tsdf_dtype, copy=False)

    sub_d_boundary: np.ndarray | None = None
    sub_pair_code: np.ndarray | None = None
    if cfg.boundary_features and phi_stack is not None:
        sub_d_boundary, sub_pair_code = compute_boundary_features_from_phi(
            phi_stack,
            mu_nm=cfg.mu_nm,
            outside_code=cfg.pair_code_outside_band,
            band_only=cfg.band_only_pair_code,
        )

    if cfg.roi_zyx is None:
        tsdf_full = sub_tsdf_store
        d_boundary_full = sub_d_boundary
        pair_code_full = sub_pair_code
    else:
        local = _local_roi(target_roi, compute_roi)
        tsdf_target = sub_tsdf_store[:, local[0], local[1], local[2]]
        tsdf_full = place_subvolume_back(
            (m,) + full_shape,
            target_roi,
            tsdf_target,
            fill_value=1.0,
        )

        if sub_d_boundary is None:
            d_boundary_full = None
        else:
            d_target = sub_d_boundary[local[0], local[1], local[2]]
            d_boundary_full = place_subvolume_back(full_shape, target_roi, d_target, fill_value=1.0)

        if sub_pair_code is None:
            pair_code_full = None
        else:
            p_target = sub_pair_code[local[0], local[1], local[2]]
            pair_code_full = place_subvolume_back(
                full_shape,
                target_roi,
                p_target,
                fill_value=cfg.pair_code_outside_band,
            )

    created_at = datetime.now(timezone.utc).isoformat()  # noqa: UP017
    extra = dict(label.meta.extra)
    extra.update(
        {
            "sdf_backend": cfg.backend,
            "sdf_engine_name": engine.name,
            "sdf_engine_version": engine.version,
            "sdf_engine_dependencies": ",".join(engine.method_card.dependencies),
            "sdf_engine_exact": str(engine.capabilities.exact),
            "sdf_engine_gpu": str(engine.capabilities.gpu_accelerated),
            "sdf_engine_method_card": str(method_card_to_dict(engine.method_card)),
            "mu_nm": str(cfg.mu_nm),
            "selected_material_ids": ",".join(str(v) for v in selected_material_ids),
            "roi_applied": str(roi_applied),
        }
    )
    tsdf_meta = Meta(
        schema_version=cfg.schema_version,
        profile_id=cfg.profile_id,
        config_hash=hash_config(cfg.to_hash_payload()),
        generator_version=label.meta.generator_version,
        git_commit=label.meta.git_commit,
        input_hash=label.meta.input_hash,
        created_at=created_at,
        extra=extra,
    )

    tsdf_volume = TSDFVolume(
        grid=label.grid,
        material=label.material,
        mu_nm=cfg.mu_nm,
        tsdf=tsdf_full,
        d_boundary=d_boundary_full,
        pair_code=pair_code_full,
        present_mask=present_mask,
        meta=tsdf_meta,
    )
    qa = compute_sdf_qa(tsdf_volume, label, cfg, selected_material_ids)
    return tsdf_volume, qa
