from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wafergeo.core.types import LabelVolume, TSDFVolume
from wafergeo.sdf.config import SDFBuildConfig


@dataclass(frozen=True)
class SDFQA:
    tsdf_min: float
    tsdf_max: float
    nan_count: int
    inf_count: int
    band_fraction: float
    present_materials: dict[int, bool]
    grad_mag_mean: float
    grad_mag_std: float
    grad_unit_error_rate: float
    notes: list[str] = field(default_factory=list)


def compute_sdf_qa(
    tsdf_volume: TSDFVolume,
    label: LabelVolume,
    cfg: SDFBuildConfig,
    selected_material_ids: list[int],
) -> SDFQA:
    tsdf = np.asarray(tsdf_volume.tsdf, dtype=np.float32)
    nan_count = int(np.isnan(tsdf).sum())
    inf_count = int(np.isinf(tsdf).sum())
    tsdf_min = float(np.nanmin(tsdf))
    tsdf_max = float(np.nanmax(tsdf))

    min_abs = np.min(np.abs(tsdf), axis=0)
    band_mask = min_abs < 1.0
    band_fraction = float(np.mean(band_mask))

    present_materials: dict[int, bool] = {}
    if tsdf_volume.present_mask is None:
        for material_id in selected_material_ids:
            present_materials[int(material_id)] = bool(np.any(label.material_id == material_id))
    else:
        for idx, material_id in enumerate(selected_material_ids):
            present_materials[int(material_id)] = bool(np.asarray(tsdf_volume.present_mask)[idx])

    grad_mag_mean = 0.0
    grad_mag_std = 0.0
    grad_unit_error_rate = 0.0
    notes: list[str] = []

    if cfg.qa_grad_check:
        phi_nm_stack = tsdf * float(cfg.mu_nm)
        best_idx = np.argmin(np.abs(tsdf), axis=0)
        best_phi = np.take_along_axis(phi_nm_stack, best_idx[np.newaxis, ...], axis=0)[0]

        if best_phi.shape[0] == 1:
            gy, gx = np.gradient(
                best_phi[0],
                label.grid.spacing[1],
                label.grid.spacing[2],
                edge_order=1,
            )
            grad_mag = np.sqrt(gx * gx + gy * gy)
            grad_field = grad_mag[np.newaxis, ...]
        else:
            gz, gy, gx = np.gradient(
                best_phi,
                label.grid.spacing[0],
                label.grid.spacing[1],
                label.grid.spacing[2],
                edge_order=1,
            )
            grad_field = np.sqrt(gx * gx + gy * gy + gz * gz)

        if np.any(band_mask):
            values = grad_field[band_mask]
            grad_mag_mean = float(np.mean(values))
            grad_mag_std = float(np.std(values))
            grad_unit_error_rate = float(np.mean(np.abs(values - 1.0) > cfg.qa_grad_tolerance))
        else:
            notes.append("band_mask is empty; gradient QA skipped")
    else:
        notes.append("qa_grad_check disabled")

    if nan_count > 0 or inf_count > 0:
        notes.append("tsdf contains NaN/Inf")
    if tsdf_min < -1.0001 or tsdf_max > 1.0001:
        notes.append("tsdf range out of [-1,1]")

    return SDFQA(
        tsdf_min=tsdf_min,
        tsdf_max=tsdf_max,
        nan_count=nan_count,
        inf_count=inf_count,
        band_fraction=band_fraction,
        present_materials=present_materials,
        grad_mag_mean=grad_mag_mean,
        grad_mag_std=grad_mag_std,
        grad_unit_error_rate=grad_unit_error_rate,
        notes=notes,
    )
