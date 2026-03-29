from __future__ import annotations

import numpy as np

from wafergeo.core.types import Obs2D
from wafergeo.metrics.base import MetricResult
from wafergeo.metrics.context import band_mask_from_obs
from wafergeo.metrics.qa import fail_result
from wafergeo.metrics.robust import robust_loss
from wafergeo.metrics.spec import MeasurementSpecV1, MetricEntrySpec
from wafergeo.metrics.weights import build_weight_map


def _resolve_band_mask(obs: Obs2D, pred: Obs2D, params: dict[str, object]) -> np.ndarray:
    band = str(params.get("band", "obs_band")).lower()
    obs_band = np.abs(np.asarray(obs.tsdf, dtype=np.float32)) < 1.0
    pred_band = np.abs(np.asarray(pred.tsdf, dtype=np.float32)) < 1.0

    if band == "obs_band":
        return obs_band
    if band == "pred_band":
        return pred_band
    if band == "union_band":
        return obs_band | pred_band
    raise ValueError(f"unsupported band: {band}")


def compute_tsdf_band_robust_weight(
    pred: Obs2D,
    obs: Obs2D,
    entry: MetricEntrySpec,
    ctx: object | None,
    *,
    fail_penalty: float,
) -> MetricResult:
    _ = ctx
    try:
        pred_tsdf = np.asarray(pred.tsdf, dtype=np.float32)
        obs_tsdf = np.asarray(obs.tsdf, dtype=np.float32)

        if pred_tsdf.shape != obs_tsdf.shape:
            return fail_result(
                "tsdf_band_robust_weight",
                "1.0.0",
                fail_penalty=fail_penalty,
                message=f"shape mismatch: pred={pred_tsdf.shape}, obs={obs_tsdf.shape}",
            )
        if not np.isfinite(pred_tsdf).all() or not np.isfinite(obs_tsdf).all():
            return fail_result(
                "tsdf_band_robust_weight",
                "1.0.0",
                fail_penalty=fail_penalty,
                message="pred/obs tsdf contains NaN/Inf",
            )

        band_mask = _resolve_band_mask(obs, pred, entry.params)
        valid_count = int(np.sum(band_mask))
        if valid_count == 0:
            return fail_result(
                "tsdf_band_robust_weight",
                "1.0.0",
                fail_penalty=fail_penalty,
                message="no valid pixels in band",
                maps={"band_mask": band_mask.astype(np.uint8, copy=False)},
            )

        weight_map = build_weight_map(obs, entry.params, band_mask)
        w_sum = float(np.sum(weight_map))
        if w_sum <= 0.0:
            return fail_result(
                "tsdf_band_robust_weight",
                "1.0.0",
                fail_penalty=fail_penalty,
                message="weight sum is zero in active band",
                maps={"weight_map": weight_map},
            )

        residual = pred_tsdf - obs_tsdf
        robust_raw = entry.params.get("robust") if isinstance(entry.params, dict) else None
        robust_spec = robust_raw if isinstance(robust_raw, dict) else None
        robust = robust_loss(residual, robust_spec)
        contrib = robust * weight_map
        loss = float(np.sum(contrib) / w_sum)

        raw_mu_nm = entry.params.get("mu_nm", 1.0)
        mu_nm = float(raw_mu_nm) if isinstance(raw_mu_nm, (int, float, str)) else 1.0
        report: dict[str, object] = {
            "loss": loss,
            "mean_abs_tsdf": float(np.sum(np.abs(residual) * weight_map) / w_sum),
            "mean_abs_nm": float(np.sum(np.abs(residual) * mu_nm * weight_map) / w_sum),
            "valid_pixels": valid_count,
            "weight_sum": w_sum,
        }
        maps = {
            "residual_map": residual.astype(np.float32, copy=False),
            "band_mask": band_mask.astype(np.uint8, copy=False),
            "weight_map": weight_map.astype(np.float32, copy=False),
            "robust_map": robust.astype(np.float32, copy=False),
            "loss_contrib_map": contrib.astype(np.float32, copy=False),
        }
        return MetricResult(
            name="tsdf_band_robust_weight",
            version="1.0.0",
            loss=loss,
            report=report,
            maps=maps,
            status="OK",
            messages=[],
            meta={"band": str(entry.params.get("band", "obs_band"))},
        )
    except Exception as exc:
        return fail_result(
            "tsdf_band_robust_weight",
            "1.0.0",
            fail_penalty=fail_penalty,
            message=f"unexpected error: {type(exc).__name__}: {exc}",
        )


class TsdfBandRobustWeightMetric:
    name = "tsdf_band_robust_weight"
    version = "1.0.0"

    def precompute_obs(
        self,
        obs: Obs2D,
        entry: MetricEntrySpec,
        measurement: MeasurementSpecV1 | None,
    ) -> object | None:
        _ = entry
        _ = measurement
        return {"band_mask_obs": band_mask_from_obs(obs)}

    def compute(
        self,
        pred: Obs2D,
        obs: Obs2D,
        entry: MetricEntrySpec,
        ctx: object | None,
        *,
        fail_penalty: float,
    ) -> MetricResult:
        return compute_tsdf_band_robust_weight(
            pred,
            obs,
            entry,
            ctx,
            fail_penalty=fail_penalty,
        )
