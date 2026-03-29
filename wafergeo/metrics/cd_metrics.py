from __future__ import annotations

import numpy as np

from wafergeo.core.types import Obs2D
from wafergeo.metrics.base import MetricResult
from wafergeo.metrics.context import LineScanCacheEntry, build_line_scan_cache
from wafergeo.metrics.qa import fail_result
from wafergeo.metrics.robust import robust_loss
from wafergeo.metrics.spec import MeasurementLineSpec, MeasurementSpecV1, MetricEntrySpec


def zero_crossings_1d(values: np.ndarray, coords_nm: np.ndarray) -> np.ndarray:
    val = np.asarray(values, dtype=np.float32)
    coord = np.asarray(coords_nm, dtype=np.float32)
    if val.ndim != 1 or coord.ndim != 1 or val.size != coord.size:
        raise ValueError("values/coords_nm must be 1D with same length")
    if val.size < 2:
        return np.zeros((0,), dtype=np.float32)

    out: list[float] = []
    for idx in range(val.size - 1):
        v0 = float(val[idx])
        v1 = float(val[idx + 1])
        c0 = float(coord[idx])
        c1 = float(coord[idx + 1])

        if v0 == 0.0:
            out.append(c0)

        if v0 * v1 < 0.0:
            t = -v0 / (v1 - v0)
            out.append(c0 + t * (c1 - c0))

        if idx == val.size - 2 and v1 == 0.0:
            out.append(c1)

    if not out:
        return np.zeros((0,), dtype=np.float32)

    arr = np.asarray(sorted(out), dtype=np.float32)
    uniq: list[float] = [float(arr[0])]
    for value in arr[1:]:
        if abs(float(value) - uniq[-1]) > 1e-6:
            uniq.append(float(value))
    return np.asarray(uniq, dtype=np.float32)


def _extract_line_values(tsdf2d: np.ndarray, cache: LineScanCacheEntry) -> np.ndarray:
    if cache.axis == "x":
        return tsdf2d[cache.fixed_index, cache.scan_indices]
    return tsdf2d[cache.scan_indices, cache.fixed_index]


def _select_edge_pair(crossings: np.ndarray, edge_pair: str) -> tuple[float, float]:
    if crossings.size < 2:
        raise ValueError("need at least 2 crossings to form CD pair")

    if edge_pair == "outer":
        return float(crossings[0]), float(crossings[-1])

    if edge_pair == "inner":
        if crossings.size < 4:
            raise ValueError("edge_pair='inner' requires at least 4 crossings")
        mid = crossings.size // 2
        return float(crossings[mid - 1]), float(crossings[mid])

    raise ValueError(f"unsupported edge_pair: {edge_pair}")


def _line_cd_from_tsdf(
    tsdf2d: np.ndarray,
    line: MeasurementLineSpec,
    cache: LineScanCacheEntry,
) -> tuple[float, np.ndarray, tuple[float, float]]:
    values = _extract_line_values(tsdf2d, cache)
    crossings = zero_crossings_1d(values, cache.scan_coords_nm)
    if crossings.size != line.expected_edges:
        raise ValueError(
            f"line '{line.id}' expected {line.expected_edges} crossings, got {crossings.size}"
        )
    left, right = _select_edge_pair(crossings, line.edge_pair)
    return float(abs(right - left)), crossings, (left, right)


def compute_cd_linescan(
    pred: Obs2D,
    obs: Obs2D,
    entry: MetricEntrySpec,
    ctx: object | None,
    *,
    fail_penalty: float,
) -> MetricResult:
    try:
        context = dict(ctx) if isinstance(ctx, dict) else {}
        pre_error = context.get("precompute_error")
        if isinstance(pre_error, str):
            return fail_result(
                "cd_linescan",
                "1.0.0",
                fail_penalty=fail_penalty,
                message=pre_error,
            )

        measurement = context.get("measurement")
        if not isinstance(measurement, MeasurementSpecV1):
            return fail_result(
                "cd_linescan",
                "1.0.0",
                fail_penalty=fail_penalty,
                message="measurement spec context is missing",
            )

        line_cache = context.get("line_scan_cache")
        if not isinstance(line_cache, dict):
            return fail_result(
                "cd_linescan",
                "1.0.0",
                fail_penalty=fail_penalty,
                message="line scan cache is missing",
            )

        obs_cd_by_line = context.get("obs_cd_by_line")
        if not isinstance(obs_cd_by_line, dict):
            return fail_result(
                "cd_linescan",
                "1.0.0",
                fail_penalty=fail_penalty,
                message="obs CD precompute is missing",
            )

        pred_tsdf = np.asarray(pred.tsdf, dtype=np.float32)
        obs_tsdf = np.asarray(obs.tsdf, dtype=np.float32)
        if pred_tsdf.shape != obs_tsdf.shape:
            return fail_result(
                "cd_linescan",
                "1.0.0",
                fail_penalty=fail_penalty,
                message=f"shape mismatch: pred={pred_tsdf.shape}, obs={obs_tsdf.shape}",
            )

        line_reports: dict[str, dict[str, object]] = {}
        cd_errors_nm: list[float] = []
        line_loss_inputs_nm: list[float] = []

        for line in measurement.lines:
            cache = line_cache.get(line.id)
            if not isinstance(cache, LineScanCacheEntry):
                line_reports[line.id] = {"status": "FAIL", "reason": "cache missing"}
                return fail_result(
                    "cd_linescan",
                    "1.0.0",
                    fail_penalty=fail_penalty,
                    message=f"line '{line.id}' cache missing",
                    report={"lines": line_reports},
                )

            try:
                pred_cd, pred_cross, pred_pair = _line_cd_from_tsdf(pred_tsdf, line, cache)
                obs_cd = float(obs_cd_by_line[line.id]["cd_nm"])
                obs_pair = obs_cd_by_line[line.id]["pair_nm"]
            except Exception as line_exc:
                line_reports[line.id] = {
                    "status": "FAIL",
                    "reason": f"{type(line_exc).__name__}: {line_exc}",
                }
                return fail_result(
                    "cd_linescan",
                    "1.0.0",
                    fail_penalty=fail_penalty,
                    message=f"line '{line.id}' failed: {line_exc}",
                    report={"lines": line_reports},
                )

            cd_err = float(pred_cd - obs_cd)
            left_err = float(pred_pair[0] - obs_pair[0])
            right_err = float(pred_pair[1] - obs_pair[1])
            position_err = 0.5 * (abs(left_err) + abs(right_err))
            cd_errors_nm.append(cd_err)
            line_loss_inputs_nm.append(position_err)
            line_reports[line.id] = {
                "status": "OK",
                "pred_cd_nm": pred_cd,
                "obs_cd_nm": obs_cd,
                "cd_error_nm": cd_err,
                "edge_left_error_nm": left_err,
                "edge_right_error_nm": right_err,
                "position_error_nm": position_err,
                "pred_pair_nm": pred_pair,
                "obs_pair_nm": obs_pair,
                "pred_crossings_nm": pred_cross.tolist(),
            }

        if not line_loss_inputs_nm:
            return fail_result(
                "cd_linescan",
                "1.0.0",
                fail_penalty=fail_penalty,
                message="no valid line-scan results",
            )

        loss_input_arr = np.asarray(line_loss_inputs_nm, dtype=np.float32)
        cd_err_arr = np.asarray(cd_errors_nm, dtype=np.float32)
        robust_raw = entry.params.get("robust") if isinstance(entry.params, dict) else None
        robust_spec = robust_raw if isinstance(robust_raw, dict) else None
        robust_vals = robust_loss(loss_input_arr, robust_spec)
        loss = float(np.mean(robust_vals))

        return MetricResult(
            name="cd_linescan",
            version="1.0.0",
            loss=loss,
            report={
                "line_count": len(line_loss_inputs_nm),
                "mean_abs_cd_error_nm": float(np.mean(np.abs(cd_err_arr))),
                "max_abs_cd_error_nm": float(np.max(np.abs(cd_err_arr))),
                "mean_position_error_nm": float(np.mean(loss_input_arr)),
                "lines": line_reports,
            },
            maps={
                "cd_error_nm": cd_err_arr.astype(np.float32, copy=False),
                "position_error_nm": loss_input_arr.astype(np.float32, copy=False),
            },
            status="OK",
            messages=[],
            meta={"measurement_name": measurement.name},
        )
    except Exception as exc:
        return fail_result(
            "cd_linescan",
            "1.0.0",
            fail_penalty=fail_penalty,
            message=f"unexpected error: {type(exc).__name__}: {exc}",
        )


class CDLineScanMetric:
    name = "cd_linescan"
    version = "1.0.0"

    def precompute_obs(
        self,
        obs: Obs2D,
        entry: MetricEntrySpec,
        measurement: MeasurementSpecV1 | None,
    ) -> object | None:
        _ = entry
        if measurement is None:
            return {"precompute_error": "measurement spec is required for cd_linescan"}

        try:
            line_cache = build_line_scan_cache(obs, measurement)
            obs_tsdf = np.asarray(obs.tsdf, dtype=np.float32)
            obs_cd_by_line: dict[str, dict[str, object]] = {}
            for line in measurement.lines:
                cache = line_cache[line.id]
                cd_nm, crossings, pair = _line_cd_from_tsdf(obs_tsdf, line, cache)
                obs_cd_by_line[line.id] = {
                    "cd_nm": cd_nm,
                    "crossings_nm": crossings,
                    "pair_nm": pair,
                }
            return {
                "measurement": measurement,
                "line_scan_cache": line_cache,
                "obs_cd_by_line": obs_cd_by_line,
            }
        except Exception as exc:
            return {"precompute_error": f"{type(exc).__name__}: {exc}"}

    def compute(
        self,
        pred: Obs2D,
        obs: Obs2D,
        entry: MetricEntrySpec,
        ctx: object | None,
        *,
        fail_penalty: float,
    ) -> MetricResult:
        return compute_cd_linescan(
            pred,
            obs,
            entry,
            ctx,
            fail_penalty=fail_penalty,
        )
