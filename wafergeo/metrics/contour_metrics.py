from __future__ import annotations

from typing import Protocol, cast

import numpy as np

from wafergeo.core.types import Obs2D
from wafergeo.metrics.base import MetricResult
from wafergeo.metrics.context import build_contour_kdtree, contour_points_from_loops
from wafergeo.metrics.qa import fail_result
from wafergeo.metrics.robust import robust_loss
from wafergeo.metrics.spec import MeasurementSpecV1, MetricEntrySpec


class _KDTreeLike(Protocol):
    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...


def compute_contour_chamfer(
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
                "contour_chamfer",
                "1.0.0",
                fail_penalty=fail_penalty,
                message=pre_error,
            )

        use_holes = bool(entry.params.get("use_holes", False))
        pred_points = contour_points_from_loops(pred, use_holes=use_holes)
        if pred_points.shape[0] == 0:
            return fail_result(
                "contour_chamfer",
                "1.0.0",
                fail_penalty=fail_penalty,
                message="pred contour points are empty",
            )

        obs_points = np.asarray(context.get("contour_points_xy"), dtype=np.float32)
        if obs_points.ndim != 2 or obs_points.shape[1] != 2 or obs_points.shape[0] == 0:
            return fail_result(
                "contour_chamfer",
                "1.0.0",
                fail_penalty=fail_penalty,
                message="obs contour precompute is empty",
            )

        obs_tree_obj = context.get("contour_kdtree")
        if obs_tree_obj is None or not hasattr(obs_tree_obj, "query"):
            return fail_result(
                "contour_chamfer",
                "1.0.0",
                fail_penalty=fail_penalty,
                message="obs contour KDTree is missing",
            )
        obs_tree = cast(_KDTreeLike, obs_tree_obj)

        pred_tree = cast(_KDTreeLike, build_contour_kdtree(pred_points))
        d_pred_to_obs = np.asarray(obs_tree.query(pred_points)[0], dtype=np.float32)
        d_obs_to_pred = np.asarray(pred_tree.query(obs_points)[0], dtype=np.float32)

        robust_raw = entry.params.get("robust") if isinstance(entry.params, dict) else None
        robust_spec = robust_raw if isinstance(robust_raw, dict) else None
        loss_forward = float(np.mean(robust_loss(d_pred_to_obs, robust_spec)))
        loss_backward = float(np.mean(robust_loss(d_obs_to_pred, robust_spec)))
        loss = 0.5 * (loss_forward + loss_backward)

        report: dict[str, object] = {
            "loss": loss,
            "forward_mean_nm": float(np.mean(d_pred_to_obs)),
            "backward_mean_nm": float(np.mean(d_obs_to_pred)),
            "sym_chamfer_nm": float(0.5 * (np.mean(d_pred_to_obs) + np.mean(d_obs_to_pred))),
            "pred_points": int(pred_points.shape[0]),
            "obs_points": int(obs_points.shape[0]),
        }

        maps = {
            "pred_to_obs_dist": d_pred_to_obs,
            "obs_to_pred_dist": d_obs_to_pred,
        }
        return MetricResult(
            name="contour_chamfer",
            version="1.0.0",
            loss=loss,
            report=report,
            maps=maps,
            status="OK",
            messages=[],
            meta={"use_holes": use_holes},
        )
    except Exception as exc:
        return fail_result(
            "contour_chamfer",
            "1.0.0",
            fail_penalty=fail_penalty,
            message=f"unexpected error: {type(exc).__name__}: {exc}",
        )


class ContourChamferMetric:
    name = "contour_chamfer"
    version = "1.0.0"

    def precompute_obs(
        self,
        obs: Obs2D,
        entry: MetricEntrySpec,
        measurement: MeasurementSpecV1 | None,
    ) -> object | None:
        _ = measurement
        use_holes = bool(entry.params.get("use_holes", False))
        points = contour_points_from_loops(obs, use_holes=use_holes)
        if points.shape[0] == 0:
            return {
                "precompute_error": "obs contour points are empty",
                "contour_points_xy": points,
                "contour_kdtree": None,
            }
        kdtree = build_contour_kdtree(points)
        return {
            "contour_points_xy": points,
            "contour_kdtree": kdtree,
            "use_holes": use_holes,
        }

    def compute(
        self,
        pred: Obs2D,
        obs: Obs2D,
        entry: MetricEntrySpec,
        ctx: object | None,
        *,
        fail_penalty: float,
    ) -> MetricResult:
        return compute_contour_chamfer(
            pred,
            obs,
            entry,
            ctx,
            fail_penalty=fail_penalty,
        )
