from __future__ import annotations

from tests.metrics.helpers import build_obs_pair_shifted
from wafergeo.metrics.aggregate import build_metric_context, compute_objective
from wafergeo.metrics.spec import MetricEntrySpec, MetricSpecV2


def test_weighted_sum_multi_observer_metric() -> None:
    pred_top, obs_top, _ = build_obs_pair_shifted(shift_x=1)
    pred_slice, obs_slice, _ = build_obs_pair_shifted(shift_x=0)

    spec = MetricSpecV2(
        schema_version="metric/v2",
        metric_set_id="agg_weighted",
        observer_weights={"topdown": 2.0, "slice": 0.5},
        metrics=(
            MetricEntrySpec(
                name="tsdf_band_robust_weight",
                weight=1.5,
                observers=("topdown", "slice"),
                params={"band": "obs_band"},
            ),
        ),
    )

    pred = {"topdown": pred_top, "slice": pred_slice}
    obs = {"topdown": obs_top, "slice": obs_slice}

    ctx = build_metric_context(obs, spec, {})
    out = compute_objective(pred, obs, spec, ctx)

    expected = 0.0
    for result in out.metric_results:
        expected += (
            float(result.meta["metric_weight"])
            * float(result.meta["observer_weight"])
            * result.loss
        )

    assert abs(out.total_loss - expected) < 1e-8
    assert out.status in {"OK", "WARN", "FAIL"}


def test_metric_result_contract_kept_on_fail() -> None:
    pred, obs, _ = build_obs_pair_shifted(shift_x=0)
    spec = MetricSpecV2(
        schema_version="metric/v2",
        metric_set_id="agg_fail",
        fail_penalty=1e6,
        metrics=(
            MetricEntrySpec(
                name="tsdf_band_robust_weight",
                weight=1.0,
                observers=("topdown",),
                params={"band": "obs_band"},
            ),
        ),
    )

    ctx = build_metric_context({"topdown": obs}, spec, {})
    out = compute_objective({"slice": pred}, {"topdown": obs}, spec, ctx)

    assert len(out.metric_results) == 1
    result = out.metric_results[0]
    assert result.status == "FAIL"
    assert result.loss == spec.fail_penalty
    assert isinstance(result.report, dict)
    assert isinstance(result.maps, dict)


def test_fail_returns_penalty_not_exception() -> None:
    _, obs, _ = build_obs_pair_shifted(shift_x=0)
    spec = MetricSpecV2(
        schema_version="metric/v2",
        metric_set_id="agg_fail2",
        fail_penalty=1e6,
        metrics=(
            MetricEntrySpec(
                name="contour_chamfer",
                weight=1.0,
                observers=("topdown",),
                params={"use_holes": False},
            ),
        ),
    )

    ctx = build_metric_context({"topdown": obs}, spec, {})
    out = compute_objective({}, {"topdown": obs}, spec, ctx)

    assert out.total_loss == spec.fail_penalty
    assert out.status == "FAIL"
    assert len(out.messages) >= 1
