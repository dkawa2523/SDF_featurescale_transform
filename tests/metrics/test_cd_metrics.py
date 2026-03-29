from __future__ import annotations

from tests.metrics.helpers import build_obs_pair_shifted
from wafergeo.metrics.cd_metrics import CDLineScanMetric, compute_cd_linescan, zero_crossings_1d
from wafergeo.metrics.spec import MeasurementLineSpec, MeasurementSpecV1, MetricEntrySpec


def _measurement() -> MeasurementSpecV1:
    return MeasurementSpecV1(
        schema_version="measurement/v1",
        name="cd_test",
        lines=(
            MeasurementLineSpec(
                id="center",
                axis="x",
                coord_nm=160.0,
                range_nm=(40.0, 280.0),
                expected_edges=2,
                edge_pair="outer",
                method="tsdf_zero_cross",
            ),
        ),
    )


def test_zero_crossings_1d_basic() -> None:
    values = [-1.0, -0.5, 0.5, 1.0]
    coords = [0.0, 1.0, 2.0, 3.0]
    out = zero_crossings_1d(values, coords)
    assert out.size == 1
    assert 1.0 < float(out[0]) < 2.0


def test_identity_near_zero() -> None:
    pred, obs, _ = build_obs_pair_shifted(shift_x=0)
    metric = CDLineScanMetric()
    entry = MetricEntrySpec(
        name="cd_linescan",
        weight=1.0,
        observers=None,
        params={"robust": {"type": "l1"}},
        measurement_ref="cd_ref",
    )

    ctx = metric.precompute_obs(obs, entry, _measurement())
    result = compute_cd_linescan(pred, obs, entry, ctx, fail_penalty=1e6)

    assert result.status == "OK"
    assert result.loss <= 1e-6


def test_translation_increases() -> None:
    pred_same, obs, _ = build_obs_pair_shifted(shift_x=0)
    pred_shift, _, _ = build_obs_pair_shifted(shift_x=1)
    metric = CDLineScanMetric()
    entry = MetricEntrySpec(
        name="cd_linescan",
        weight=1.0,
        observers=None,
        params={"robust": {"type": "l1"}},
        measurement_ref="cd_ref",
    )

    ctx = metric.precompute_obs(obs, entry, _measurement())
    same = compute_cd_linescan(pred_same, obs, entry, ctx, fail_penalty=1e6)
    shifted = compute_cd_linescan(pred_shift, obs, entry, ctx, fail_penalty=1e6)

    assert shifted.status == "OK"
    assert shifted.loss > same.loss
