from __future__ import annotations

import pytest

from tests.metrics.helpers import build_obs_pair_shifted
from wafergeo.metrics.contour_metrics import ContourChamferMetric, compute_contour_chamfer
from wafergeo.metrics.spec import MetricEntrySpec

pytest.importorskip("scipy")


def test_identity_near_zero() -> None:
    pred, obs, _ = build_obs_pair_shifted(shift_x=0)
    metric = ContourChamferMetric()
    entry = MetricEntrySpec(
        name="contour_chamfer",
        weight=1.0,
        observers=None,
        params={"use_holes": False, "robust": {"type": "l1"}},
    )

    ctx = metric.precompute_obs(obs, entry, measurement=None)
    result = compute_contour_chamfer(pred, obs, entry, ctx, fail_penalty=1e6)

    assert result.status == "OK"
    assert result.loss <= 1e-6


def test_translation_increases() -> None:
    pred_same, obs, _ = build_obs_pair_shifted(shift_x=0)
    pred_shift, _, _ = build_obs_pair_shifted(shift_x=2)
    metric = ContourChamferMetric()
    entry = MetricEntrySpec(
        name="contour_chamfer",
        weight=1.0,
        observers=None,
        params={"use_holes": False, "robust": {"type": "l1"}},
    )

    ctx = metric.precompute_obs(obs, entry, measurement=None)
    same = compute_contour_chamfer(pred_same, obs, entry, ctx, fail_penalty=1e6)
    shifted = compute_contour_chamfer(pred_shift, obs, entry, ctx, fail_penalty=1e6)

    assert shifted.status == "OK"
    assert shifted.loss > same.loss
