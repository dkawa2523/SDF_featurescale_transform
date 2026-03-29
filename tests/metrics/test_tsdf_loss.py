from __future__ import annotations

import numpy as np

from tests.metrics.helpers import build_obs_pair_shifted
from wafergeo.metrics.spec import MetricEntrySpec
from wafergeo.metrics.tsdf_loss import compute_tsdf_band_robust_weight


def test_identity_near_zero() -> None:
    pred, obs, _ = build_obs_pair_shifted(shift_x=0)
    entry = MetricEntrySpec(
        name="tsdf_band_robust_weight",
        weight=1.0,
        observers=None,
        params={"band": "obs_band", "robust": {"type": "huber", "delta": 0.1}},
    )

    result = compute_tsdf_band_robust_weight(pred, obs, entry, None, fail_penalty=1e6)

    assert result.status == "OK"
    assert result.loss <= 1e-6


def test_shift_increases() -> None:
    pred_same, obs, _ = build_obs_pair_shifted(shift_x=0)
    pred_shift, _, _ = build_obs_pair_shifted(shift_x=1)
    entry = MetricEntrySpec(
        name="tsdf_band_robust_weight",
        weight=1.0,
        observers=None,
        params={"band": "obs_band", "robust": {"type": "huber", "delta": 0.1}},
    )

    result_same = compute_tsdf_band_robust_weight(pred_same, obs, entry, None, fail_penalty=1e6)
    result_shift = compute_tsdf_band_robust_weight(pred_shift, obs, entry, None, fail_penalty=1e6)

    assert result_shift.status == "OK"
    assert result_shift.loss > result_same.loss


def test_sem_weight_affects_loss() -> None:
    pred, obs, _ = build_obs_pair_shifted(shift_x=1, with_weight=True)
    assert obs.weight is not None

    uniform_entry = MetricEntrySpec(
        name="tsdf_band_robust_weight",
        weight=1.0,
        observers=None,
        params={"band": "obs_band", "weight_mode": "uniform"},
    )
    sem_entry = MetricEntrySpec(
        name="tsdf_band_robust_weight",
        weight=1.0,
        observers=None,
        params={"band": "obs_band", "weight_mode": "sem_weight"},
    )

    uniform = compute_tsdf_band_robust_weight(pred, obs, uniform_entry, None, fail_penalty=1e6)
    sem = compute_tsdf_band_robust_weight(pred, obs, sem_entry, None, fail_penalty=1e6)

    assert uniform.status == "OK"
    assert sem.status == "OK"
    assert np.isfinite(uniform.loss)
    assert np.isfinite(sem.loss)
    assert sem.loss != uniform.loss
