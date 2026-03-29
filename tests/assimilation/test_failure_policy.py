from __future__ import annotations

import numpy as np

from tests.assimilation.helpers import count_assim_trial_dirs, make_case_and_store
from wafergeo.assimilation.evaluator import evaluate_one
from wafergeo.assimilation.session import create_evaluation_session


class AlwaysFailSurrogate:
    name = "always_fail"

    def predict(self, params: dict[str, object]):
        _ = params
        raise RuntimeError("forced failure")


class BadGeomSurrogate:
    name = "bad_geom"

    def predict(self, params: dict[str, object]):
        _ = params
        return {"not": "a geometry"}


def test_surrogate_exception_returns_fail_with_penalty(tmp_path) -> None:
    case, store, _ = make_case_and_store(
        tmp_path,
        model_obj=AlwaysFailSurrogate(),
        oob_policy="clamp",
    )
    session = create_evaluation_session(case, store)
    result = evaluate_one(session, np.array([0.0], dtype=np.float64), candidate_id="fail_surrogate")
    assert result.status == "FAIL"
    assert result.total_loss == case.failure_policy.penalty
    assert any("surrogate.predict failed" in message for message in result.messages)


def test_observer_shape_mismatch_returns_fail_with_penalty(tmp_path) -> None:
    case, store, _ = make_case_and_store(
        tmp_path,
        model_shape_zyx=(3, 20, 20),
        oob_policy="clamp",
    )
    session = create_evaluation_session(case, store)
    result = evaluate_one(session, np.array([0.0], dtype=np.float64), candidate_id="fail_observer")
    assert result.status == "FAIL"
    assert result.total_loss == case.failure_policy.penalty
    assert any("strict_sim_grid" in message for message in result.messages)


def test_surrogate_exception_policy_penalty_can_write_trial(tmp_path) -> None:
    case, store, _ = make_case_and_store(
        tmp_path,
        model_obj=AlwaysFailSurrogate(),
        logging_mode="all",
        on_surrogate_exception="penalty",
    )
    session = create_evaluation_session(case, store)
    result = evaluate_one(
        session,
        np.array([0.0], dtype=np.float64),
        candidate_id="fail_surrogate_penalty",
    )
    assert result.status == "FAIL"
    assert result.artifacts is not None
    assert "trial_artifact_id" in result.artifacts
    assert count_assim_trial_dirs(store) == 1


def test_surrogate_exception_policy_fail_skips_trial_write(tmp_path) -> None:
    case, store, _ = make_case_and_store(
        tmp_path,
        model_obj=AlwaysFailSurrogate(),
        logging_mode="all",
        on_surrogate_exception="fail",
    )
    session = create_evaluation_session(case, store)
    result = evaluate_one(
        session,
        np.array([0.0], dtype=np.float64),
        candidate_id="fail_surrogate_hard",
    )
    assert result.status == "FAIL"
    assert result.artifacts is None
    assert any("mode=fail" in message for message in result.messages)
    assert count_assim_trial_dirs(store) == 0


def test_observer_exception_policy_penalty_can_write_trial(tmp_path) -> None:
    case, store, _ = make_case_and_store(
        tmp_path,
        model_obj=BadGeomSurrogate(),
        logging_mode="all",
        on_observer_exception="penalty",
    )
    session = create_evaluation_session(case, store)
    result = evaluate_one(
        session,
        np.array([0.0], dtype=np.float64),
        candidate_id="fail_observer_penalty",
    )
    assert result.status == "FAIL"
    assert result.artifacts is not None
    assert "trial_artifact_id" in result.artifacts
    assert count_assim_trial_dirs(store) == 1


def test_observer_exception_policy_fail_skips_trial_write(tmp_path) -> None:
    case, store, _ = make_case_and_store(
        tmp_path,
        model_obj=BadGeomSurrogate(),
        logging_mode="all",
        on_observer_exception="fail",
    )
    session = create_evaluation_session(case, store)
    result = evaluate_one(
        session,
        np.array([0.0], dtype=np.float64),
        candidate_id="fail_observer_hard",
    )
    assert result.status == "FAIL"
    assert result.artifacts is None
    assert any("mode=fail" in message for message in result.messages)
    assert count_assim_trial_dirs(store) == 0
