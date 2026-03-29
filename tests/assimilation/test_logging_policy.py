from __future__ import annotations

import numpy as np

from tests.assimilation.helpers import count_assim_trial_dirs, make_case_and_store
from wafergeo.assimilation.evaluator import evaluate_one
from wafergeo.assimilation.session import create_evaluation_session


def _run_many(session, xs: list[float]) -> None:
    for value in xs:
        evaluate_one(session, np.array([value], dtype=np.float64))


def test_logging_policy_none(tmp_path) -> None:
    case, store, _ = make_case_and_store(tmp_path, logging_mode="none")
    session = create_evaluation_session(case, store)
    _run_many(session, [0.0, 1.0, 2.0])
    assert count_assim_trial_dirs(store) == 0


def test_logging_policy_all(tmp_path) -> None:
    case, store, _ = make_case_and_store(tmp_path, logging_mode="all")
    session = create_evaluation_session(case, store)
    _run_many(session, [0.0, 1.0, 2.0, 3.0])
    assert count_assim_trial_dirs(store) == 4


def test_logging_policy_periodic(tmp_path) -> None:
    case, store, _ = make_case_and_store(tmp_path, logging_mode="periodic", logging_period=2)
    session = create_evaluation_session(case, store)
    _run_many(session, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert count_assim_trial_dirs(store) == 2


def test_logging_policy_best_only(tmp_path) -> None:
    case, store, _ = make_case_and_store(tmp_path, logging_mode="best_only")
    session = create_evaluation_session(case, store)
    _run_many(session, [3.0, 2.0, 1.0, 0.0, 1.0])
    saved = count_assim_trial_dirs(store)
    assert saved >= 1
    assert saved <= 5
