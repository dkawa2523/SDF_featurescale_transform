from __future__ import annotations

import numpy as np

from tests.assimilation.helpers import make_case_and_store
from wafergeo.assimilation.evaluator import evaluate_one
from wafergeo.assimilation.session import create_evaluation_session


def test_evaluate_one_pipeline_with_dummy_surrogate(tmp_path) -> None:
    case, store, _ = make_case_and_store(tmp_path, logging_mode="none")
    session = create_evaluation_session(case, store)

    baseline = evaluate_one(session, np.array([0.0], dtype=np.float64), candidate_id="baseline")
    shifted = evaluate_one(session, np.array([1.0], dtype=np.float64), candidate_id="shifted")

    assert baseline.status in {"OK", "WARN"}
    assert baseline.total_loss < 1e-6
    assert shifted.total_loss > baseline.total_loss
