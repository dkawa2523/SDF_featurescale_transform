from __future__ import annotations

import numpy as np

from tests.assimilation.helpers import make_case_and_store
from wafergeo.assimilation.evaluator import evaluate_batch
from wafergeo.assimilation.session import create_evaluation_session


def test_evaluate_batch_continues_when_one_candidate_fails(tmp_path) -> None:
    case, store, _ = make_case_and_store(tmp_path, oob_policy="penalty")
    session = create_evaluation_session(case, store)
    X = np.array([[0.0], [100.0], [1.0]], dtype=np.float64)
    out = evaluate_batch(session, X, candidate_id_prefix="batch")
    assert len(out) == 3
    assert out[0].status in {"OK", "WARN"}
    assert out[1].status == "FAIL"
    assert out[1].total_loss == case.failure_policy.penalty
    assert out[2].status in {"OK", "WARN"}
