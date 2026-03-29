from __future__ import annotations

import numpy as np

from wafergeo.assimilation.objective import evaluate_candidate
from wafergeo.assimilation.session import EvaluationSession
from wafergeo.assimilation.types import EvalResult


def evaluate_one(
    session: EvaluationSession,
    x: np.ndarray,
    *,
    candidate_id: str | None = None,
    return_maps: bool = False,
) -> EvalResult:
    if candidate_id is None:
        candidate_id = f"cand_{session.eval_count + 1:06d}"
    return evaluate_candidate(
        session,
        x,
        candidate_id,
        return_maps=return_maps,
    )


def evaluate_batch(
    session: EvaluationSession,
    X: np.ndarray,
    *,
    candidate_id_prefix: str = "cand",
    return_maps: bool = False,
) -> list[EvalResult]:
    rows = np.asarray(X, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"X must be 2D [B,D], got ndim={rows.ndim}")

    out: list[EvalResult] = []
    for index in range(rows.shape[0]):
        candidate_id = f"{candidate_id_prefix}_{index:06d}"
        out.append(
            evaluate_candidate(
                session,
                rows[index],
                candidate_id,
                return_maps=return_maps,
            )
        )
    return out


def objective_only(session: EvaluationSession, x: np.ndarray) -> float:
    return float(evaluate_one(session, x, return_maps=False).total_loss)
