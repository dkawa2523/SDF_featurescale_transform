from __future__ import annotations

from tests.assimilation.helpers import make_case_and_store
from wafergeo.assimilation.session import create_evaluation_session


def test_create_evaluation_session_builds_obs_model_and_precompute(tmp_path) -> None:
    case, store, sem_obs_id = make_case_and_store(tmp_path)
    session = create_evaluation_session(case, store)

    assert session.case.case_id == "assim_case_test"
    assert "topdown" in session.obs_sem
    assert session.obs_sem["topdown"].meta.schema_version in {"sem_obs/v1", "observer/v2"}
    assert session.surrogate.name == "shift_surrogate"
    assert "topdown" in session.observers
    assert store.exists(sem_obs_id)
    assert (0, "topdown") in session.metric_ctx.precomputed
