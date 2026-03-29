from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from wafergeo.assimilation.types import EvalResult
from wafergeo.core.hashing import hash_config
from wafergeo.core.meta import Meta
from wafergeo.core.types import Obs2D
from wafergeo.io.artifact_store import ArtifactStore


def _serialize_metric_results(result: EvalResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in result.metric_results:
        rows.append(
            {
                "name": item.name,
                "version": item.version,
                "loss": float(item.loss),
                "status": item.status,
                "messages": list(item.messages),
                "report": dict(item.report),
                "meta": dict(item.meta),
                "maps": dict(item.maps),
            }
        )
    return rows


def build_trial_payload(
    result: EvalResult,
    case_id: str,
    *,
    pred_obs: dict[str, Obs2D] | None = None,
    save_pred_obs: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "assim_trial/v1",
        "case_id": case_id,
        "candidate_id": result.candidate_id,
        "x": np.asarray(result.x, dtype=np.float64),
        "params": dict(result.params),
        "total_loss": float(result.total_loss),
        "per_observer": dict(result.per_observer),
        "status": result.status,
        "messages": list(result.messages),
        "timings": dict(result.timings),
        "metric_results": _serialize_metric_results(result),
        "artifacts": dict(result.artifacts or {}),
    }
    if save_pred_obs and pred_obs is not None:
        payload["pred_obs"] = dict(pred_obs)
    return payload


def write_trial_artifact(
    store: ArtifactStore,
    result: EvalResult,
    case_id: str,
    *,
    pred_obs: dict[str, Obs2D] | None = None,
    save_pred_obs: bool = False,
) -> str:
    payload = build_trial_payload(
        result,
        case_id,
        pred_obs=pred_obs,
        save_pred_obs=save_pred_obs,
    )
    config_hash = hash_config(
        {
            "schema_version": payload["schema_version"],
            "case_id": case_id,
            "candidate_id": result.candidate_id,
            "status": result.status,
        }
    )
    input_hash = hash_config(
        {
            "case_id": case_id,
            "candidate_id": result.candidate_id,
            "x": np.asarray(result.x, dtype=np.float64).tolist(),
        }
    )
    meta = Meta(
        schema_version="assim_trial/v1",
        profile_id=case_id,
        config_hash=config_hash,
        generator_version="0.1.0",
        git_commit="unknown",
        input_hash=input_hash,
        created_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        extra={"status": result.status, "candidate_id": result.candidate_id},
    )
    return store.write("assim_trial", payload, meta)
