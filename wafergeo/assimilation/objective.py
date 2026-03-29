from __future__ import annotations

from time import perf_counter

import numpy as np

from wafergeo.assimilation.artifacts import write_trial_artifact
from wafergeo.assimilation.policies import LoggingPolicy
from wafergeo.assimilation.session import EvaluationSession
from wafergeo.assimilation.types import EvalResult
from wafergeo.core.types import Obs2D, Status
from wafergeo.metrics.aggregate import compute_objective
from wafergeo.metrics.base import MetricResult
from wafergeo.metrics.qa import combine_status


def _fail_eval_result(
    *,
    candidate_id: str,
    x: np.ndarray,
    penalty: float,
    message: str,
    params: dict[str, object] | None = None,
    status: Status = "FAIL",
    timings: dict[str, float] | None = None,
) -> EvalResult:
    return EvalResult(
        candidate_id=candidate_id,
        x=np.asarray(x, dtype=np.float64),
        params=dict(params or {}),
        total_loss=float(penalty),
        per_observer={},
        metric_results=[],
        status=status,
        messages=[message],
        timings=dict(timings or {}),
        artifacts=None,
    )


def _validate_strict_sim_grid(session: EvaluationSession, pred_obs: dict[str, Obs2D]) -> list[str]:
    messages: list[str] = []
    for name, obs_pred in pred_obs.items():
        obs_ref = session.obs_sem.get(name)
        if obs_ref is None:
            messages.append(f"observer='{name}' missing in sem obs")
            continue

        ref_grid = obs_ref.grid2d
        pred_grid = obs_pred.grid2d
        if (
            pred_grid.spacing != ref_grid.spacing
            or pred_grid.origin != ref_grid.origin
            or pred_grid.axis_order != ref_grid.axis_order
            or pred_grid.units != ref_grid.units
            or pred_grid.sample_location != ref_grid.sample_location
        ):
            messages.append(f"observer='{name}' grid2d mismatch under strict_sim_grid")
        if obs_pred.mask.shape != obs_ref.mask.shape:
            messages.append(f"observer='{name}' mask shape mismatch under strict_sim_grid")
        if obs_pred.tsdf.shape != obs_ref.tsdf.shape:
            messages.append(f"observer='{name}' tsdf shape mismatch under strict_sim_grid")
    return messages


def _filtered_metric_results(
    metric_results: list[MetricResult],
    *,
    return_maps: bool,
) -> list[MetricResult]:
    if return_maps:
        return metric_results
    out: list[MetricResult] = []
    for row in metric_results:
        out.append(
            MetricResult(
                name=row.name,
                version=row.version,
                loss=row.loss,
                report=dict(row.report),
                maps={},
                status=row.status,
                messages=list(row.messages),
                meta=dict(row.meta),
            )
        )
    return out


def _should_save(
    policy: LoggingPolicy,
    session: EvaluationSession,
    result: EvalResult,
) -> bool:
    if policy.mode == "none":
        return False
    if policy.mode == "all":
        return True
    if policy.mode == "periodic":
        return session.eval_count % policy.period == 0
    if policy.mode == "best_only":
        return result.total_loss < session.best_loss
    return False


def _with_trial_artifact(
    session: EvaluationSession,
    result: EvalResult,
) -> EvalResult:
    trial_artifact_id = write_trial_artifact(
        session.store,
        result,
        session.case.case_id,
        pred_obs=None,
        save_pred_obs=False,
    )
    artifacts = dict(result.artifacts or {})
    artifacts["trial_artifact_id"] = trial_artifact_id
    return EvalResult(
        candidate_id=result.candidate_id,
        x=result.x,
        params=result.params,
        total_loss=result.total_loss,
        per_observer=result.per_observer,
        metric_results=result.metric_results,
        status=result.status,
        messages=result.messages,
        timings=result.timings,
        artifacts=artifacts,
    )


def _apply_exception_policy(
    *,
    session: EvaluationSession,
    policy_mode: str,
    result: EvalResult,
) -> EvalResult:
    if policy_mode == "penalty" and _should_save(session.case.logging_policy, session, result):
        return _with_trial_artifact(session, result)
    if policy_mode == "fail":
        fail_message = "failure policy mode=fail: no trial artifact written"
        return EvalResult(
            candidate_id=result.candidate_id,
            x=result.x,
            params=result.params,
            total_loss=result.total_loss,
            per_observer=result.per_observer,
            metric_results=result.metric_results,
            status=result.status,
            messages=list(result.messages) + [fail_message],
            timings=result.timings,
            artifacts=result.artifacts,
        )
    return result


def evaluate_candidate(
    session: EvaluationSession,
    x: np.ndarray,
    candidate_id: str,
    *,
    return_maps: bool = False,
) -> EvalResult:
    penalty = float(session.case.failure_policy.penalty)
    x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
    session.eval_count += 1

    time_decode = 0.0
    time_predict = 0.0
    time_observe = 0.0
    time_metrics = 0.0
    t_all = perf_counter()

    t0 = perf_counter()
    try:
        params, decode_warnings, oob = session.case.param_spec.decode(
            x_vec,
            session.case.failure_policy.out_of_bounds,
        )
    except Exception as exc:
        return _fail_eval_result(
            candidate_id=candidate_id,
            x=x_vec,
            penalty=penalty,
            message=f"param decode failed: {type(exc).__name__}: {exc}",
            timings={"total": perf_counter() - t_all},
        )
    time_decode = perf_counter() - t0

    if oob and session.case.failure_policy.out_of_bounds in {"penalty", "fail"}:
        return _fail_eval_result(
            candidate_id=candidate_id,
            x=x_vec,
            penalty=penalty,
            message=(
                "out-of-bounds parameter detected under "
                f"policy={session.case.failure_policy.out_of_bounds}: "
                + " | ".join(decode_warnings)
            ),
            params=params,
            timings={"decode": time_decode, "total": perf_counter() - t_all},
        )

    status: Status = "WARN" if decode_warnings else "OK"
    messages = list(decode_warnings)

    t0 = perf_counter()
    try:
        geom = session.surrogate.predict(params)
    except Exception as exc:
        fail_result = _fail_eval_result(
            candidate_id=candidate_id,
            x=x_vec,
            penalty=penalty,
            message=(
                "surrogate.predict failed "
                f"(policy={session.case.failure_policy.on_surrogate_exception}): "
                f"{type(exc).__name__}: {exc}"
            ),
            params=params,
            timings={
                "decode": time_decode,
                "predict": perf_counter() - t0,
                "total": perf_counter() - t_all,
            },
        )
        return _apply_exception_policy(
            session=session,
            policy_mode=session.case.failure_policy.on_surrogate_exception,
            result=fail_result,
        )
    time_predict = perf_counter() - t0

    pred_obs: dict[str, Obs2D] = {}
    t0 = perf_counter()
    for observer_name, observer in session.observers.items():
        spec = session.case.observer_specs[observer_name]
        try:
            pred_obs[observer_name] = observer.observe(geom, spec)
        except Exception as exc:
            fail_result = _fail_eval_result(
                candidate_id=candidate_id,
                x=x_vec,
                penalty=penalty,
                message=(
                    f"observer='{observer_name}' failed "
                    f"(policy={session.case.failure_policy.on_observer_exception}): "
                    f"{type(exc).__name__}: {exc}"
                ),
                params=params,
                timings={
                    "decode": time_decode,
                    "predict": time_predict,
                    "observe": perf_counter() - t0,
                    "total": perf_counter() - t_all,
                },
            )
            return _apply_exception_policy(
                session=session,
                policy_mode=session.case.failure_policy.on_observer_exception,
                result=fail_result,
            )
    time_observe = perf_counter() - t0

    transform_mode = session.case.transform_policy.mode
    if transform_mode == "strict_sim_grid":
        strict_errors = _validate_strict_sim_grid(session, pred_obs)
        if strict_errors:
            return _fail_eval_result(
                candidate_id=candidate_id,
                x=x_vec,
                penalty=penalty,
                message=" | ".join(strict_errors),
                params=params,
                timings={
                    "decode": time_decode,
                    "predict": time_predict,
                    "observe": time_observe,
                    "total": perf_counter() - t_all,
                },
            )
    else:
        status = combine_status(status, "WARN")
        messages.append(
            f"transform_policy.mode={transform_mode} is not implemented in Phase 7; no-op applied"
        )

    t0 = perf_counter()
    try:
        objective = compute_objective(
            pred_obs,
            session.obs_sem,
            session.case.metric_spec,
            session.metric_ctx,
        )
    except Exception as exc:
        return _fail_eval_result(
            candidate_id=candidate_id,
            x=x_vec,
            penalty=penalty,
            message=f"metrics.compute_objective failed: {type(exc).__name__}: {exc}",
            params=params,
            timings={
                "decode": time_decode,
                "predict": time_predict,
                "observe": time_observe,
                "metrics": perf_counter() - t0,
                "total": perf_counter() - t_all,
            },
        )
    time_metrics = perf_counter() - t0

    status = combine_status(status, objective.status)
    messages.extend(objective.messages)
    filtered_metrics = _filtered_metric_results(
        objective.metric_results,
        return_maps=return_maps,
    )
    result = EvalResult(
        candidate_id=candidate_id,
        x=x_vec,
        params=params,
        total_loss=float(objective.total_loss),
        per_observer=dict(objective.by_observer_loss),
        metric_results=filtered_metrics,
        status=status,
        messages=messages,
        timings={
            "decode": time_decode,
            "predict": time_predict,
            "observe": time_observe,
            "metrics": time_metrics,
            "total": perf_counter() - t_all,
        },
        artifacts=None,
    )

    if _should_save(session.case.logging_policy, session, result):
        trial_artifact_id = write_trial_artifact(
            session.store,
            result,
            session.case.case_id,
            pred_obs=pred_obs,
            save_pred_obs=session.case.logging_policy.save_pred_obs,
        )
        artifacts = dict(result.artifacts or {})
        artifacts["trial_artifact_id"] = trial_artifact_id
        result = EvalResult(
            candidate_id=result.candidate_id,
            x=result.x,
            params=result.params,
            total_loss=result.total_loss,
            per_observer=result.per_observer,
            metric_results=result.metric_results,
            status=result.status,
            messages=result.messages,
            timings=result.timings,
            artifacts=artifacts,
        )
        if result.total_loss < session.best_loss:
            session.best_loss = result.total_loss
            session.best_trial_artifact_id = trial_artifact_id
    elif result.total_loss < session.best_loss:
        session.best_loss = result.total_loss

    return result
