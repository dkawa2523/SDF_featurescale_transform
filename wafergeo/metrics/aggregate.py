from __future__ import annotations

from dataclasses import dataclass, field

from wafergeo.core.types import Obs2D, Status
from wafergeo.metrics.base import MetricResult, ObjectiveResult
from wafergeo.metrics.qa import combine_status, ensure_finite_or_fail, fail_result
from wafergeo.metrics.registry import get_metric
from wafergeo.metrics.spec import MeasurementSpecV1, MetricEntrySpec, MetricSpecV2


@dataclass(frozen=True)
class MetricContext:
    precomputed: dict[tuple[int, str], object | None] = field(default_factory=dict)


def _entry_observers(entry: MetricEntrySpec, all_observers: list[str]) -> list[str]:
    if entry.observers is None:
        return list(all_observers)
    return [name for name in entry.observers]


def build_metric_context(
    obs_by_observer: dict[str, Obs2D],
    metric_spec: MetricSpecV2,
    measurement_specs_by_ref: dict[str, MeasurementSpecV1],
) -> MetricContext:
    precomputed: dict[tuple[int, str], object | None] = {}
    all_observers = sorted(obs_by_observer.keys())

    for entry_index, entry in enumerate(metric_spec.metrics):
        metric = get_metric(entry.name)
        measurement: MeasurementSpecV1 | None = None
        if entry.measurement_ref is not None:
            measurement = measurement_specs_by_ref.get(entry.measurement_ref)
            if measurement is None:
                msg = f"measurement_ref '{entry.measurement_ref}' not found"
                for observer_name in _entry_observers(entry, all_observers):
                    precomputed[(entry_index, observer_name)] = {"precompute_error": msg}
                continue

        for observer_name in _entry_observers(entry, all_observers):
            obs = obs_by_observer.get(observer_name)
            if obs is None:
                precomputed[(entry_index, observer_name)] = {
                    "precompute_error": f"observer '{observer_name}' is missing in obs inputs"
                }
                continue

            try:
                precomputed[(entry_index, observer_name)] = metric.precompute_obs(
                    obs,
                    entry,
                    measurement,
                )
            except Exception as exc:
                precomputed[(entry_index, observer_name)] = {
                    "precompute_error": f"{type(exc).__name__}: {exc}"
                }

    return MetricContext(precomputed=precomputed)


def _observer_weight(metric_spec: MetricSpecV2, observer_name: str) -> float:
    return float((metric_spec.observer_weights or {}).get(observer_name, 1.0))


def compute_objective(
    pred_by_observer: dict[str, Obs2D],
    obs_by_observer: dict[str, Obs2D],
    metric_spec: MetricSpecV2,
    metric_ctx: MetricContext,
) -> ObjectiveResult:
    total_loss = 0.0
    by_observer_loss: dict[str, float] = {
        name: 0.0 for name in sorted(set(obs_by_observer.keys()) | set(pred_by_observer.keys()))
    }
    metric_results: list[MetricResult] = []
    messages: list[str] = []
    overall_status: Status = "OK"

    all_observers = sorted(obs_by_observer.keys())

    for entry_index, entry in enumerate(metric_spec.metrics):
        metric = get_metric(entry.name)
        for observer_name in _entry_observers(entry, all_observers):
            pred = pred_by_observer.get(observer_name)
            obs = obs_by_observer.get(observer_name)
            ctx = metric_ctx.precomputed.get((entry_index, observer_name))

            if pred is None or obs is None:
                result = fail_result(
                    metric.name,
                    metric.version,
                    fail_penalty=metric_spec.fail_penalty,
                    message=f"observer '{observer_name}' missing in pred/obs",
                )
            elif isinstance(ctx, dict) and isinstance(ctx.get("precompute_error"), str):
                result = fail_result(
                    metric.name,
                    metric.version,
                    fail_penalty=metric_spec.fail_penalty,
                    message=str(ctx["precompute_error"]),
                )
            else:
                try:
                    result = metric.compute(
                        pred,
                        obs,
                        entry,
                        ctx,
                        fail_penalty=metric_spec.fail_penalty,
                    )
                except Exception as exc:
                    result = fail_result(
                        metric.name,
                        metric.version,
                        fail_penalty=metric_spec.fail_penalty,
                        message=f"{type(exc).__name__}: {exc}",
                    )

            result = ensure_finite_or_fail(
                result,
                fail_penalty=metric_spec.fail_penalty,
                message_prefix=f"{metric.name}:{observer_name}",
            )
            meta = dict(result.meta)
            meta.update(
                {
                    "observer": observer_name,
                    "metric_weight": entry.weight,
                    "observer_weight": _observer_weight(metric_spec, observer_name),
                    "entry_index": entry_index,
                }
            )
            wrapped = MetricResult(
                name=result.name,
                version=result.version,
                loss=result.loss,
                report=dict(result.report),
                maps=dict(result.maps),
                status=result.status,
                messages=list(result.messages),
                meta=meta,
            )
            metric_results.append(wrapped)

            weighted_loss = (
                float(entry.weight)
                * _observer_weight(metric_spec, observer_name)
                * wrapped.loss
            )
            total_loss += weighted_loss
            by_observer_loss[observer_name] = (
                by_observer_loss.get(observer_name, 0.0) + weighted_loss
            )

            overall_status = combine_status(overall_status, wrapped.status)
            if wrapped.status != "OK":
                messages.append(
                    f"observer={observer_name} metric={wrapped.name} status={wrapped.status}: "
                    + " | ".join(wrapped.messages)
                )

    return ObjectiveResult(
        total_loss=float(total_loss),
        metric_results=metric_results,
        by_observer_loss=by_observer_loss,
        status=overall_status,
        messages=messages,
    )
