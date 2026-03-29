from __future__ import annotations

from dataclasses import dataclass, field

from wafergeo.assimilation.registry import get_model_loader
from wafergeo.assimilation.types import CaseSpec, SurrogateModelProtocol
from wafergeo.core.types import Obs2D
from wafergeo.io.artifact_store import ArtifactStore
from wafergeo.metrics.aggregate import MetricContext, build_metric_context
from wafergeo.observe.base import ObserverProtocol
from wafergeo.observe.factory import create_observer
from wafergeo.sem.artifact import read_sem_obs_artifact


@dataclass
class EvaluationSession:
    case: CaseSpec
    store: ArtifactStore
    surrogate: SurrogateModelProtocol
    observers: dict[str, ObserverProtocol]
    obs_sem: dict[str, Obs2D]
    metric_ctx: MetricContext
    best_loss: float = float("inf")
    best_trial_artifact_id: str | None = None
    eval_count: int = 0
    caches: dict[str, object] = field(default_factory=dict)


def create_evaluation_session(case: CaseSpec, store: ArtifactStore) -> EvaluationSession:
    obs_sem: dict[str, Obs2D] = {}
    for observer_name, artifact_id in case.sem_obs_ids.items():
        obs, _payload = read_sem_obs_artifact(store, artifact_id)
        obs_sem[observer_name] = obs

    observers: dict[str, ObserverProtocol] = {}
    for observer_name, spec in case.observer_specs.items():
        observer = create_observer(spec.kind)
        observers[observer_name] = observer

    loader = get_model_loader(case.model_package.loader_key)
    surrogate = loader.load(case.model_package.model_ref, store)

    metric_ctx = build_metric_context(
        obs_sem,
        case.metric_spec,
        case.measurement_specs_by_ref,
    )

    return EvaluationSession(
        case=case,
        store=store,
        surrogate=surrogate,
        observers=observers,
        obs_sem=obs_sem,
        metric_ctx=metric_ctx,
    )
