"""Assimilation layer: external-optimizer objective evaluator."""

from wafergeo.assimilation.artifacts import build_trial_payload, write_trial_artifact
from wafergeo.assimilation.evaluator import evaluate_batch, evaluate_one, objective_only
from wafergeo.assimilation.policies import FailurePolicy, LoggingPolicy, TransformPolicy
from wafergeo.assimilation.registry import (
    get_model_loader,
    list_model_loaders,
    register_default_model_loaders,
    register_model_loader,
)
from wafergeo.assimilation.session import EvaluationSession, create_evaluation_session
from wafergeo.assimilation.types import (
    CaseSpec,
    EvalResult,
    ModelLoaderProtocol,
    ModelPackageSpec,
    ParamAxis,
    ParamSpec,
    SurrogateModelProtocol,
)

__all__ = [
    "SurrogateModelProtocol",
    "ModelLoaderProtocol",
    "ParamAxis",
    "ParamSpec",
    "ModelPackageSpec",
    "CaseSpec",
    "EvalResult",
    "FailurePolicy",
    "TransformPolicy",
    "LoggingPolicy",
    "register_model_loader",
    "get_model_loader",
    "list_model_loaders",
    "register_default_model_loaders",
    "EvaluationSession",
    "create_evaluation_session",
    "evaluate_one",
    "evaluate_batch",
    "objective_only",
    "build_trial_payload",
    "write_trial_artifact",
]
