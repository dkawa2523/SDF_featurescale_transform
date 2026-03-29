"""Metrics layer: Obs2D vs Obs2D comparison."""

from wafergeo.metrics.aggregate import MetricContext, build_metric_context, compute_objective
from wafergeo.metrics.base import MetricProtocol, MetricResult, ObjectiveResult
from wafergeo.metrics.registry import get_metric, list_metrics, register_metric
from wafergeo.metrics.spec import (
    MeasurementSpecV1,
    MetricEntrySpec,
    MetricSpecV2,
    load_measurement_spec_yaml,
    load_metric_spec_yaml,
    measurement_spec_hash,
    metric_spec_hash,
)

__all__ = [
    "MetricProtocol",
    "MetricResult",
    "ObjectiveResult",
    "MetricContext",
    "MetricSpecV2",
    "MetricEntrySpec",
    "MeasurementSpecV1",
    "load_metric_spec_yaml",
    "load_measurement_spec_yaml",
    "metric_spec_hash",
    "measurement_spec_hash",
    "register_metric",
    "get_metric",
    "list_metrics",
    "build_metric_context",
    "compute_objective",
]
