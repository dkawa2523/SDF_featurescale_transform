from __future__ import annotations

from wafergeo.compare.metric_cd import compute_cd
from wafergeo.compare.metric_distance import compute_chamfer
from wafergeo.compare.metric_region import (
    compute_iou,
    compute_sdf,
    compute_sdf_band,
    compute_sdf_material,
)
from wafergeo.compare.metric_types import (
    MetricComputation,
    MetricContext,
    MetricDefinition,
)

METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    "cd": MetricDefinition("cd", frozenset({"contour"}), compute_cd, loss_scale=10.0),
    "chamfer": MetricDefinition(
        "chamfer",
        frozenset({"contour"}),
        compute_chamfer,
        loss_scale=10.0,
    ),
    "sdf": MetricDefinition("sdf", frozenset({"sdf"}), compute_sdf, loss_scale=10.0),
    "sdf_material": MetricDefinition(
        "sdf_material",
        frozenset({"sdf"}),
        compute_sdf_material,
        loss_scale=10.0,
    ),
    "sdf_band": MetricDefinition(
        "sdf_band",
        frozenset({"sdf"}),
        compute_sdf_band,
        loss_scale=10.0,
    ),
    "iou": MetricDefinition("iou", frozenset(), compute_iou, loss_scale=1.0),
}


def public_metric_names() -> set[str]:
    return set(METRIC_DEFINITIONS)


def required_features_for_metric(name: str) -> frozenset[str]:
    return METRIC_DEFINITIONS[name].required_features


__all__ = [
    "METRIC_DEFINITIONS",
    "MetricComputation",
    "MetricContext",
    "MetricDefinition",
    "public_metric_names",
    "required_features_for_metric",
]
