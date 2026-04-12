"""Simple feature transform and shape comparison facade."""

from wafergeo.compare.batch_runner import run_batch_compare_from_config
from wafergeo.compare.loader import (
    CONTOUR_LOADERS,
    LABEL_LOADERS,
    ContourData,
    ContourItem,
    is_contour_input_kind,
    is_label_input_kind,
    load_contour_json,
    load_simulation_label,
)
from wafergeo.compare.metric_defs import METRIC_DEFINITIONS, MetricContext, MetricDefinition
from wafergeo.compare.runner import (
    run_compare_from_config,
    run_transform_from_config,
)
from wafergeo.compare.schema import (
    BatchCompareSpec,
    CdGaugeSpec,
    CompareSpec,
    TransformSpec,
    load_batch_compare_spec_yaml,
    load_compare_spec_yaml,
    load_transform_spec_yaml,
)

__all__ = [
    "ContourData",
    "ContourItem",
    "TransformSpec",
    "CompareSpec",
    "BatchCompareSpec",
    "CdGaugeSpec",
    "MetricDefinition",
    "MetricContext",
    "LABEL_LOADERS",
    "CONTOUR_LOADERS",
    "METRIC_DEFINITIONS",
    "is_label_input_kind",
    "is_contour_input_kind",
    "load_contour_json",
    "load_simulation_label",
    "load_transform_spec_yaml",
    "load_compare_spec_yaml",
    "load_batch_compare_spec_yaml",
    "run_transform_from_config",
    "run_compare_from_config",
    "run_batch_compare_from_config",
]
