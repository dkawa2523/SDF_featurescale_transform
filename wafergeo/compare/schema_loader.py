from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from wafergeo.compare.feature_taxonomy import resolve_transform_eval_code_name
from wafergeo.compare.metric_defs import METRIC_DEFINITIONS
from wafergeo.compare.schema_types import (
    COMPARE_FEATURE_NAMES,
    TRANSFORM_FEATURE_NAMES,
    AxisName,
    BatchCompareSpec,
    BatchTransformSpec,
    CdGaugeSpec,
    CompareEvalMetricSetSpec,
    CompareEvalSpec,
    CompareSpec,
    FeatureName,
    FeatureSpec,
    MetricName,
    MetricSpec,
    OutputSpec,
    ProcessSpec,
    SimulationInputSpec,
    SimulationKind,
    TargetInputSpec,
    TargetKind,
    TargetShapeName,
    TransformEvalFeatureSpec,
    TransformEvalMethodName,
    TransformEvalSpec,
    TransformSpec,
    ViewSpec,
)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("simple run YAML root must be a mapping")
    return {str(k): v for k, v in raw.items()}


def _require_task(raw: dict[str, Any], expected: str) -> None:
    actual = str(raw.get("task", ""))
    if actual != expected:
        raise ValueError(f"task must be '{expected}', got '{actual or '<missing>'}'")


def _mapping(
    parent: dict[str, Any],
    key: str,
    *,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if key not in parent:
        if default is None:
            raise ValueError(f"missing required key: {key}")
        return default
    value = parent[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return {str(k): v for k, v in value.items()}


def _list(parent: dict[str, Any], key: str, default: list[Any]) -> list[Any]:
    value = parent.get(key, default)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return list(value)


def _bool(parent: dict[str, Any], key: str, default: bool) -> bool:
    value = parent.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _simulation(raw: dict[str, Any]) -> SimulationInputSpec:
    void_raw = raw.get("void_id")
    return SimulationInputSpec(
        kind=cast(SimulationKind, str(raw.get("kind", ""))),
        path=str(raw.get("path", "")),
        void_id=None if void_raw is None else int(cast(int | str, void_raw)),
    )


def _optional_simulation(parent: dict[str, Any], key: str) -> SimulationInputSpec | None:
    if key not in parent:
        return None
    return _simulation(_mapping(parent, key))


def _target(raw: dict[str, Any]) -> TargetInputSpec:
    void_raw = raw.get("void_id")
    return TargetInputSpec(
        kind=cast(TargetKind, str(raw.get("kind", "contour_json"))),
        path=str(raw.get("path", "")),
        units=str(raw.get("units", "nm")),
        void_id=None if void_raw is None else int(cast(int | str, void_raw)),
    )


def _view(raw: dict[str, Any]) -> ViewSpec:
    axes = tuple(str(v) for v in _list(raw, "axes", ["x", "y"]))
    if len(axes) != 2:
        raise ValueError("view.axes must contain exactly two axes")
    return ViewSpec(
        kind=str(raw.get("kind", "topview")),
        axes=cast(tuple[AxisName, AxisName], axes),
        depth_axis=cast(AxisName, str(raw.get("depth_axis", "z"))),
    )


def _features(raw: dict[str, Any]) -> FeatureSpec:
    return FeatureSpec(
        use=tuple(cast(FeatureName, str(v)) for v in _list(raw, "use", ["sdf", "contour"]))
    )


def _require_allowed_features(
    features: FeatureSpec,
    *,
    task: str,
    allowed: set[str],
) -> None:
    unsupported = sorted(set(features.use).difference(allowed))
    if unsupported:
        raise ValueError(f"{task} does not support features.use values: {unsupported}")


def _metrics(raw: dict[str, Any]) -> MetricSpec:
    weights_raw = _mapping(raw, "weights", default={})
    cd_raw = _mapping(raw, "cd", default={})
    cd_material_ids = None
    if "material_ids" in cd_raw:
        cd_material_ids = tuple(int(v) for v in _list(cd_raw, "material_ids", []))
    cd_gauge = None
    if "gauge" in cd_raw:
        gauge_raw = _mapping(cd_raw, "gauge")
        height_range = None
        if "height_range" in gauge_raw:
            values = _list(gauge_raw, "height_range", [])
            if len(values) != 2:
                raise ValueError("metrics.cd.gauge.height_range must contain two values")
            height_range = (float(values[0]), float(values[1]))
        cd_gauge = CdGaugeSpec(
            axis=(
                None
                if gauge_raw.get("axis") is None
                else cast(AxisName, str(gauge_raw.get("axis")))
            ),
            height_axis=cast(AxisName, str(gauge_raw.get("height_axis", "z"))),
            center=None if gauge_raw.get("center") is None else float(gauge_raw["center"]),
            height_range=height_range,
        )
    return MetricSpec(
        use=tuple(
            cast(MetricName, str(v))
            for v in _list(raw, "use", ["cd", "sdf", "iou"])
        ),
        weights={str(k): float(v) for k, v in weights_raw.items()},
        cd_material_ids=cd_material_ids,
        cd_gauge=cd_gauge,
    )


def _require_metric_feature_dependencies(features: FeatureSpec, metrics: MetricSpec) -> None:
    selected_features = set(features.use)
    missing: dict[str, list[str]] = {}
    for metric in metrics.use:
        for feature in METRIC_DEFINITIONS[metric].required_features:
            if feature not in selected_features:
                missing.setdefault(feature, []).append(metric)
    if missing:
        parts = [
            f"feature '{feature}' is required by metrics {metric_names}"
            for feature, metric_names in sorted(missing.items())
        ]
        raise ValueError("metrics.use is inconsistent with features.use: " + "; ".join(parts))


def _require_cd_gauge_compatible(view: ViewSpec, metrics: MetricSpec) -> None:
    if "cd" not in metrics.use or metrics.cd_gauge is None:
        return
    gauge = metrics.cd_gauge
    if gauge.height_axis not in view.axes:
        raise ValueError("metrics.cd.gauge.height_axis must be included in view.axes")
    if gauge.axis is not None and gauge.axis not in view.axes:
        raise ValueError("metrics.cd.gauge.axis must be included in view.axes")
    if gauge.axis is not None and gauge.axis == gauge.height_axis:
        raise ValueError("metrics.cd.gauge.axis and height_axis must be different")


def _output(raw: dict[str, Any]) -> OutputSpec:
    return OutputSpec(
        dir=str(raw.get("dir", "")),
        difference_image=_bool(raw, "difference_image", True),
        difference_images=_bool(raw, "difference_images", True),
        ranking=_bool(raw, "ranking", True),
    )


def _process(raw: dict[str, Any]) -> ProcessSpec:
    return ProcessSpec(enabled=_bool(raw, "enabled", False))


def load_transform_spec_yaml(path: str | Path) -> TransformSpec:
    raw = _read_yaml(path)
    _require_task(raw, "transform")
    input_raw = _mapping(raw, "input")
    features = _features(_mapping(raw, "features", default={"use": ["sdf_raw"]}))
    _require_allowed_features(features, task="transform", allowed=TRANSFORM_FEATURE_NAMES)
    return TransformSpec(
        task="transform",
        simulation=_simulation(_mapping(input_raw, "simulation")),
        view=_view(_mapping(raw, "view", default={})),
        features=features,
        output=_output(_mapping(raw, "output")),
        reference=_optional_simulation(input_raw, "reference"),
        process=_process(_mapping(raw, "process", default={})),
    )


def load_batch_transform_spec_yaml(path: str | Path) -> BatchTransformSpec:
    raw = _read_yaml(path)
    _require_task(raw, "batch-transform")
    input_raw = _mapping(raw, "input")
    features = _features(_mapping(raw, "features", default={"use": ["sdf_raw"]}))
    _require_allowed_features(features, task="batch-transform", allowed=TRANSFORM_FEATURE_NAMES)
    return BatchTransformSpec(
        task="batch-transform",
        index=str(input_raw.get("index", "")),
        view=_view(_mapping(raw, "view", default={})),
        features=features,
        output=_output(_mapping(raw, "output")),
        process=_process(_mapping(raw, "process", default={})),
    )


def _transform_eval_features(raw: dict[str, Any]) -> tuple[TransformEvalFeatureSpec, ...]:
    eval_raw = _mapping(raw, "eval")
    features_raw = eval_raw.get("features")
    if not isinstance(features_raw, list) or not features_raw:
        raise ValueError("eval.features must be a non-empty list")
    features: list[TransformEvalFeatureSpec] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(features_raw):
        if not isinstance(value, dict):
            raise ValueError(f"eval.features[{index}] must be a mapping")
        item = {str(k): v for k, v in value.items()}
        target_shape = cast(TargetShapeName, str(item.get("target_shape", "")))
        method = cast(TransformEvalMethodName, str(item.get("method", "")))
        code_name = resolve_transform_eval_code_name(target_shape, method)
        key = (target_shape, method)
        if key in seen:
            raise ValueError(
                "duplicate transform-eval feature: "
                f"target_shape={target_shape!r}, method={method!r}"
            )
        seen.add(key)
        features.append(
            TransformEvalFeatureSpec(
                target_shape=target_shape,
                method=method,
                code_name=cast(FeatureName, code_name),
            )
        )
    return tuple(features)


def load_transform_eval_spec_yaml(path: str | Path) -> TransformEvalSpec:
    raw = _read_yaml(path)
    _require_task(raw, "transform-eval")
    input_raw = _mapping(raw, "input")
    return TransformEvalSpec(
        task="transform-eval",
        index=str(input_raw.get("index", "")),
        view=_view(_mapping(raw, "view", default={})),
        features=_transform_eval_features(raw),
        output=_output(_mapping(raw, "output")),
        process=_process(_mapping(raw, "process", default={})),
    )


def load_compare_spec_yaml(path: str | Path) -> CompareSpec:
    raw = _read_yaml(path)
    _require_task(raw, "compare")
    input_raw = _mapping(raw, "input")
    features = _features(_mapping(raw, "features", default={"use": ["sdf", "contour"]}))
    _require_allowed_features(features, task="compare", allowed=COMPARE_FEATURE_NAMES)
    metrics = _metrics(
        _mapping(raw, "metrics", default={"use": ["cd", "sdf", "iou"]})
    )
    _require_metric_feature_dependencies(features, metrics)
    view = _view(_mapping(raw, "view", default={}))
    _require_cd_gauge_compatible(view, metrics)
    return CompareSpec(
        task="compare",
        simulation=_simulation(_mapping(input_raw, "simulation")),
        target=_target(_mapping(input_raw, "target")),
        view=view,
        features=features,
        metrics=metrics,
        output=_output(_mapping(raw, "output")),
    )


def load_batch_compare_spec_yaml(path: str | Path) -> BatchCompareSpec:
    raw = _read_yaml(path)
    _require_task(raw, "batch-compare")
    input_raw = _mapping(raw, "input")
    features = _features(_mapping(raw, "features", default={"use": ["sdf", "contour"]}))
    _require_allowed_features(features, task="batch-compare", allowed=COMPARE_FEATURE_NAMES)
    metrics = _metrics(
        _mapping(raw, "metrics", default={"use": ["cd", "sdf", "iou"]})
    )
    _require_metric_feature_dependencies(features, metrics)
    view = _view(_mapping(raw, "view", default={}))
    _require_cd_gauge_compatible(view, metrics)
    return BatchCompareSpec(
        task="batch-compare",
        index=str(input_raw.get("index", "")),
        view=view,
        features=features,
        metrics=metrics,
        output=_output(_mapping(raw, "output")),
    )


def _compare_eval_metric_sets(
    raw: dict[str, Any],
    *,
    view: ViewSpec,
) -> dict[str, CompareEvalMetricSetSpec]:
    eval_raw = _mapping(raw, "eval")
    metric_sets_raw = _mapping(eval_raw, "metric_sets")
    metric_sets: dict[str, CompareEvalMetricSetSpec] = {}
    for name, value in metric_sets_raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"eval.metric_sets.{name} must be a mapping")
        metric_set_raw = {str(k): v for k, v in value.items()}
        features = _features(
            _mapping(metric_set_raw, "features", default={"use": ["sdf", "contour"]})
        )
        _require_allowed_features(
            features,
            task="compare-eval",
            allowed=COMPARE_FEATURE_NAMES,
        )
        metrics = _metrics(
            _mapping(metric_set_raw, "metrics", default={"use": ["cd", "sdf", "iou"]})
        )
        _require_metric_feature_dependencies(features, metrics)
        _require_cd_gauge_compatible(view, metrics)
        metric_sets[str(name)] = CompareEvalMetricSetSpec(features=features, metrics=metrics)
    return metric_sets


def load_compare_eval_spec_yaml(path: str | Path) -> CompareEvalSpec:
    raw = _read_yaml(path)
    _require_task(raw, "compare-eval")
    input_raw = _mapping(raw, "input")
    view = _view(_mapping(raw, "view", default={}))
    return CompareEvalSpec(
        task="compare-eval",
        index=str(input_raw.get("index", "")),
        view=view,
        metric_sets=_compare_eval_metric_sets(raw, view=view),
        output=_output(_mapping(raw, "output")),
    )
