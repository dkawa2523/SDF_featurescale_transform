from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from wafergeo.compare.metric_defs import public_metric_names

SimulationKind = Literal["vti_label", "npz_label"]
TargetKind = Literal["contour_json", "vti_label", "npz_label"]
FeatureName = str
MetricName = str
AxisName = Literal["x", "y", "z"]

FEATURE_NAMES = {"sdf", "sdf3d", "sdf_views", "mesh", "contour", "slice"}
METRIC_NAMES = public_metric_names()
AXIS_NAMES = {"x", "y", "z"}


@dataclass(frozen=True)
class SimulationInputSpec:
    kind: SimulationKind
    path: str
    void_id: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"vti_label", "npz_label"}:
            raise ValueError(f"unsupported simulation kind: {self.kind}")
        if not self.path:
            raise ValueError("simulation.path must be non-empty")


@dataclass(frozen=True)
class TargetInputSpec:
    kind: TargetKind
    path: str
    units: str = "nm"
    void_id: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"contour_json", "vti_label", "npz_label"}:
            raise ValueError(f"unsupported target kind: {self.kind}")
        if not self.path:
            raise ValueError("target.path must be non-empty")
        if self.kind == "contour_json" and not self.units:
            raise ValueError("target.units must be non-empty")


@dataclass(frozen=True)
class ViewSpec:
    kind: str = "topview"
    axes: tuple[AxisName, AxisName] = ("x", "y")
    depth_axis: AxisName = "z"

    def __post_init__(self) -> None:
        if self.kind != "topview":
            raise ValueError("only view.kind='topview' is supported in simple compare v1")
        for axis in (*self.axes, self.depth_axis):
            if axis not in AXIS_NAMES:
                raise ValueError(f"unsupported view axis: {axis}")
        if len(set(self.axes)) != 2:
            raise ValueError("view.axes must contain two distinct axes")
        if self.depth_axis in self.axes:
            raise ValueError("view.depth_axis must not be included in view.axes")
        if set((*self.axes, self.depth_axis)) != {"x", "y", "z"}:
            raise ValueError("view.axes + view.depth_axis must cover x/y/z")


@dataclass(frozen=True)
class FeatureSpec:
    use: tuple[FeatureName, ...] = ("sdf", "contour")

    def __post_init__(self) -> None:
        if not self.use:
            raise ValueError("features.use must be non-empty")
        unknown = sorted(set(self.use).difference(FEATURE_NAMES))
        if unknown:
            raise ValueError(f"unsupported features.use values: {unknown}")


@dataclass(frozen=True)
class CdGaugeSpec:
    axis: AxisName | None = None
    height_axis: AxisName = "z"
    center: float | None = None
    height_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.axis is not None and self.axis not in AXIS_NAMES:
            raise ValueError(f"unsupported metrics.cd.gauge.axis: {self.axis}")
        if self.height_axis not in AXIS_NAMES:
            raise ValueError(f"unsupported metrics.cd.gauge.height_axis: {self.height_axis}")
        if self.axis is not None and self.axis == self.height_axis:
            raise ValueError("metrics.cd.gauge.axis and height_axis must be different")
        if self.height_range is not None:
            start, stop = self.height_range
            if stop < start:
                raise ValueError("metrics.cd.gauge.height_range must be [min, max]")


@dataclass(frozen=True)
class MetricSpec:
    use: tuple[MetricName, ...] = ("cd", "sdf", "iou")
    weights: dict[str, float] = field(default_factory=dict)
    cd_material_ids: tuple[int, ...] | None = None
    cd_gauge: CdGaugeSpec | None = None

    def __post_init__(self) -> None:
        if not self.use:
            raise ValueError("metrics.use must be non-empty")
        unknown = sorted(set(self.use).difference(METRIC_NAMES))
        if unknown:
            raise ValueError(f"unsupported metrics.use values: {unknown}")
        for name, value in self.weights.items():
            if name not in METRIC_NAMES:
                raise ValueError(f"unsupported metrics.weights key: {name}")
            if value < 0:
                raise ValueError(f"metrics.weights.{name} must be >= 0")
        if self.cd_material_ids is not None and not self.cd_material_ids:
            raise ValueError("metrics.cd.material_ids must be non-empty when specified")

    def weight_for(self, name: str) -> float:
        return float(self.weights.get(name, 1.0))


@dataclass(frozen=True)
class OutputSpec:
    dir: str
    difference_image: bool = True
    difference_images: bool = True
    ranking: bool = True

    def __post_init__(self) -> None:
        if not self.dir:
            raise ValueError("output.dir must be non-empty")


@dataclass(frozen=True)
class TransformSpec:
    task: Literal["transform"]
    simulation: SimulationInputSpec
    view: ViewSpec
    features: FeatureSpec
    output: OutputSpec


@dataclass(frozen=True)
class CompareSpec:
    task: Literal["compare"]
    simulation: SimulationInputSpec
    target: TargetInputSpec
    view: ViewSpec
    features: FeatureSpec
    metrics: MetricSpec
    output: OutputSpec


@dataclass(frozen=True)
class BatchCompareSpec:
    task: Literal["batch-compare"]
    index: str
    view: ViewSpec
    features: FeatureSpec
    metrics: MetricSpec
    output: OutputSpec

    def __post_init__(self) -> None:
        if not self.index:
            raise ValueError("input.index must be non-empty")
