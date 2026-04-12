from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from wafergeo.compare.features import ViewFeature


@dataclass(frozen=True)
class MetricComputation:
    name: str
    loss: float
    value: float
    status: str = "OK"
    details: dict[str, object] | None = None
    cd_profile: list[dict[str, float]] = field(default_factory=list)
    cd_profile_summary: dict[str, object] | None = None


@dataclass(frozen=True)
class MetricContext:
    cd_material_ids: tuple[int, ...] | None = None
    cd_gauge_axis: str | None = None
    cd_gauge_height_axis: str | None = None
    cd_gauge_center_nm: float | None = None
    cd_gauge_height_range_nm: tuple[float, float] | None = None


MetricCompute = Callable[[ViewFeature, ViewFeature, MetricContext], MetricComputation]


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    required_features: frozenset[str]
    compute: MetricCompute
    loss_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.loss_scale <= 0:
            raise ValueError(f"metric loss_scale must be > 0: {self.name}")
