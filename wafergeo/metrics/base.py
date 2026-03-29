from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from wafergeo.core.types import Obs2D, Status
from wafergeo.metrics.spec import MeasurementSpecV1, MetricEntrySpec


@dataclass(frozen=True)
class MetricResult:
    name: str
    version: str
    loss: float
    report: dict[str, object] = field(default_factory=dict)
    maps: dict[str, np.ndarray] = field(default_factory=dict)
    status: Status = "OK"
    messages: list[str] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")
        if not np.isfinite(self.loss):
            raise ValueError(f"loss must be finite, got {self.loss}")


@dataclass(frozen=True)
class ObjectiveResult:
    total_loss: float
    metric_results: list[MetricResult]
    by_observer_loss: dict[str, float]
    status: Status
    messages: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not np.isfinite(self.total_loss):
            raise ValueError(f"total_loss must be finite, got {self.total_loss}")


class MetricProtocol(Protocol):
    name: str
    version: str

    def precompute_obs(
        self,
        obs: Obs2D,
        entry: MetricEntrySpec,
        measurement: MeasurementSpecV1 | None,
    ) -> object | None:
        ...

    def compute(
        self,
        pred: Obs2D,
        obs: Obs2D,
        entry: MetricEntrySpec,
        ctx: object | None,
        *,
        fail_penalty: float,
    ) -> MetricResult:
        ...
