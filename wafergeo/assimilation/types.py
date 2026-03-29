from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from wafergeo.assimilation.policies import FailurePolicy, LoggingPolicy, TransformPolicy
from wafergeo.core.types import LabelVolume, MeshGeom, Status, TSDFVolume
from wafergeo.metrics.base import MetricResult
from wafergeo.metrics.spec import MeasurementSpecV1, MetricSpecV2
from wafergeo.observe.spec import ObserverSpecV2

GeomOutput = LabelVolume | TSDFVolume | MeshGeom
ParamTransform = Literal["identity", "log", "logit"]
ParamKind = Literal["continuous", "int", "categorical"]
OutOfBoundsPolicy = Literal["clamp", "penalty", "fail"]


@runtime_checkable
class SurrogateModelProtocol(Protocol):
    name: str

    def predict(self, params: dict[str, object]) -> GeomOutput:
        ...


@runtime_checkable
class ModelLoaderProtocol(Protocol):
    name: str

    def load(
        self,
        model_ref: dict[str, object],
        store: object,
    ) -> SurrogateModelProtocol:
        ...


@dataclass(frozen=True)
class ParamAxis:
    name: str
    kind: ParamKind
    bounds: tuple[float, float] | None = None
    units: str | None = None
    transform: ParamTransform = "identity"
    default: float | int | str = 0.0
    choices: list[str] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ParamAxis.name must be non-empty")
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError("categorical axis requires non-empty choices")
            if not isinstance(self.default, str):
                raise ValueError("categorical axis default must be str")
            if self.default not in self.choices:
                raise ValueError(f"default='{self.default}' must be in choices")
        else:
            if self.bounds is None:
                raise ValueError(f"{self.kind} axis requires bounds")
            if self.bounds[0] >= self.bounds[1]:
                raise ValueError(f"invalid bounds={self.bounds} for axis={self.name}")
            if not isinstance(self.default, (float, int)):
                raise ValueError(f"{self.kind} axis default must be numeric")


@dataclass(frozen=True)
class ParamSpec:
    axes: list[ParamAxis]
    vector_order: list[str]

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("ParamSpec.axes must be non-empty")
        axis_names = [axis.name for axis in self.axes]
        if len(set(axis_names)) != len(axis_names):
            raise ValueError("ParamSpec axis names must be unique")
        if set(self.vector_order) != set(axis_names):
            raise ValueError("vector_order must contain exactly all axis names")
        if len(self.vector_order) != len(axis_names):
            raise ValueError("vector_order length mismatch")

    def _axis_by_name(self) -> dict[str, ParamAxis]:
        return {axis.name: axis for axis in self.axes}

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + float(np.exp(-x)))

    def _encode_axis_value(self, axis: ParamAxis, value: object) -> float:
        if axis.kind == "categorical":
            assert axis.choices is not None
            return float(axis.choices.index(str(value)))

        if not isinstance(value, (int, float)):
            raise TypeError(f"axis={axis.name} requires numeric value, got {type(value).__name__}")
        numeric = float(value)
        assert axis.bounds is not None
        lower, upper = axis.bounds

        if axis.transform == "identity":
            return numeric
        if axis.transform == "log":
            if numeric <= 0.0:
                raise ValueError(f"log transform requires >0, got {numeric} for axis={axis.name}")
            return float(np.log(numeric))
        if axis.transform == "logit":
            width = upper - lower
            y = (numeric - lower) / width
            y = float(np.clip(y, 1e-6, 1.0 - 1e-6))
            return float(np.log(y / (1.0 - y)))
        raise ValueError(f"unknown transform={axis.transform}")

    def _decode_axis_numeric(self, axis: ParamAxis, raw_value: float) -> float:
        assert axis.bounds is not None
        lower, upper = axis.bounds
        if axis.transform == "identity":
            return raw_value
        if axis.transform == "log":
            return float(np.exp(raw_value))
        if axis.transform == "logit":
            y = self._sigmoid(raw_value)
            return float(lower + y * (upper - lower))
        raise ValueError(f"unknown transform={axis.transform}")

    def encode(self, params_dict: dict[str, object]) -> np.ndarray:
        axis_map = self._axis_by_name()
        out = np.empty((len(self.vector_order),), dtype=np.float64)
        for idx, axis_name in enumerate(self.vector_order):
            axis = axis_map[axis_name]
            value = params_dict.get(axis_name, axis.default)
            out[idx] = self._encode_axis_value(axis, value)
        return out

    def decode(
        self,
        x: np.ndarray,
        out_of_bounds_policy: OutOfBoundsPolicy,
    ) -> tuple[dict[str, object], list[str], bool]:
        values = np.asarray(x, dtype=np.float64).reshape(-1)
        if values.shape[0] != len(self.vector_order):
            raise ValueError(
                f"x length must be {len(self.vector_order)}, got {values.shape[0]}"
            )

        axis_map = self._axis_by_name()
        params: dict[str, object] = {}
        warnings: list[str] = []
        oob = False

        for idx, axis_name in enumerate(self.vector_order):
            axis = axis_map[axis_name]
            raw = float(values[idx])

            if axis.kind == "categorical":
                assert axis.choices is not None
                max_idx = len(axis.choices) - 1
                rounded = int(np.rint(raw))
                if rounded < 0 or rounded > max_idx:
                    oob = True
                    warnings.append(
                        f"axis={axis.name} categorical index={rounded} out of [0,{max_idx}]"
                    )
                if out_of_bounds_policy == "clamp":
                    rounded = int(np.clip(rounded, 0, max_idx))
                else:
                    rounded = int(np.clip(rounded, 0, max_idx))
                params[axis.name] = axis.choices[rounded]
                continue

            decoded = self._decode_axis_numeric(axis, raw)
            assert axis.bounds is not None
            lower, upper = axis.bounds
            if decoded < lower or decoded > upper:
                oob = True
                warnings.append(
                    f"axis={axis.name} value={decoded:.6g} out of bounds [{lower:.6g},{upper:.6g}]"
                )
                if out_of_bounds_policy == "clamp":
                    decoded = float(np.clip(decoded, lower, upper))
            if axis.kind == "int":
                params[axis.name] = int(np.rint(decoded))
            else:
                params[axis.name] = float(decoded)

        return params, warnings, oob


@dataclass(frozen=True)
class ModelPackageSpec:
    model_package_id: str
    loader_key: str
    model_ref: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_package_id:
            raise ValueError("model_package_id must be non-empty")
        if not self.loader_key:
            raise ValueError("loader_key must be non-empty")


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    sem_obs_ids: dict[str, str]
    observer_specs: dict[str, ObserverSpecV2]
    metric_spec: MetricSpecV2
    measurement_specs_by_ref: dict[str, MeasurementSpecV1]
    model_package: ModelPackageSpec
    param_spec: ParamSpec
    transform_policy: TransformPolicy
    failure_policy: FailurePolicy
    logging_policy: LoggingPolicy

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if not self.sem_obs_ids:
            raise ValueError("sem_obs_ids must be non-empty")
        if not self.observer_specs:
            raise ValueError("observer_specs must be non-empty")
        if set(self.sem_obs_ids.keys()) != set(self.observer_specs.keys()):
            raise ValueError(
                "sem_obs_ids keys must match observer_specs keys exactly"
            )


@dataclass(frozen=True)
class EvalResult:
    candidate_id: str
    x: np.ndarray
    params: dict[str, object]
    total_loss: float
    per_observer: dict[str, float]
    metric_results: list[MetricResult]
    status: Status
    messages: list[str]
    timings: dict[str, float]
    artifacts: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if not np.isfinite(self.total_loss):
            raise ValueError(f"total_loss must be finite, got {self.total_loss}")
