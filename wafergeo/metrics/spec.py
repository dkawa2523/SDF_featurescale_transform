from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]

from wafergeo.core.hashing import hash_config

MetricEntryName = Literal["tsdf_band_robust_weight", "contour_chamfer", "cd_linescan"]
EdgePair = Literal["outer", "inner"]
LineAxis = Literal["x", "y"]


@dataclass(frozen=True)
class MetricEntrySpec:
    name: MetricEntryName
    weight: float
    observers: tuple[str, ...] | None
    params: dict[str, object]
    measurement_ref: str | None = None

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError(f"metric weight must be >= 0, got {self.weight}")
        if self.name == "cd_linescan" and not self.measurement_ref:
            raise ValueError("measurement_ref is required for cd_linescan")

    def to_hash_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "weight": self.weight,
            "observers": None if self.observers is None else list(self.observers),
            "params": dict(self.params),
            "measurement_ref": self.measurement_ref,
        }


@dataclass(frozen=True)
class MetricSpecV2:
    schema_version: str
    metric_set_id: str
    fail_penalty: float = 1e6
    observer_weights: dict[str, float] | None = None
    metrics: tuple[MetricEntrySpec, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != "metric/v2":
            raise ValueError(f"schema_version must be 'metric/v2', got {self.schema_version}")
        if not self.metric_set_id:
            raise ValueError("metric_set_id must be non-empty")
        if self.fail_penalty <= 0.0:
            raise ValueError("fail_penalty must be > 0")

    def to_hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "metric_set_id": self.metric_set_id,
            "fail_penalty": self.fail_penalty,
            "observer_weights": dict(self.observer_weights or {}),
            "metrics": [entry.to_hash_payload() for entry in self.metrics],
        }


@dataclass(frozen=True)
class MeasurementLineSpec:
    id: str
    axis: LineAxis
    coord_nm: float
    range_nm: tuple[float, float]
    expected_edges: int
    edge_pair: EdgePair
    method: Literal["tsdf_zero_cross"]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("measurement line id must be non-empty")
        if self.range_nm[0] >= self.range_nm[1]:
            raise ValueError(f"range_nm must be increasing, got {self.range_nm}")
        if self.expected_edges <= 0:
            raise ValueError("expected_edges must be > 0")

    def to_hash_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "axis": self.axis,
            "coord_nm": self.coord_nm,
            "range_nm": list(self.range_nm),
            "expected_edges": self.expected_edges,
            "edge_pair": self.edge_pair,
            "method": self.method,
        }


@dataclass(frozen=True)
class MeasurementSpecV1:
    schema_version: str
    name: str
    lines: tuple[MeasurementLineSpec, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "measurement/v1":
            raise ValueError(
                f"schema_version must be 'measurement/v1', got {self.schema_version}"
            )
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.lines:
            raise ValueError("lines must be non-empty")

    def to_hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "lines": [line.to_hash_payload() for line in self.lines],
        }


def _read_yaml(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("YAML root must be mapping")
    return {str(k): v for k, v in raw.items()}


def _as_mapping(
    parent: dict[str, Any],
    key: str,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if key not in parent:
        if default is None:
            raise ValueError(f"missing required key: {key}")
        return default
    value = parent[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be mapping")
    return {str(k): v for k, v in value.items()}


def _as_list(parent: dict[str, Any], key: str, default: list[Any] | None = None) -> list[Any]:
    if key not in parent:
        return default or []
    value = parent[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be list")
    return list(value)


def _as_str(parent: dict[str, Any], key: str, default: str | None = None) -> str:
    if key not in parent:
        if default is None:
            raise ValueError(f"missing required key: {key}")
        return default
    return str(parent[key])


def _as_float(parent: dict[str, Any], key: str, default: float | None = None) -> float:
    if key not in parent:
        if default is None:
            raise ValueError(f"missing required key: {key}")
        return float(default)
    return float(parent[key])


def _as_int(parent: dict[str, Any], key: str, default: int | None = None) -> int:
    if key not in parent:
        if default is None:
            raise ValueError(f"missing required key: {key}")
        return int(default)
    return int(parent[key])


def _as_observer_weights(raw: dict[str, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in raw.items()}


def load_metric_spec_yaml(path: str | Path) -> MetricSpecV2:
    raw = _read_yaml(path)
    metric_items = _as_list(raw, "metrics")

    metrics: list[MetricEntrySpec] = []
    for idx, item in enumerate(metric_items):
        if not isinstance(item, dict):
            raise ValueError(f"metrics[{idx}] must be mapping")
        row = {str(k): v for k, v in item.items()}
        observers_raw = row.get("observers")
        observers: tuple[str, ...] | None
        if observers_raw is None:
            observers = None
        else:
            if not isinstance(observers_raw, list):
                raise ValueError(f"metrics[{idx}].observers must be list or null")
            observers = tuple(str(v) for v in observers_raw)

        params = _as_mapping(row, "params", default={})
        metrics.append(
            MetricEntrySpec(
                name=cast(MetricEntryName, _as_str(row, "name")),
                weight=_as_float(row, "weight", 1.0),
                observers=observers,
                params={str(k): v for k, v in params.items()},
                measurement_ref=str(row["measurement_ref"]) if "measurement_ref" in row else None,
            )
        )

    observer_weights = _as_observer_weights(_as_mapping(raw, "observer_weights", default={}))

    return MetricSpecV2(
        schema_version=_as_str(raw, "schema_version"),
        metric_set_id=_as_str(raw, "metric_set_id"),
        fail_penalty=_as_float(raw, "fail_penalty", 1e6),
        observer_weights=observer_weights,
        metrics=tuple(metrics),
    )


def load_measurement_spec_yaml(path: str | Path) -> MeasurementSpecV1:
    raw = _read_yaml(path)
    line_items = _as_list(raw, "lines")

    lines: list[MeasurementLineSpec] = []
    for idx, item in enumerate(line_items):
        if not isinstance(item, dict):
            raise ValueError(f"lines[{idx}] must be mapping")
        row = {str(k): v for k, v in item.items()}
        range_raw = row.get("range_nm")
        if not isinstance(range_raw, list) or len(range_raw) != 2:
            raise ValueError(f"lines[{idx}].range_nm must be [min,max]")

        lines.append(
            MeasurementLineSpec(
                id=_as_str(row, "id"),
                axis=cast(LineAxis, _as_str(row, "axis")),
                coord_nm=_as_float(row, "coord_nm"),
                range_nm=(float(range_raw[0]), float(range_raw[1])),
                expected_edges=_as_int(row, "expected_edges"),
                edge_pair=cast(EdgePair, _as_str(row, "edge_pair", "outer")),
                method=cast(Literal["tsdf_zero_cross"], _as_str(row, "method", "tsdf_zero_cross")),
            )
        )

    return MeasurementSpecV1(
        schema_version=_as_str(raw, "schema_version"),
        name=_as_str(raw, "name"),
        lines=tuple(lines),
    )


def metric_spec_hash(spec: MetricSpecV2) -> str:
    return hash_config(spec.to_hash_payload())


def measurement_spec_hash(spec: MeasurementSpecV1) -> str:
    return hash_config(spec.to_hash_payload())
