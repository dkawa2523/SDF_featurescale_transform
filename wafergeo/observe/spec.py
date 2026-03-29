from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]

from wafergeo.core.grid import AxisOrder, GridSpec, SampleLocation
from wafergeo.core.hashing import hash_config
from wafergeo.observe.errors import ObserverSpecValidationError

ObserverKind = Literal["topdown_exposed", "slice"]
ContourSource = Literal["tsdf", "mask"]


@dataclass(frozen=True)
class MaskDefSpec:
    kind: str
    include_materials: tuple[str, ...] = ()
    include_ids: tuple[int, ...] = ()
    ignore_materials: tuple[str, ...] = ()
    ignore_ids: tuple[int, ...] = ()
    slab_thickness_nm: float = 0.0

    def __post_init__(self) -> None:
        allowed = {"binary_solid", "material_union", "exposed_union"}
        if self.kind not in allowed:
            raise ObserverSpecValidationError(
                f"mask_definition.kind must be one of {sorted(allowed)}, got {self.kind}"
            )
        if self.slab_thickness_nm < 0.0:
            raise ObserverSpecValidationError("mask_definition.slab_thickness_nm must be >= 0")


@dataclass(frozen=True)
class Tsdf2DSpec:
    mu_nm: float
    engine: str = "scipy"
    band_only: bool = True

    def __post_init__(self) -> None:
        if self.mu_nm <= 0.0:
            raise ObserverSpecValidationError(f"tsdf2d.mu_nm must be > 0, got {self.mu_nm}")
        if not self.engine:
            raise ObserverSpecValidationError("tsdf2d.engine must be non-empty")


@dataclass(frozen=True)
class ContourSpec:
    source: ContourSource = "tsdf"
    level: float = 0.0
    smoothing_sigma_nm: float = 0.0
    resample_points: int = 256
    simplify_tolerance_nm: float = 0.0
    backend: str = "skimage"
    allow_missing_backend: bool = True

    def __post_init__(self) -> None:
        if self.resample_points < 0:
            raise ObserverSpecValidationError("contour.resample_points must be >= 0")
        if self.smoothing_sigma_nm < 0.0:
            raise ObserverSpecValidationError("contour.smoothing_sigma_nm must be >= 0")
        if self.simplify_tolerance_nm < 0.0:
            raise ObserverSpecValidationError("contour.simplify_tolerance_nm must be >= 0")
        if not self.backend:
            raise ObserverSpecValidationError("contour.backend must be non-empty")


@dataclass(frozen=True)
class ObserverSpecV2:
    schema_version: str
    name: str
    kind: ObserverKind
    target_grid_2d: GridSpec
    roi: dict[str, float]
    mask_definition: MaskDefSpec
    tsdf2d: Tsdf2DSpec
    contour: ContourSpec
    params: dict[str, object] = field(default_factory=dict)
    debug: dict[str, object] = field(default_factory=dict)
    qa: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != "observer/v2":
            raise ObserverSpecValidationError(
                f"schema_version must be 'observer/v2', got {self.schema_version}"
            )
        if not self.name:
            raise ObserverSpecValidationError("name must be non-empty")
        if self.target_grid_2d.dim != 2:
            raise ObserverSpecValidationError("target_grid_2d.dim must be 2")

    def to_hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "kind": self.kind,
            "target_grid_2d": {
                "dim": self.target_grid_2d.dim,
                "spacing": list(self.target_grid_2d.spacing),
                "origin": list(self.target_grid_2d.origin),
                "axis_order": self.target_grid_2d.axis_order,
                "sample_location": self.target_grid_2d.sample_location,
                "units": self.target_grid_2d.units,
            },
            "roi": dict(self.roi),
            "mask_definition": asdict(self.mask_definition),
            "tsdf2d": asdict(self.tsdf2d),
            "contour": asdict(self.contour),
            "params": dict(self.params),
            "debug": dict(self.debug),
            "qa": dict(self.qa),
        }


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ObserverSpecValidationError("observer spec YAML root must be a mapping")
    return {str(k): v for k, v in raw.items()}


def _as_mapping(
    parent: dict[str, Any],
    key: str,
    *,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if key not in parent:
        if default is None:
            raise ObserverSpecValidationError(f"missing required key: {key}")
        return default
    value = parent[key]
    if not isinstance(value, dict):
        raise ObserverSpecValidationError(f"{key} must be a mapping")
    return {str(k): v for k, v in value.items()}


def _as_str(parent: dict[str, Any], key: str, default: str | None = None) -> str:
    if key not in parent:
        if default is None:
            raise ObserverSpecValidationError(f"missing required key: {key}")
        return default
    return str(parent[key])


def _as_float(parent: dict[str, Any], key: str, default: float | None = None) -> float:
    if key not in parent:
        if default is None:
            raise ObserverSpecValidationError(f"missing required key: {key}")
        return float(default)
    return float(parent[key])


def _as_int(parent: dict[str, Any], key: str, default: int | None = None) -> int:
    if key not in parent:
        if default is None:
            raise ObserverSpecValidationError(f"missing required key: {key}")
        return int(default)
    return int(parent[key])


def _as_bool(parent: dict[str, Any], key: str, default: bool) -> bool:
    if key not in parent:
        return bool(default)
    value = parent[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(value)


def _as_list(parent: dict[str, Any], key: str) -> list[Any]:
    value = parent.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ObserverSpecValidationError(f"{key} must be a list")
    return list(value)


def _as_float_mapping(raw: dict[str, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in raw.items()}


def load_observer_spec_yaml(path: str | Path) -> ObserverSpecV2:
    spec_path = Path(path)
    raw = _read_yaml(spec_path)

    grid_raw = _as_mapping(raw, "target_grid_2d")
    spacing_raw = _as_list(grid_raw, "spacing")
    origin_raw = _as_list(grid_raw, "origin")

    target_grid_2d = GridSpec(
        dim=_as_int(grid_raw, "dim"),
        spacing=(float(spacing_raw[0]), float(spacing_raw[1])),
        origin=(float(origin_raw[0]), float(origin_raw[1])),
        axis_order=cast(AxisOrder, _as_str(grid_raw, "axis_order")),
        sample_location=cast(SampleLocation, _as_str(grid_raw, "sample_location")),
        units=_as_str(grid_raw, "units"),
    )

    mask_raw = _as_mapping(raw, "mask_definition")
    tsdf_raw = _as_mapping(raw, "tsdf2d")
    contour_raw = _as_mapping(raw, "contour", default={})

    return ObserverSpecV2(
        schema_version=_as_str(raw, "schema_version"),
        name=_as_str(raw, "name"),
        kind=cast(ObserverKind, _as_str(raw, "kind")),
        target_grid_2d=target_grid_2d,
        roi=_as_float_mapping(_as_mapping(raw, "roi", default={})),
        mask_definition=MaskDefSpec(
            kind=_as_str(mask_raw, "kind"),
            include_materials=tuple(str(v) for v in _as_list(mask_raw, "include_materials")),
            include_ids=tuple(int(v) for v in _as_list(mask_raw, "include_ids")),
            ignore_materials=tuple(str(v) for v in _as_list(mask_raw, "ignore_materials")),
            ignore_ids=tuple(int(v) for v in _as_list(mask_raw, "ignore_ids")),
            slab_thickness_nm=_as_float(mask_raw, "slab_thickness_nm", 0.0),
        ),
        tsdf2d=Tsdf2DSpec(
            mu_nm=_as_float(tsdf_raw, "mu_nm"),
            engine=_as_str(tsdf_raw, "engine", "scipy"),
            band_only=_as_bool(tsdf_raw, "band_only", True),
        ),
        contour=ContourSpec(
            source=cast(ContourSource, _as_str(contour_raw, "source", "tsdf")),
            level=_as_float(contour_raw, "level", 0.0),
            smoothing_sigma_nm=_as_float(contour_raw, "smoothing_sigma_nm", 0.0),
            resample_points=_as_int(contour_raw, "resample_points", 256),
            simplify_tolerance_nm=_as_float(contour_raw, "simplify_tolerance_nm", 0.0),
            backend=_as_str(contour_raw, "backend", "skimage"),
            allow_missing_backend=_as_bool(contour_raw, "allow_missing_backend", True),
        ),
        params=_as_mapping(raw, "params", default={}),
        debug=_as_mapping(raw, "debug", default={}),
        qa=_as_mapping(raw, "qa", default={}),
    )


def observer_spec_hash(spec: ObserverSpecV2) -> str:
    return hash_config(spec.to_hash_payload())
