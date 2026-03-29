from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]

from wafergeo.core.grid import AxisOrder, GridSpec, SampleLocation
from wafergeo.core.hashing import hash_config

ContourFormat = Literal["csv", "json", "auto"]
CoordSystem = Literal["pixel", "nm"]
CoordUnits = Literal["px", "nm", "um"]
PixelYPolicy = Literal["flip_y", "keep"]
SEMTSDFMode = Literal["signed_region", "unsigned_curve", "auto"]
SEMWeightMode = Literal["none", "uniform", "from_image"]


@dataclass(frozen=True)
class SEMInputSpec:
    contour_format: ContourFormat = "auto"
    coord_system: CoordSystem = "pixel"
    units: CoordUnits = "px"
    pixel_size_nm: float | None = None
    pixel_y_policy: PixelYPolicy = "flip_y"

    def __post_init__(self) -> None:
        if (
            self.coord_system == "pixel"
            and self.pixel_size_nm is not None
            and self.pixel_size_nm <= 0.0
        ):
            raise ValueError("input.pixel_size_nm must be >0 when provided")


@dataclass(frozen=True)
class ContourNormalizeSpec:
    close_tol_nm: float = 5.0
    enforce_orientation: bool = True
    resample_points_closed: int = 256
    resample_points_open: int = 256

    def __post_init__(self) -> None:
        if self.close_tol_nm < 0.0:
            raise ValueError("normalize.close_tol_nm must be >= 0")
        if self.resample_points_closed < 2:
            raise ValueError("normalize.resample_points_closed must be >= 2")
        if self.resample_points_open < 2:
            raise ValueError("normalize.resample_points_open must be >= 2")


@dataclass(frozen=True)
class SEMTSDFSpec:
    mode: SEMTSDFMode = "auto"
    mu_nm: float = 30.0
    open_tube_radius_nm: float = 10.0
    distance_backend: str = "scipy"

    def __post_init__(self) -> None:
        if self.mu_nm <= 0.0:
            raise ValueError("tsdf.mu_nm must be > 0")
        if self.open_tube_radius_nm <= 0.0:
            raise ValueError("tsdf.open_tube_radius_nm must be > 0")
        if not self.distance_backend:
            raise ValueError("tsdf.distance_backend must be non-empty")


@dataclass(frozen=True)
class SEMWeightSpec:
    mode: SEMWeightMode = "uniform"
    default_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.default_weight < 0.0:
            raise ValueError("weight.default_weight must be >= 0")


@dataclass(frozen=True)
class SEMQASpec:
    min_mask_fraction: float = 0.0
    max_open_contours: int = 32

    def __post_init__(self) -> None:
        if self.min_mask_fraction < 0.0 or self.min_mask_fraction > 1.0:
            raise ValueError("qa.min_mask_fraction must be within [0,1]")
        if self.max_open_contours < 0:
            raise ValueError("qa.max_open_contours must be >= 0")


@dataclass(frozen=True)
class SEMOverlaySpec:
    enable: bool = False
    draw_contours: bool = True


@dataclass(frozen=True)
class SEMPrepareSpecV1:
    schema_version: str
    profile_id: str
    target_grid_2d: GridSpec
    target_shape_yx: tuple[int, int] | None
    input: SEMInputSpec
    normalize: ContourNormalizeSpec
    tsdf: SEMTSDFSpec
    weight: SEMWeightSpec
    qa: SEMQASpec
    overlay: SEMOverlaySpec

    def __post_init__(self) -> None:
        if self.schema_version != "sem_prepare/v1":
            raise ValueError(
                f"schema_version must be 'sem_prepare/v1', got {self.schema_version}"
            )
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        if self.target_grid_2d.dim != 2:
            raise ValueError("target_grid_2d.dim must be 2")
        if self.target_shape_yx is not None:
            y_size, x_size = self.target_shape_yx
            if y_size <= 1 or x_size <= 1:
                raise ValueError("target_shape_yx entries must be > 1")

    def to_hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "target_grid_2d": {
                "dim": self.target_grid_2d.dim,
                "spacing": list(self.target_grid_2d.spacing),
                "origin": list(self.target_grid_2d.origin),
                "axis_order": self.target_grid_2d.axis_order,
                "sample_location": self.target_grid_2d.sample_location,
                "units": self.target_grid_2d.units,
            },
            "target_shape_yx": (
                None if self.target_shape_yx is None else list(self.target_shape_yx)
            ),
            "input": asdict(self.input),
            "normalize": asdict(self.normalize),
            "tsdf": asdict(self.tsdf),
            "weight": asdict(self.weight),
            "qa": asdict(self.qa),
            "overlay": asdict(self.overlay),
        }


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SEM prepare spec YAML root must be a mapping")
    return {str(k): v for k, v in raw.items()}


def _as_mapping(
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


def _as_list(parent: dict[str, Any], key: str) -> list[Any]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return list(value)


def _as_optional_shape2(parent: dict[str, Any], key: str) -> tuple[int, int] | None:
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{key} must be [Y,X] or null")
    return (int(value[0]), int(value[1]))


def _as_str(parent: dict[str, Any], key: str, default: str | None = None) -> str:
    if key not in parent:
        if default is None:
            raise ValueError(f"missing required key: {key}")
        return default
    return str(parent[key])


def _as_int(parent: dict[str, Any], key: str, default: int | None = None) -> int:
    if key not in parent:
        if default is None:
            raise ValueError(f"missing required key: {key}")
        return int(default)
    return int(parent[key])


def _as_float(parent: dict[str, Any], key: str, default: float | None = None) -> float:
    if key not in parent:
        if default is None:
            raise ValueError(f"missing required key: {key}")
        return float(default)
    return float(parent[key])


def _as_bool(parent: dict[str, Any], key: str, default: bool) -> bool:
    if key not in parent:
        return default
    value = parent[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def load_sem_prepare_spec_yaml(path: str | Path) -> SEMPrepareSpecV1:
    raw = _read_yaml(Path(path))

    grid_raw = _as_mapping(raw, "target_grid_2d")
    spacing_raw = _as_list(grid_raw, "spacing")
    origin_raw = _as_list(grid_raw, "origin")
    if len(spacing_raw) != 2:
        raise ValueError("target_grid_2d.spacing must have length 2")
    if len(origin_raw) != 2:
        raise ValueError("target_grid_2d.origin must have length 2")

    target_grid_2d = GridSpec(
        dim=_as_int(grid_raw, "dim"),
        spacing=(float(spacing_raw[0]), float(spacing_raw[1])),
        origin=(float(origin_raw[0]), float(origin_raw[1])),
        axis_order=cast(AxisOrder, _as_str(grid_raw, "axis_order")),
        sample_location=cast(SampleLocation, _as_str(grid_raw, "sample_location")),
        units=_as_str(grid_raw, "units"),
    )

    input_raw = _as_mapping(raw, "input", default={})
    normalize_raw = _as_mapping(raw, "normalize", default={})
    tsdf_raw = _as_mapping(raw, "tsdf", default={})
    weight_raw = _as_mapping(raw, "weight", default={})
    qa_raw = _as_mapping(raw, "qa", default={})
    overlay_raw = _as_mapping(raw, "overlay", default={})

    pixel_size_nm: float | None
    if "pixel_size_nm" in input_raw and input_raw["pixel_size_nm"] is not None:
        pixel_size_nm = float(input_raw["pixel_size_nm"])
    else:
        pixel_size_nm = None

    return SEMPrepareSpecV1(
        schema_version=_as_str(raw, "schema_version"),
        profile_id=_as_str(raw, "profile_id"),
        target_grid_2d=target_grid_2d,
        target_shape_yx=_as_optional_shape2(raw, "target_shape_yx"),
        input=SEMInputSpec(
            contour_format=cast(ContourFormat, _as_str(input_raw, "contour_format", "auto")),
            coord_system=cast(CoordSystem, _as_str(input_raw, "coord_system", "pixel")),
            units=cast(CoordUnits, _as_str(input_raw, "units", "px")),
            pixel_size_nm=pixel_size_nm,
            pixel_y_policy=cast(PixelYPolicy, _as_str(input_raw, "pixel_y_policy", "flip_y")),
        ),
        normalize=ContourNormalizeSpec(
            close_tol_nm=_as_float(normalize_raw, "close_tol_nm", 5.0),
            enforce_orientation=_as_bool(normalize_raw, "enforce_orientation", True),
            resample_points_closed=_as_int(normalize_raw, "resample_points_closed", 256),
            resample_points_open=_as_int(normalize_raw, "resample_points_open", 256),
        ),
        tsdf=SEMTSDFSpec(
            mode=cast(SEMTSDFMode, _as_str(tsdf_raw, "mode", "auto")),
            mu_nm=_as_float(tsdf_raw, "mu_nm", 30.0),
            open_tube_radius_nm=_as_float(tsdf_raw, "open_tube_radius_nm", 10.0),
            distance_backend=_as_str(tsdf_raw, "distance_backend", "scipy"),
        ),
        weight=SEMWeightSpec(
            mode=cast(SEMWeightMode, _as_str(weight_raw, "mode", "uniform")),
            default_weight=_as_float(weight_raw, "default_weight", 1.0),
        ),
        qa=SEMQASpec(
            min_mask_fraction=_as_float(qa_raw, "min_mask_fraction", 0.0),
            max_open_contours=_as_int(qa_raw, "max_open_contours", 32),
        ),
        overlay=SEMOverlaySpec(
            enable=_as_bool(overlay_raw, "enable", False),
            draw_contours=_as_bool(overlay_raw, "draw_contours", True),
        ),
    )


def sem_prepare_spec_hash(spec: SEMPrepareSpecV1) -> str:
    return hash_config(spec.to_hash_payload())
