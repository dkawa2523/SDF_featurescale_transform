from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]

from wafergeo.core.hashing import hash_config

ScenarioName = Literal["cube", "layers3", "thin_shell", "diagonal", "real_vti"]
PointToCellPolicy = Literal["nearest", "majority", "majority_nearest_tie"]
MeshBackendName = Literal["vtk", "naive_interface"]
MeshModeName = Literal["material_shell", "interface_mesh"]
DiagnosisScope = Literal["auto", "real_vti", "global_max"]


def _default_thresholds() -> dict[str, float]:
    return {
        "mesh_boundary_iou_min": 0.80,
        "mesh_boundary_chamfer_nm_max": 2.0,
        "mesh_boundary_coverage_min": 0.70,
        "sdf_roundtrip_acc_min": 0.999,
        "render_diff_rate_max": 0.10,
        "policy_gap_max": 0.05,
    }


@dataclass(frozen=True)
class BenchmarkSpecV1:
    schema_version: str
    case_id: str
    scenarios: tuple[ScenarioName, ...]
    point_to_cell_policies: tuple[PointToCellPolicy, ...] = ("nearest", "majority")
    mesh_backends: tuple[MeshBackendName, ...] = ("naive_interface",)
    mesh_modes: tuple[MeshModeName, ...] = ("material_shell", "interface_mesh")
    diagnosis_scope: DiagnosisScope = "auto"
    thresholds: dict[str, float] = field(default_factory=_default_thresholds)
    real_vti_path: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "correspondence_bench/v1":
            raise ValueError(
                "schema_version must be 'correspondence_bench/v1', "
                f"got {self.schema_version}"
            )
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if not self.scenarios:
            raise ValueError("scenarios must be non-empty")
        for key in (
            "mesh_boundary_iou_min",
            "mesh_boundary_chamfer_nm_max",
            "mesh_boundary_coverage_min",
            "sdf_roundtrip_acc_min",
            "render_diff_rate_max",
            "policy_gap_max",
        ):
            if key not in self.thresholds:
                raise ValueError(f"thresholds missing required key: {key}")
        if "real_vti" in self.scenarios and not self.real_vti_path:
            raise ValueError("real_vti_path is required when scenarios includes 'real_vti'")
        if self.diagnosis_scope not in {"auto", "real_vti", "global_max"}:
            raise ValueError(
                "diagnosis_scope must be one of {'auto','real_vti','global_max'}, "
                f"got {self.diagnosis_scope}"
            )

    def to_hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "scenarios": list(self.scenarios),
            "point_to_cell_policies": list(self.point_to_cell_policies),
            "mesh_backends": list(self.mesh_backends),
            "mesh_modes": list(self.mesh_modes),
            "diagnosis_scope": self.diagnosis_scope,
            "thresholds": dict(self.thresholds),
            "real_vti_path": self.real_vti_path,
        }


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("benchmark spec YAML root must be mapping")
    return {str(k): v for k, v in raw.items()}


def _as_list(raw: dict[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    if value is None:
        raise ValueError(f"missing required key: {key}")
    if not isinstance(value, list):
        raise ValueError(f"{key} must be list")
    return list(value)


def _as_str(raw: dict[str, Any], key: str, default: str | None = None) -> str:
    if key not in raw:
        if default is None:
            raise ValueError(f"missing required key: {key}")
        return default
    return str(raw[key])


def _as_mapping(
    raw: dict[str, Any],
    key: str,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if key not in raw:
        return default or {}
    value = raw[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be mapping")
    return {str(k): v for k, v in value.items()}


def load_benchmark_spec_yaml(path: str | Path) -> BenchmarkSpecV1:
    raw = _read_yaml(Path(path))
    scenario_values = tuple(str(v) for v in _as_list(raw, "scenarios"))
    policy_values = tuple(str(v) for v in _as_list(raw, "point_to_cell_policies"))
    backend_values = tuple(str(v) for v in _as_list(raw, "mesh_backends"))
    mode_values = tuple(str(v) for v in _as_list(raw, "mesh_modes"))

    thresholds_raw = _as_mapping(raw, "thresholds", default={})
    thresholds = _default_thresholds()
    thresholds.update({str(k): float(v) for k, v in thresholds_raw.items()})

    return BenchmarkSpecV1(
        schema_version=_as_str(raw, "schema_version"),
        case_id=_as_str(raw, "case_id"),
        scenarios=tuple(scenario_values),  # type: ignore[arg-type]
        point_to_cell_policies=tuple(policy_values),  # type: ignore[arg-type]
        mesh_backends=tuple(backend_values),  # type: ignore[arg-type]
        mesh_modes=tuple(mode_values),  # type: ignore[arg-type]
        diagnosis_scope=_as_str(raw, "diagnosis_scope", default="auto"),  # type: ignore[arg-type]
        thresholds=thresholds,
        real_vti_path=raw.get("real_vti_path"),
    )


def benchmark_spec_hash(spec: BenchmarkSpecV1) -> str:
    return hash_config(spec.to_hash_payload())
