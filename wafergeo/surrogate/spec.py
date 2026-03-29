from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]

from wafergeo.core.hashing import hash_config

TaskKind = Literal["sdf", "mesh", "hybrid"]
StorageMode = Literal["linked", "packed"]


@dataclass(frozen=True)
class GroupSplitSpec:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if any(v < 0.0 for v in (self.train_ratio, self.val_ratio, self.test_ratio)):
            raise ValueError("split ratios must be >= 0")
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")


@dataclass(frozen=True)
class DatasetQASpec:
    require_material_count_leq: int = 5
    warn_missing_material_rate_gt: float = 0.3
    warn_interface_imbalance_gt: float = 0.8

    def __post_init__(self) -> None:
        if self.require_material_count_leq <= 0:
            raise ValueError("require_material_count_leq must be > 0")
        if not 0.0 <= self.warn_missing_material_rate_gt <= 1.0:
            raise ValueError("warn_missing_material_rate_gt must be within [0,1]")
        if not 0.0 <= self.warn_interface_imbalance_gt <= 1.0:
            raise ValueError("warn_interface_imbalance_gt must be within [0,1]")


@dataclass(frozen=True)
class DatasetBuildSpecV1:
    schema_version: str
    profile_id: str
    dataset_id_prefix: str
    task_kind: TaskKind
    storage_mode: StorageMode
    include_sdf_features: dict[str, bool]
    include_mesh: bool
    include_obs2d_pack: bool
    obs_targets: dict[str, object]
    split: GroupSplitSpec
    qa: DatasetQASpec
    input_manifest_path: str
    param_spec_hash: str
    fail_on_qa_status: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != "surrogate_build/v1":
            raise ValueError(
                f"schema_version must be 'surrogate_build/v1', got {self.schema_version}"
            )
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        if not self.dataset_id_prefix:
            raise ValueError("dataset_id_prefix must be non-empty")
        if not self.input_manifest_path:
            raise ValueError("input_manifest_path must be non-empty")
        if not self.param_spec_hash:
            raise ValueError("param_spec_hash must be non-empty")

    def to_hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "dataset_id_prefix": self.dataset_id_prefix,
            "task_kind": self.task_kind,
            "storage_mode": self.storage_mode,
            "include_sdf_features": dict(self.include_sdf_features),
            "include_mesh": self.include_mesh,
            "include_obs2d_pack": self.include_obs2d_pack,
            "obs_targets": dict(self.obs_targets),
            "split": asdict(self.split),
            "qa": asdict(self.qa),
            "input_manifest_path": self.input_manifest_path,
            "param_spec_hash": self.param_spec_hash,
            "fail_on_qa_status": self.fail_on_qa_status,
        }


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Dataset build spec YAML root must be a mapping")
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


def _as_str(parent: dict[str, Any], key: str, default: str | None = None) -> str:
    if key not in parent:
        if default is None:
            raise ValueError(f"missing required key: {key}")
        return default
    return str(parent[key])


def _as_float(parent: dict[str, Any], key: str, default: float) -> float:
    if key not in parent:
        return float(default)
    return float(parent[key])


def _as_int(parent: dict[str, Any], key: str, default: int) -> int:
    if key not in parent:
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
    return bool(value)


def load_dataset_build_spec_yaml(path: str | Path) -> DatasetBuildSpecV1:
    raw = _read_yaml(Path(path))
    split_raw = _as_mapping(raw, "split", default={})
    qa_raw = _as_mapping(raw, "qa", default={})

    include_sdf_features_raw = _as_mapping(raw, "include_sdf_features", default={})
    include_sdf_features = {
        "pair_code": bool(include_sdf_features_raw.get("pair_code", False)),
        "d_boundary": bool(include_sdf_features_raw.get("d_boundary", False)),
        "present_mask": bool(include_sdf_features_raw.get("present_mask", False)),
    }

    return DatasetBuildSpecV1(
        schema_version=_as_str(raw, "schema_version"),
        profile_id=_as_str(raw, "profile_id"),
        dataset_id_prefix=_as_str(raw, "dataset_id_prefix"),
        task_kind=cast(TaskKind, _as_str(raw, "task_kind")),
        storage_mode=cast(StorageMode, _as_str(raw, "storage_mode")),
        include_sdf_features=include_sdf_features,
        include_mesh=_as_bool(raw, "include_mesh", True),
        include_obs2d_pack=_as_bool(raw, "include_obs2d_pack", False),
        obs_targets=_as_mapping(raw, "obs_targets", default={}),
        split=GroupSplitSpec(
            train_ratio=_as_float(split_raw, "train_ratio", 0.8),
            val_ratio=_as_float(split_raw, "val_ratio", 0.1),
            test_ratio=_as_float(split_raw, "test_ratio", 0.1),
            seed=_as_int(split_raw, "seed", 0),
        ),
        qa=DatasetQASpec(
            require_material_count_leq=_as_int(qa_raw, "require_material_count_leq", 5),
            warn_missing_material_rate_gt=_as_float(qa_raw, "warn_missing_material_rate_gt", 0.3),
            warn_interface_imbalance_gt=_as_float(
                qa_raw,
                "warn_interface_imbalance_gt",
                0.8,
            ),
        ),
        input_manifest_path=_as_str(raw, "input_manifest_path"),
        param_spec_hash=_as_str(raw, "param_spec_hash"),
        fail_on_qa_status=_as_bool(raw, "fail_on_qa_status", False),
    )


def dataset_build_spec_hash(spec: DatasetBuildSpecV1) -> str:
    return hash_config(spec.to_hash_payload())
