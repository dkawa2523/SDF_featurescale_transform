from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

StorageMode = Literal["linked", "packed"]
TaskKind = Literal["sdf", "mesh", "hybrid"]


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    group_id: str
    recipe_params: dict[str, object] = field(default_factory=dict)
    param_vector: list[float] | None = None
    label_artifact_id: str | None = None
    tsdf_artifact_id: str | None = None
    mesh_artifact_id: str | None = None
    obs2d_sim_ids: dict[str, str] = field(default_factory=dict)
    packed_paths: dict[str, str] = field(default_factory=dict)
    qa: dict[str, object] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if not self.group_id:
            raise ValueError("group_id must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "recipe_params": dict(self.recipe_params),
            "param_vector": None if self.param_vector is None else list(self.param_vector),
            "label_artifact_id": self.label_artifact_id,
            "tsdf_artifact_id": self.tsdf_artifact_id,
            "mesh_artifact_id": self.mesh_artifact_id,
            "obs2d_sim_ids": dict(self.obs2d_sim_ids),
            "packed_paths": dict(self.packed_paths),
            "qa": dict(self.qa),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SampleRecord:
        return cls(
            sample_id=str(raw["sample_id"]),
            group_id=str(raw["group_id"]),
            recipe_params={str(k): v for k, v in dict(raw.get("recipe_params", {})).items()},
            param_vector=(
                None
                if raw.get("param_vector") is None
                else [float(v) for v in list(raw.get("param_vector", []))]
            ),
            label_artifact_id=(
                None
                if raw.get("label_artifact_id") in (None, "")
                else str(raw["label_artifact_id"])
            ),
            tsdf_artifact_id=(
                None if raw.get("tsdf_artifact_id") in (None, "") else str(raw["tsdf_artifact_id"])
            ),
            mesh_artifact_id=(
                None if raw.get("mesh_artifact_id") in (None, "") else str(raw["mesh_artifact_id"])
            ),
            obs2d_sim_ids={str(k): str(v) for k, v in dict(raw.get("obs2d_sim_ids", {})).items()},
            packed_paths={str(k): str(v) for k, v in dict(raw.get("packed_paths", {})).items()},
            qa={str(k): v for k, v in dict(raw.get("qa", {})).items()},
            meta={str(k): v for k, v in dict(raw.get("meta", {})).items()},
        )


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    dataset_id: str
    profile_id: str
    created_at: str
    generator_version: str
    git_commit: str
    storage_mode: StorageMode
    task_kind: TaskKind
    materials: dict[str, object]
    grid3d: dict[str, object]
    observers: list[str]
    param_spec_hash: str
    observer_spec_hashes: dict[str, str]
    build_spec_hash: str
    samples: list[SampleRecord]
    splits: dict[str, list[str]]
    stats: dict[str, object]
    qa_summary: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != "surrogate_dataset/v3":
            raise ValueError(
                f"schema_version must be 'surrogate_dataset/v3', got {self.schema_version}"
            )
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        if self.storage_mode not in {"linked", "packed"}:
            raise ValueError(f"invalid storage_mode: {self.storage_mode}")
        if self.task_kind not in {"sdf", "mesh", "hybrid"}:
            raise ValueError(f"invalid task_kind: {self.task_kind}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "profile_id": self.profile_id,
            "created_at": self.created_at,
            "generator_version": self.generator_version,
            "git_commit": self.git_commit,
            "storage_mode": self.storage_mode,
            "task_kind": self.task_kind,
            "materials": dict(self.materials),
            "grid3d": dict(self.grid3d),
            "observers": list(self.observers),
            "param_spec_hash": self.param_spec_hash,
            "observer_spec_hashes": dict(self.observer_spec_hashes),
            "build_spec_hash": self.build_spec_hash,
            "samples": [sample.to_dict() for sample in self.samples],
            "splits": {str(k): list(v) for k, v in self.splits.items()},
            "stats": dict(self.stats),
            "qa_summary": dict(self.qa_summary),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetManifest:
        return cls(
            schema_version=str(raw["schema_version"]),
            dataset_id=str(raw["dataset_id"]),
            profile_id=str(raw["profile_id"]),
            created_at=str(raw["created_at"]),
            generator_version=str(raw["generator_version"]),
            git_commit=str(raw["git_commit"]),
            storage_mode=raw["storage_mode"],
            task_kind=raw["task_kind"],
            materials={str(k): v for k, v in dict(raw.get("materials", {})).items()},
            grid3d={str(k): v for k, v in dict(raw.get("grid3d", {})).items()},
            observers=[str(v) for v in list(raw.get("observers", []))],
            param_spec_hash=str(raw.get("param_spec_hash", "")),
            observer_spec_hashes={
                str(k): str(v) for k, v in dict(raw.get("observer_spec_hashes", {})).items()
            },
            build_spec_hash=str(raw.get("build_spec_hash", "")),
            samples=[
                SampleRecord.from_dict(dict(item))
                for item in list(raw.get("samples", []))
                if isinstance(item, dict)
            ],
            splits={
                str(k): [str(v) for v in list(values)]
                for k, values in dict(raw.get("splits", {})).items()
            },
            stats={str(k): v for k, v in dict(raw.get("stats", {})).items()},
            qa_summary={str(k): v for k, v in dict(raw.get("qa_summary", {})).items()},
        )
