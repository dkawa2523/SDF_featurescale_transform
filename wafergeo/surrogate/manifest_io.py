from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wafergeo.surrogate.schema import DatasetManifest


@dataclass(frozen=True)
class SampleInputRow:
    sample_id: str
    group_id: str
    recipe_params: dict[str, object] = field(default_factory=dict)
    param_vector: list[float] | None = None
    label_artifact_id: str | None = None
    tsdf_artifact_id: str | None = None
    mesh_artifact_id: str | None = None
    obs2d_sim_ids: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if not self.group_id:
            raise ValueError("group_id must be non-empty")



def _ensure_mapping(raw: object, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")
    return {str(k): v for k, v in raw.items()}


def _to_input_row(raw: dict[str, Any]) -> SampleInputRow:
    return SampleInputRow(
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
    )


def load_input_manifest_json(path: str | Path) -> list[SampleInputRow]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    rows_raw: list[Any]
    if isinstance(payload, list):
        rows_raw = payload
    elif isinstance(payload, dict):
        maybe_rows = payload.get("samples")
        if not isinstance(maybe_rows, list):
            raise ValueError("input manifest object must contain list key 'samples'")
        rows_raw = maybe_rows
    else:
        raise ValueError("input manifest root must be list or mapping")

    rows: list[SampleInputRow] = []
    for idx, item in enumerate(rows_raw):
        row_map = _ensure_mapping(item, label=f"samples[{idx}]")
        rows.append(_to_input_row(row_map))
    return rows


def write_dataset_manifest(path: str | Path, manifest: DatasetManifest) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_dataset_manifest(path: str | Path) -> DatasetManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest root must be a mapping")
    return DatasetManifest.from_dict({str(k): v for k, v in payload.items()})
