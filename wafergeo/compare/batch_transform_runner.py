from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import cast

from wafergeo.compare.runner import run_transform_spec
from wafergeo.compare.runtime_io import resolve_path, write_json, write_run_info
from wafergeo.compare.schema import (
    FeatureSpec,
    OutputSpec,
    ProcessSpec,
    SimulationInputSpec,
    SimulationKind,
    TransformSpec,
    ViewSpec,
    load_batch_transform_spec_yaml,
)


def _safe_case_id(value: str, *, row_number: int) -> str:
    raw = value.strip() or f"case_{row_number:04d}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    if not safe:
        safe = f"case_{row_number:04d}"
    if safe in {".", ".."}:
        raise ValueError(f"invalid case_id at row {row_number}: {value!r}")
    return safe


def read_transform_index(index_path: Path) -> list[dict[str, str]]:
    required = {"case_id", "input_kind", "input_path"}
    rows: list[dict[str, str]] = []
    with index_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(f"batch transform index is missing required columns: {missing}")
        seen_case_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=1):
            cleaned = {str(k): str(v or "").strip() for k, v in row.items() if k is not None}
            case_id = _safe_case_id(cleaned.get("case_id", ""), row_number=row_number)
            if case_id in seen_case_ids:
                raise ValueError(f"duplicate case_id in batch transform index: {case_id}")
            seen_case_ids.add(case_id)
            cleaned["case_id"] = case_id
            for key in required:
                if not cleaned.get(key):
                    raise ValueError(f"batch transform index row {row_number} has empty {key}")
            rows.append(cleaned)
    if not rows:
        raise ValueError("batch transform index has no rows")
    return rows


def transform_spec_from_index_row(
    row: dict[str, str],
    *,
    batch_spec_features: FeatureSpec,
    batch_spec_process: ProcessSpec | None = None,
    batch_spec_view: ViewSpec,
    output_dir: Path,
) -> TransformSpec:
    reference: SimulationInputSpec | None = None
    if row.get("reference_kind") or row.get("reference_path"):
        if not row.get("reference_kind") or not row.get("reference_path"):
            raise ValueError("reference_kind and reference_path must be specified together")
        reference = SimulationInputSpec(
            kind=cast(SimulationKind, row["reference_kind"]),
            path=row["reference_path"],
            void_id=None if not row.get("reference_void_id") else int(row["reference_void_id"]),
        )
    return TransformSpec(
        task="transform",
        simulation=SimulationInputSpec(
            kind=cast(SimulationKind, row["input_kind"]),
            path=row["input_path"],
            void_id=None if not row.get("void_id") else int(row["void_id"]),
        ),
        view=batch_spec_view,
        features=batch_spec_features,
        output=OutputSpec(dir=str(output_dir)),
        reference=reference,
        process=ProcessSpec() if batch_spec_process is None else batch_spec_process,
    )


def _write_dataset_index(
    output_dir: Path,
    rows: list[dict[str, str]],
) -> None:
    fieldnames = ["case_id", "input_kind", "input_path", "output_dir"]
    if any(row.get("reference_path") for row in rows):
        fieldnames[3:3] = ["reference_kind", "reference_path"]
    with (output_dir / "dataset_index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def feature_rows_from_summary(case_id: str, summary_path: Path) -> list[dict[str, object]]:
    if not summary_path.exists():
        return []
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if not isinstance(features, list):
        return []

    rows: list[dict[str, object]] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        array = feature.get("array", {})
        if not isinstance(array, dict):
            array = {}
        rows.append(
            {
                "case_id": case_id,
                "feature": feature.get("name", ""),
                "path": feature.get("path", ""),
                "semantics": feature.get("semantics", ""),
                "size_mb": feature.get("size_mb", ""),
                "shape": json.dumps(array.get("shape", []), separators=(",", ":")),
                "dtype": array.get("dtype", ""),
                "min": array.get("min", ""),
                "max": array.get("max", ""),
                "mean": array.get("mean", ""),
                "std": array.get("std", ""),
                "nan_count": array.get("nan_count", ""),
                "inf_count": array.get("inf_count", ""),
            }
        )
    return rows


def _write_features_summary(output_dir: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "case_id",
        "feature",
        "path",
        "semantics",
        "size_mb",
        "shape",
        "dtype",
        "min",
        "max",
        "mean",
        "std",
        "nan_count",
        "inf_count",
    ]
    with (output_dir / "features_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_batch_transform_from_config(config_path: str | Path) -> dict[str, object]:
    config_file = Path(config_path).resolve()
    spec = load_batch_transform_spec_yaml(config_file)
    base_dir = config_file.parent
    index_path = resolve_path(spec.index, base_dir=base_dir)
    index_dir = index_path.parent
    out_dir = resolve_path(spec.output.dir, base_dir=base_dir)
    cases_dir = out_dir / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    rows = read_transform_index(index_path)
    dataset_rows: list[dict[str, str]] = []
    feature_rows: list[dict[str, object]] = []

    for row in rows:
        case_id = row["case_id"]
        case_out = cases_dir / case_id
        case_row = dict(row)
        case_row["input_path"] = str(resolve_path(case_row["input_path"], base_dir=index_dir))
        if case_row.get("reference_path"):
            case_row["reference_path"] = str(
                resolve_path(case_row["reference_path"], base_dir=index_dir)
            )
        transform_spec = transform_spec_from_index_row(
            case_row,
            batch_spec_features=spec.features,
            batch_spec_process=spec.process,
            batch_spec_view=spec.view,
            output_dir=case_out,
        )
        run_transform_spec(
            spec=transform_spec,
            config_file=config_file,
            base_dir=index_dir,
            output_dir=case_out,
            write_run_metadata=False,
        )
        dataset_rows.append(
            {
                "case_id": case_id,
                "input_kind": case_row["input_kind"],
                "input_path": case_row["input_path"],
                "reference_kind": case_row.get("reference_kind", ""),
                "reference_path": case_row.get("reference_path", ""),
                "output_dir": str(case_out.relative_to(out_dir)),
            }
        )
        feature_rows.extend(feature_rows_from_summary(case_id, case_out / "feature_summary.json"))

    _write_dataset_index(out_dir, dataset_rows)
    _write_features_summary(out_dir, feature_rows)
    summary: dict[str, object] = {
        "task": "batch-transform",
        "status": "OK",
        "case_count": len(rows),
        "features": list(spec.features.use),
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "summary.json", summary)
    write_run_info(
        config_path=config_file,
        output_dir=out_dir,
        task="batch-transform",
        inputs={"index": str(index_path)},
    )
    return summary
