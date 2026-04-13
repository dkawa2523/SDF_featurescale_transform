from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from wafergeo.compare.batch_transform_runner import (
    read_transform_index,
    transform_spec_from_index_row,
)
from wafergeo.compare.feature_taxonomy import classify_feature
from wafergeo.compare.runner import run_transform_spec
from wafergeo.compare.runtime_io import resolve_path, write_json, write_run_info
from wafergeo.compare.schema import load_transform_eval_spec_yaml
from wafergeo.compare.schema_types import FeatureSpec
from wafergeo.compare.transform_eval_figures import write_transform_eval_figures

EVAL_ARRAY_NAMES = {
    "sdf_nm",
    "tsdf_10nm",
    "tsdf_30nm",
    "tsdf_100nm",
    "log_abs_sdf",
    "udf_nm",
    "mask",
    "changed_sdf_nm",
    "etched_sdf_nm",
    "deposited_sdf_nm",
    "material_changed_sdf_nm",
    "changed_mask",
    "etched_mask",
    "deposited_mask",
    "material_changed_mask",
    "interface_distance_nm",
    "nearest_material_id",
    "second_material_id",
    "pair_code",
    "distance_gap_nm",
    "interface_band_10nm",
    "interface_band_30nm",
    "interface_band_100nm",
    "transition_code",
    "transition_distance_nm",
    "transition_band_10nm",
    "transition_band_30nm",
    "transition_band_100nm",
}


def _safe_execution_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if not safe or safe in {".", ".."}:
        raise ValueError(f"invalid transform-eval feature label: {value!r}")
    return safe


def _dir_size_mb(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return float(total) / (1024.0 * 1024.0)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _clean_eval_output_dirs(out_dir: Path) -> None:
    for name in ("eval_features", "figures"):
        path = out_dir / name
        if path.exists():
            shutil.rmtree(path)
    for name in (
        "feature_profile_values.csv",
        "profile_variation_summary.csv",
    ):
        path = out_dir / name
        if path.exists():
            path.unlink()


def _array_stats(array: np.ndarray) -> dict[str, object]:
    values = np.asarray(array)
    finite = values[np.isfinite(values)]
    contiguous = np.ascontiguousarray(values)
    row: dict[str, object] = {
        "shape": json.dumps([int(v) for v in values.shape], separators=(",", ":")),
        "dtype": str(values.dtype),
        "array_hash": hashlib.sha256(contiguous.tobytes()).hexdigest()[:16],
        "nan_count": int(np.isnan(values).sum()) if np.issubdtype(values.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(values).sum()) if np.issubdtype(values.dtype, np.floating) else 0,
    }
    if finite.size:
        row.update(
            {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean": float(finite.mean()),
                "std": float(finite.std()),
            }
        )
    return row


def _feature_summary_items(summary_path: Path) -> list[dict[str, object]]:
    if not summary_path.exists():
        return []
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if not isinstance(features, list):
        return []
    return [item for item in features if isinstance(item, dict)]


def _feature_stats_rows(
    *,
    execution_label: str,
    case_id: str,
    case_out: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in _feature_summary_items(case_out / "feature_summary.json"):
        feature_name = str(feature.get("name", ""))
        taxonomy = classify_feature(feature_name)
        rel_path = str(feature.get("path", ""))
        feature_path = case_out / "features" / rel_path
        if feature_path.suffix != ".npz" or not feature_path.exists():
            continue
        with np.load(feature_path, allow_pickle=False) as data:
            for array_name in data.files:
                if array_name not in EVAL_ARRAY_NAMES:
                    continue
                values = np.asarray(data[array_name])
                if not np.issubdtype(values.dtype, np.number):
                    continue
                rows.append(
                    {
                        "execution_label": execution_label,
                        "target_shape": taxonomy.target_shape,
                        "method": taxonomy.method,
                        "code_name": feature_name,
                        "case_id": case_id,
                        "array_name": array_name,
                        "path": rel_path,
                        "semantics": feature.get("semantics", ""),
                        "size_mb": feature.get("size_mb", ""),
                        **_array_stats(values),
                    }
                )
    return rows


def _material_coverage_rows(
    *,
    execution_label: str,
    case_id: str,
    case_out: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in _feature_summary_items(case_out / "feature_summary.json"):
        feature_name = str(feature.get("name", ""))
        taxonomy = classify_feature(feature_name)
        rel_path = str(feature.get("path", ""))
        feature_path = case_out / "features" / rel_path
        if feature_path.suffix != ".npz" or not feature_path.exists():
            continue
        with np.load(feature_path, allow_pickle=False) as data:
            if "material_ids" not in data.files or "voxel_counts" not in data.files:
                continue
            material_ids = [int(v) for v in np.asarray(data["material_ids"]).tolist()]
            voxel_counts = [int(v) for v in np.asarray(data["voxel_counts"]).tolist()]
        total = float(sum(voxel_counts))
        for material_id, voxel_count in zip(material_ids, voxel_counts, strict=True):
            rows.append(
                {
                    "execution_label": execution_label,
                    "target_shape": taxonomy.target_shape,
                    "method": taxonomy.method,
                    "code_name": feature_name,
                    "case_id": case_id,
                    "material_id": material_id,
                    "voxel_count": voxel_count,
                    "voxel_fraction": (float(voxel_count) / total) if total else 0.0,
                }
            )
    return rows


def _summary_scalar_rows(
    *,
    execution_label: str,
    case_id: str,
    case_out: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in _feature_summary_items(case_out / "feature_summary.json"):
        feature_name = str(feature.get("name", ""))
        taxonomy = classify_feature(feature_name)
        outputs = feature.get("outputs", {})
        if not isinstance(outputs, dict):
            continue
        summary_rel = outputs.get("summary")
        if not isinstance(summary_rel, str) or not summary_rel:
            continue
        summary_path = case_out / "features" / summary_rel
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for scalar, value in sorted(payload.items()):
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            rows.append(
                {
                    "execution_label": execution_label,
                    "target_shape": taxonomy.target_shape,
                    "method": taxonomy.method,
                    "code_name": feature_name,
                    "case_id": case_id,
                    "path": summary_rel,
                    "scalar": scalar,
                    "value": float(value),
                }
            )
    return rows


def _float_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in ("", None):
            continue
        if isinstance(value, int | float | str):
            values.append(float(value))
    return values


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        return int(value)
    return 0


def _case_variation_rows(
    feature_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in feature_rows:
        key = (str(row["execution_label"]), str(row["code_name"]), str(row["array_name"]))
        groups.setdefault(key, []).append(row)

    rows: list[dict[str, object]] = []
    variable_array_count_by_execution_label: dict[str, int] = {}
    for (execution_label, code_name, array_name), items in sorted(groups.items()):
        taxonomy = classify_feature(code_name)
        hashes = {str(item.get("array_hash", "")) for item in items}
        unique_array_count = len(hashes)
        varies = unique_array_count > 1
        if varies:
            variable_array_count_by_execution_label[execution_label] = (
                variable_array_count_by_execution_label.get(execution_label, 0) + 1
            )
        means = _float_values(items, "mean")
        mins = _float_values(items, "min")
        maxs = _float_values(items, "max")
        rows.append(
            {
                "execution_label": execution_label,
                "target_shape": taxonomy.target_shape,
                "method": taxonomy.method,
                "code_name": code_name,
                "array_name": array_name,
                "case_count": len(items),
                "unique_array_count": unique_array_count,
                "varies": str(varies).lower(),
                "mean_min": min(means) if means else "",
                "mean_max": max(means) if means else "",
                "mean_range": (max(means) - min(means)) if means else "",
                "value_min": min(mins) if mins else "",
                "value_max": max(maxs) if maxs else "",
            }
        )
    return rows, variable_array_count_by_execution_label


def _scalar_variation_rows(
    scalar_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in scalar_rows:
        key = (str(row["execution_label"]), str(row["code_name"]), str(row["scalar"]))
        groups.setdefault(key, []).append(row)

    rows: list[dict[str, object]] = []
    variable_scalar_count_by_execution_label: dict[str, int] = {}
    for (execution_label, code_name, scalar), items in sorted(groups.items()):
        taxonomy = classify_feature(code_name)
        values = _float_values(items, "value")
        unique_value_count = len({f"{value:.12g}" for value in values})
        varies = unique_value_count > 1
        if varies:
            variable_scalar_count_by_execution_label[execution_label] = (
                variable_scalar_count_by_execution_label.get(execution_label, 0) + 1
            )
        rows.append(
            {
                "execution_label": execution_label,
                "target_shape": taxonomy.target_shape,
                "method": taxonomy.method,
                "code_name": code_name,
                "scalar": scalar,
                "case_count": len(items),
                "unique_value_count": unique_value_count,
                "varies": str(varies).lower(),
                "value_min": min(values) if values else "",
                "value_max": max(values) if values else "",
                "value_range": (max(values) - min(values)) if values else "",
            }
        )
    return rows, variable_scalar_count_by_execution_label


def _append_limited(values: list[str], value: str, *, limit: int = 12) -> None:
    if value in values:
        return
    if len(values) < limit:
        values.append(value)
    elif len(values) == limit:
        values.append("...")


def _eval_feature_signal_rows(
    *,
    eval_feature_summary_rows: list[dict[str, object]],
    variation_rows: list[dict[str, object]],
    scalar_variation_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    varying_arrays: dict[str, list[str]] = {}
    varying_scalars: dict[str, list[str]] = {}

    for row in variation_rows:
        if row.get("varies") != "true":
            continue
        execution_label = str(row["execution_label"])
        item = f"{row['code_name']}.{row['array_name']}"
        _append_limited(varying_arrays.setdefault(execution_label, []), item)

    for row in scalar_variation_rows:
        if row.get("varies") != "true":
            continue
        execution_label = str(row["execution_label"])
        item = f"{row['code_name']}.{row['scalar']}"
        _append_limited(varying_scalars.setdefault(execution_label, []), item)

    rows: list[dict[str, object]] = []
    for source in eval_feature_summary_rows:
        execution_label = str(source["execution_label"])
        array_count = _int_value(source.get("variable_array_count", 0))
        scalar_count = _int_value(source.get("variable_scalar_count", 0))
        total = array_count + scalar_count
        rows.append(
            {
                "execution_label": execution_label,
                "target_shape": source.get("target_shape", ""),
                "method": source.get("method", ""),
                "code_name": source.get("code_name", ""),
                "case_count": source.get("case_count", ""),
                "signal_status": "varies" if total else "constant",
                "varying_output_count": total,
                "varying_array_count": array_count,
                "varying_scalar_count": scalar_count,
                "mean_runtime_sec": source.get("mean_runtime_sec", ""),
                "mean_size_mb": source.get("mean_size_mb", ""),
                "varying_arrays": "|".join(varying_arrays.get(execution_label, [])),
                "varying_scalars": "|".join(varying_scalars.get(execution_label, [])),
            }
        )
    return rows


def run_transform_eval_from_config(config_path: str | Path) -> dict[str, object]:
    config_file = Path(config_path).resolve()
    spec = load_transform_eval_spec_yaml(config_file)
    base_dir = config_file.parent
    index_path = resolve_path(spec.index, base_dir=base_dir)
    index_dir = index_path.parent
    out_dir = resolve_path(spec.output.dir, base_dir=base_dir)
    eval_features_dir = out_dir / "eval_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_eval_output_dirs(out_dir)
    eval_features_dir.mkdir(parents=True, exist_ok=True)

    rows = read_transform_index(index_path)
    resolved_rows: list[dict[str, str]] = []
    for row in rows:
        resolved = dict(row)
        resolved["input_path"] = str(resolve_path(resolved["input_path"], base_dir=index_dir))
        if resolved.get("reference_path"):
            resolved["reference_path"] = str(
                resolve_path(resolved["reference_path"], base_dir=index_dir)
            )
        resolved_rows.append(resolved)
    eval_feature_summary_rows: list[dict[str, object]] = []
    case_summary_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    material_rows: list[dict[str, object]] = []
    scalar_rows: list[dict[str, object]] = []
    seen_execution_ids: dict[str, str] = {}

    for eval_feature in spec.features:
        execution_label = f"{eval_feature.target_shape}_{eval_feature.method}"
        execution_id = _safe_execution_id(execution_label)
        existing_name = seen_execution_ids.get(execution_id)
        if existing_name is not None:
            raise ValueError(
                "transform-eval feature labels collide after path sanitization: "
                f"{existing_name!r} and {execution_label!r} -> {execution_id!r}"
            )
        seen_execution_ids[execution_id] = execution_label
        feature_dir = eval_features_dir / eval_feature.target_shape / eval_feature.method
        feature_spec = FeatureSpec(use=(eval_feature.code_name,))
        total_runtime_sec = 0.0
        total_size_mb = 0.0
        output_feature_count = 0

        for row in resolved_rows:
            case_id = row["case_id"]
            case_row = dict(row)
            case_out = feature_dir / "cases" / case_id
            transform_spec = transform_spec_from_index_row(
                case_row,
                batch_spec_features=feature_spec,
                batch_spec_process=spec.process,
                batch_spec_view=spec.view,
                output_dir=case_out,
            )

            start = time.perf_counter()
            run_transform_spec(
                spec=transform_spec,
                config_file=config_file,
                base_dir=index_dir,
                output_dir=case_out,
                write_run_metadata=False,
            )
            runtime_sec = time.perf_counter() - start
            size_mb = _dir_size_mb(case_out)
            total_runtime_sec += runtime_sec
            total_size_mb += size_mb

            case_summary_rows.append(
                {
                    "execution_label": execution_label,
                    "target_shape": eval_feature.target_shape,
                    "method": eval_feature.method,
                    "code_name": eval_feature.code_name,
                    "case_id": case_id,
                    "runtime_sec": runtime_sec,
                    "output_size_mb": size_mb,
                    "output_dir": str(case_out.relative_to(out_dir)),
                }
            )
            feature_items = _feature_summary_items(case_out / "feature_summary.json")
            output_feature_count += len(feature_items)
            feature_rows.extend(
                _feature_stats_rows(
                    execution_label=execution_label,
                    case_id=case_id,
                    case_out=case_out,
                )
            )
            material_rows.extend(
                _material_coverage_rows(
                    execution_label=execution_label,
                    case_id=case_id,
                    case_out=case_out,
                )
            )
            scalar_rows.extend(
                _summary_scalar_rows(
                    execution_label=execution_label,
                    case_id=case_id,
                    case_out=case_out,
                )
            )
        case_count = len(rows)
        eval_feature_summary_rows.append(
            {
                "execution_label": execution_label,
                "target_shape": eval_feature.target_shape,
                "method": eval_feature.method,
                "code_name": eval_feature.code_name,
                "case_count": case_count,
                "output_feature_count": output_feature_count,
                "total_runtime_sec": total_runtime_sec,
                "mean_runtime_sec": total_runtime_sec / case_count,
                "total_size_mb": total_size_mb,
                "mean_size_mb": total_size_mb / case_count,
            }
        )

    variation_rows, variable_array_count_by_execution_label = _case_variation_rows(feature_rows)
    scalar_variation_rows, variable_scalar_count_by_execution_label = _scalar_variation_rows(
        scalar_rows
    )
    for eval_feature_row in eval_feature_summary_rows:
        execution_label = str(eval_feature_row["execution_label"])
        eval_feature_row["variable_array_count"] = variable_array_count_by_execution_label.get(
            execution_label,
            0,
        )
        eval_feature_row["variable_scalar_count"] = variable_scalar_count_by_execution_label.get(
            execution_label,
            0,
        )
    eval_feature_signal_rows = _eval_feature_signal_rows(
        eval_feature_summary_rows=eval_feature_summary_rows,
        variation_rows=variation_rows,
        scalar_variation_rows=scalar_variation_rows,
    )

    _write_csv(
        out_dir / "eval_feature_summary.csv",
        eval_feature_summary_rows,
        [
            "execution_label",
            "target_shape",
            "method",
            "code_name",
            "case_count",
            "output_feature_count",
            "total_runtime_sec",
            "mean_runtime_sec",
            "total_size_mb",
            "mean_size_mb",
            "variable_array_count",
            "variable_scalar_count",
        ],
    )
    _write_csv(
        out_dir / "eval_feature_signal.csv",
        eval_feature_signal_rows,
        [
            "execution_label",
            "target_shape",
            "method",
            "code_name",
            "case_count",
            "signal_status",
            "varying_output_count",
            "varying_array_count",
            "varying_scalar_count",
            "mean_runtime_sec",
            "mean_size_mb",
            "varying_arrays",
            "varying_scalars",
        ],
    )
    write_json(out_dir / "eval_feature_signal.json", {"features": eval_feature_signal_rows})
    _write_csv(
        out_dir / "case_summary.csv",
        case_summary_rows,
        [
            "execution_label",
            "target_shape",
            "method",
            "code_name",
            "case_id",
            "runtime_sec",
            "output_size_mb",
            "output_dir",
        ],
    )
    _write_csv(
        out_dir / "feature_stats.csv",
        feature_rows,
        [
            "execution_label",
            "target_shape",
            "method",
            "code_name",
            "case_id",
            "array_name",
            "path",
            "semantics",
            "size_mb",
            "shape",
            "dtype",
            "array_hash",
            "min",
            "max",
            "mean",
            "std",
            "nan_count",
            "inf_count",
        ],
    )
    _write_csv(
        out_dir / "case_variation_summary.csv",
        variation_rows,
        [
            "execution_label",
            "target_shape",
            "method",
            "code_name",
            "array_name",
            "case_count",
            "unique_array_count",
            "varies",
            "mean_min",
            "mean_max",
            "mean_range",
            "value_min",
            "value_max",
        ],
    )
    _write_csv(
        out_dir / "material_coverage.csv",
        material_rows,
        [
            "execution_label",
            "target_shape",
            "method",
            "code_name",
            "case_id",
            "material_id",
            "voxel_count",
            "voxel_fraction",
        ],
    )
    _write_csv(
        out_dir / "feature_scalar_summary.csv",
        scalar_rows,
        [
            "execution_label",
            "target_shape",
            "method",
            "code_name",
            "case_id",
            "path",
            "scalar",
            "value",
        ],
    )
    _write_csv(
        out_dir / "scalar_variation_summary.csv",
        scalar_variation_rows,
        [
            "execution_label",
            "target_shape",
            "method",
            "code_name",
            "scalar",
            "case_count",
            "unique_value_count",
            "varies",
            "value_min",
            "value_max",
            "value_range",
        ],
    )
    figures_index = write_transform_eval_figures(
        out_dir=out_dir,
        view=spec.view,
        index_rows=resolved_rows,
        eval_feature_summary_rows=eval_feature_summary_rows,
        eval_feature_signal_rows=eval_feature_signal_rows,
        variation_rows=variation_rows,
        scalar_variation_rows=scalar_variation_rows,
        case_summary_rows=case_summary_rows,
    )
    summary: dict[str, object] = {
        "task": "transform-eval",
        "status": "OK",
        "case_count": len(rows),
        "eval_feature_count": len(spec.features),
        "eval_features": [
            {
                "target_shape": item.target_shape,
                "method": item.method,
                "code_name": item.code_name,
            }
            for item in spec.features
        ],
        "figures": figures_index,
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "summary.json", summary)
    write_run_info(
        config_path=config_file,
        output_dir=out_dir,
        task="transform-eval",
        inputs={"index": str(index_path)},
    )
    return summary
