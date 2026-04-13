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
from wafergeo.compare.eval_figures import write_transform_eval_figures
from wafergeo.compare.runner import run_transform_spec
from wafergeo.compare.runtime_io import resolve_path, write_json, write_run_info
from wafergeo.compare.schema import load_transform_eval_spec_yaml

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
}


def _safe_candidate_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if not safe or safe in {".", ".."}:
        raise ValueError(f"invalid transform-eval candidate name: {value!r}")
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
    for name in ("candidates", "figures"):
        path = out_dir / name
        if path.exists():
            shutil.rmtree(path)


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
    candidate: str,
    case_id: str,
    case_out: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in _feature_summary_items(case_out / "feature_summary.json"):
        feature_name = str(feature.get("name", ""))
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
                        "candidate": candidate,
                        "case_id": case_id,
                        "feature": feature_name,
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
    candidate: str,
    case_id: str,
    case_out: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in _feature_summary_items(case_out / "feature_summary.json"):
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
                    "candidate": candidate,
                    "case_id": case_id,
                    "feature": feature.get("name", ""),
                    "material_id": material_id,
                    "voxel_count": voxel_count,
                    "voxel_fraction": (float(voxel_count) / total) if total else 0.0,
                }
            )
    return rows


def _summary_scalar_rows(
    *,
    candidate: str,
    case_id: str,
    case_out: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in _feature_summary_items(case_out / "feature_summary.json"):
        feature_name = str(feature.get("name", ""))
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
                    "candidate": candidate,
                    "case_id": case_id,
                    "feature": feature_name,
                    "path": summary_rel,
                    "scalar": scalar,
                    "value": float(value),
                }
            )
    return rows


def _try_float(value: object) -> float | None:
    if value in ("", None):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _profile_key(row: dict[str, str]) -> tuple[str, str] | None:
    if row.get("transition_key"):
        return "transition_key", row["transition_key"]
    if row.get("material_id"):
        return "material_id", row["material_id"]
    return None


def _profile_value_rows(
    *,
    candidate: str,
    case_id: str,
    case_out: Path,
) -> list[dict[str, object]]:
    skip_columns = {
        "transition_key",
        "change_type",
        "material_id",
        "material_name",
        "is_void",
        "initial_material_id",
        "initial_material_name",
        "final_material_id",
        "final_material_name",
    }
    rows: list[dict[str, object]] = []
    for feature in _feature_summary_items(case_out / "feature_summary.json"):
        feature_name = str(feature.get("name", ""))
        outputs = feature.get("outputs", {})
        if not isinstance(outputs, dict):
            continue
        profile_rel = outputs.get("profile")
        if not isinstance(profile_rel, str) or not profile_rel:
            continue
        profile_path = case_out / "features" / profile_rel
        if not profile_path.exists():
            continue
        with profile_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                cleaned = {str(k): str(v or "").strip() for k, v in row.items() if k}
                key = _profile_key(cleaned)
                if key is None:
                    continue
                key_type, key_value = key
                for scalar, raw_value in sorted(cleaned.items()):
                    if scalar in skip_columns:
                        continue
                    value = _try_float(raw_value)
                    if value is None:
                        continue
                    rows.append(
                        {
                            "candidate": candidate,
                            "case_id": case_id,
                            "feature": feature_name,
                            "path": profile_rel,
                            "key_type": key_type,
                            "key": key_value,
                            "scalar": scalar,
                            "value": value,
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
        key = (str(row["candidate"]), str(row["feature"]), str(row["array_name"]))
        groups.setdefault(key, []).append(row)

    rows: list[dict[str, object]] = []
    variable_array_count_by_candidate: dict[str, int] = {}
    for (candidate, feature, array_name), items in sorted(groups.items()):
        hashes = {str(item.get("array_hash", "")) for item in items}
        unique_array_count = len(hashes)
        varies = unique_array_count > 1
        if varies:
            variable_array_count_by_candidate[candidate] = (
                variable_array_count_by_candidate.get(candidate, 0) + 1
            )
        means = _float_values(items, "mean")
        mins = _float_values(items, "min")
        maxs = _float_values(items, "max")
        rows.append(
            {
                "candidate": candidate,
                "feature": feature,
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
    return rows, variable_array_count_by_candidate


def _scalar_variation_rows(
    scalar_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in scalar_rows:
        key = (str(row["candidate"]), str(row["feature"]), str(row["scalar"]))
        groups.setdefault(key, []).append(row)

    rows: list[dict[str, object]] = []
    variable_scalar_count_by_candidate: dict[str, int] = {}
    for (candidate, feature, scalar), items in sorted(groups.items()):
        values = _float_values(items, "value")
        unique_value_count = len({f"{value:.12g}" for value in values})
        varies = unique_value_count > 1
        if varies:
            variable_scalar_count_by_candidate[candidate] = (
                variable_scalar_count_by_candidate.get(candidate, 0) + 1
            )
        rows.append(
            {
                "candidate": candidate,
                "feature": feature,
                "scalar": scalar,
                "case_count": len(items),
                "unique_value_count": unique_value_count,
                "varies": str(varies).lower(),
                "value_min": min(values) if values else "",
                "value_max": max(values) if values else "",
                "value_range": (max(values) - min(values)) if values else "",
            }
        )
    return rows, variable_scalar_count_by_candidate


def _profile_variation_rows(
    profile_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = {}
    for row in profile_rows:
        key = (
            str(row["candidate"]),
            str(row["feature"]),
            str(row["key_type"]),
            str(row["key"]),
            str(row["scalar"]),
        )
        groups.setdefault(key, []).append(row)

    rows: list[dict[str, object]] = []
    variable_profile_count_by_candidate: dict[str, int] = {}
    for (candidate, feature, key_type, key_value, scalar), items in sorted(groups.items()):
        values = _float_values(items, "value")
        unique_value_count = len({f"{value:.12g}" for value in values})
        varies = unique_value_count > 1
        if varies:
            variable_profile_count_by_candidate[candidate] = (
                variable_profile_count_by_candidate.get(candidate, 0) + 1
            )
        rows.append(
            {
                "candidate": candidate,
                "feature": feature,
                "key_type": key_type,
                "key": key_value,
                "scalar": scalar,
                "case_count": len(items),
                "unique_value_count": unique_value_count,
                "varies": str(varies).lower(),
                "value_min": min(values) if values else "",
                "value_max": max(values) if values else "",
                "value_range": (max(values) - min(values)) if values else "",
            }
        )
    return rows, variable_profile_count_by_candidate


def _append_limited(values: list[str], value: str, *, limit: int = 12) -> None:
    if value in values:
        return
    if len(values) < limit:
        values.append(value)
    elif len(values) == limit:
        values.append("...")


def _candidate_eval_summary_rows(
    *,
    candidate_summary_rows: list[dict[str, object]],
    variation_rows: list[dict[str, object]],
    scalar_variation_rows: list[dict[str, object]],
    profile_variation_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    varying_arrays: dict[str, list[str]] = {}
    varying_scalars: dict[str, list[str]] = {}
    varying_profiles: dict[str, list[str]] = {}

    for row in variation_rows:
        if row.get("varies") != "true":
            continue
        candidate = str(row["candidate"])
        item = f"{row['feature']}.{row['array_name']}"
        _append_limited(varying_arrays.setdefault(candidate, []), item)

    for row in scalar_variation_rows:
        if row.get("varies") != "true":
            continue
        candidate = str(row["candidate"])
        item = f"{row['feature']}.{row['scalar']}"
        _append_limited(varying_scalars.setdefault(candidate, []), item)

    for row in profile_variation_rows:
        if row.get("varies") != "true":
            continue
        candidate = str(row["candidate"])
        item = f"{row['feature']}.{row['key_type']}:{row['key']}.{row['scalar']}"
        _append_limited(varying_profiles.setdefault(candidate, []), item)

    rows: list[dict[str, object]] = []
    for source in candidate_summary_rows:
        candidate = str(source["candidate"])
        array_count = _int_value(source.get("variable_array_count", 0))
        scalar_count = _int_value(source.get("variable_scalar_count", 0))
        profile_count = _int_value(source.get("variable_profile_count", 0))
        total = array_count + scalar_count + profile_count
        rows.append(
            {
                "candidate": candidate,
                "features": source.get("features", ""),
                "case_count": source.get("case_count", ""),
                "signal_status": "varies" if total else "constant",
                "varying_output_count": total,
                "varying_array_count": array_count,
                "varying_scalar_count": scalar_count,
                "varying_profile_count": profile_count,
                "mean_runtime_sec": source.get("mean_runtime_sec", ""),
                "mean_size_mb": source.get("mean_size_mb", ""),
                "varying_arrays": "|".join(varying_arrays.get(candidate, [])),
                "varying_scalars": "|".join(varying_scalars.get(candidate, [])),
                "varying_profiles": "|".join(varying_profiles.get(candidate, [])),
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
    candidates_dir = out_dir / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_eval_output_dirs(out_dir)
    candidates_dir.mkdir(parents=True, exist_ok=True)

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
    candidate_summary_rows: list[dict[str, object]] = []
    case_summary_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    material_rows: list[dict[str, object]] = []
    scalar_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    seen_candidate_ids: dict[str, str] = {}

    for candidate_name, candidate_features in spec.candidates.items():
        candidate_id = _safe_candidate_id(candidate_name)
        existing_name = seen_candidate_ids.get(candidate_id)
        if existing_name is not None:
            raise ValueError(
                "transform-eval candidate names collide after path sanitization: "
                f"{existing_name!r} and {candidate_name!r} -> {candidate_id!r}"
            )
        seen_candidate_ids[candidate_id] = candidate_name
        candidate_dir = candidates_dir / candidate_id
        total_runtime_sec = 0.0
        total_size_mb = 0.0
        candidate_feature_count = 0

        for row in resolved_rows:
            case_id = row["case_id"]
            case_row = dict(row)
            case_out = candidate_dir / "cases" / case_id
            transform_spec = transform_spec_from_index_row(
                case_row,
                batch_spec_features=candidate_features,
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
                    "candidate": candidate_name,
                    "case_id": case_id,
                    "runtime_sec": runtime_sec,
                    "output_size_mb": size_mb,
                    "output_dir": str(case_out.relative_to(out_dir)),
                }
            )
            feature_items = _feature_summary_items(case_out / "feature_summary.json")
            candidate_feature_count += len(feature_items)
            feature_rows.extend(
                _feature_stats_rows(
                    candidate=candidate_name,
                    case_id=case_id,
                    case_out=case_out,
                )
            )
            material_rows.extend(
                _material_coverage_rows(
                    candidate=candidate_name,
                    case_id=case_id,
                    case_out=case_out,
                )
            )
            scalar_rows.extend(
                _summary_scalar_rows(
                    candidate=candidate_name,
                    case_id=case_id,
                    case_out=case_out,
                )
            )
            profile_rows.extend(
                _profile_value_rows(
                    candidate=candidate_name,
                    case_id=case_id,
                    case_out=case_out,
                )
            )

        case_count = len(rows)
        candidate_summary_rows.append(
            {
                "candidate": candidate_name,
                "features": "|".join(candidate_features.use),
                "case_count": case_count,
                "feature_count": candidate_feature_count,
                "total_runtime_sec": total_runtime_sec,
                "mean_runtime_sec": total_runtime_sec / case_count,
                "total_size_mb": total_size_mb,
                "mean_size_mb": total_size_mb / case_count,
            }
        )

    variation_rows, variable_array_count_by_candidate = _case_variation_rows(feature_rows)
    scalar_variation_rows, variable_scalar_count_by_candidate = _scalar_variation_rows(
        scalar_rows
    )
    profile_variation_rows, variable_profile_count_by_candidate = _profile_variation_rows(
        profile_rows
    )
    for candidate_row in candidate_summary_rows:
        candidate = str(candidate_row["candidate"])
        candidate_row["variable_array_count"] = variable_array_count_by_candidate.get(candidate, 0)
        candidate_row["variable_scalar_count"] = variable_scalar_count_by_candidate.get(
            candidate,
            0,
        )
        candidate_row["variable_profile_count"] = variable_profile_count_by_candidate.get(
            candidate,
            0,
        )
    candidate_eval_rows = _candidate_eval_summary_rows(
        candidate_summary_rows=candidate_summary_rows,
        variation_rows=variation_rows,
        scalar_variation_rows=scalar_variation_rows,
        profile_variation_rows=profile_variation_rows,
    )

    _write_csv(
        out_dir / "candidate_summary.csv",
        candidate_summary_rows,
        [
            "candidate",
            "features",
            "case_count",
            "feature_count",
            "total_runtime_sec",
            "mean_runtime_sec",
            "total_size_mb",
            "mean_size_mb",
            "variable_array_count",
            "variable_scalar_count",
            "variable_profile_count",
        ],
    )
    _write_csv(
        out_dir / "candidate_eval_summary.csv",
        candidate_eval_rows,
        [
            "candidate",
            "features",
            "case_count",
            "signal_status",
            "varying_output_count",
            "varying_array_count",
            "varying_scalar_count",
            "varying_profile_count",
            "mean_runtime_sec",
            "mean_size_mb",
            "varying_arrays",
            "varying_scalars",
            "varying_profiles",
        ],
    )
    write_json(out_dir / "candidate_eval_summary.json", {"candidates": candidate_eval_rows})
    _write_csv(
        out_dir / "case_summary.csv",
        case_summary_rows,
        ["candidate", "case_id", "runtime_sec", "output_size_mb", "output_dir"],
    )
    _write_csv(
        out_dir / "feature_stats.csv",
        feature_rows,
        [
            "candidate",
            "case_id",
            "feature",
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
            "candidate",
            "feature",
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
            "candidate",
            "case_id",
            "feature",
            "material_id",
            "voxel_count",
            "voxel_fraction",
        ],
    )
    _write_csv(
        out_dir / "feature_scalar_summary.csv",
        scalar_rows,
        ["candidate", "case_id", "feature", "path", "scalar", "value"],
    )
    _write_csv(
        out_dir / "scalar_variation_summary.csv",
        scalar_variation_rows,
        [
            "candidate",
            "feature",
            "scalar",
            "case_count",
            "unique_value_count",
            "varies",
            "value_min",
            "value_max",
            "value_range",
        ],
    )
    _write_csv(
        out_dir / "feature_profile_values.csv",
        profile_rows,
        ["candidate", "case_id", "feature", "path", "key_type", "key", "scalar", "value"],
    )
    _write_csv(
        out_dir / "profile_variation_summary.csv",
        profile_variation_rows,
        [
            "candidate",
            "feature",
            "key_type",
            "key",
            "scalar",
            "case_count",
            "unique_value_count",
            "varies",
            "value_min",
            "value_max",
            "value_range",
        ],
    )
    figures_manifest = write_transform_eval_figures(
        out_dir=out_dir,
        view=spec.view,
        index_rows=resolved_rows,
        candidate_summary_rows=candidate_summary_rows,
        candidate_eval_rows=candidate_eval_rows,
        variation_rows=variation_rows,
        scalar_variation_rows=scalar_variation_rows,
        profile_variation_rows=profile_variation_rows,
        case_summary_rows=case_summary_rows,
    )
    summary: dict[str, object] = {
        "task": "transform-eval",
        "status": "OK",
        "case_count": len(rows),
        "candidate_count": len(spec.candidates),
        "candidates": list(spec.candidates),
        "figures": figures_manifest,
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
