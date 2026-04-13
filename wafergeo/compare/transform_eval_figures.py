from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from wafergeo.compare.eval_figure_common import (
    SCORE_EMPTY,
    TRANSFORM_FIGURE_NOTES,
    _as_float,
    _boundary_mask,
    _clamp01,
    _feature_summary_items,
    _imshow_or_note,
    _iou,
    _load_matplotlib_pyplot,
    _mask_for_array,
    _mean,
    _ok_figure_index,
    _optional_int,
    _rel,
    _safe_file_stem,
    _shape_section_xz,
    _shape_section_yz,
    _write_csv,
)
from wafergeo.compare.feature_taxonomy import classify_feature
from wafergeo.compare.features import extract_view_feature
from wafergeo.compare.loader import load_simulation_label
from wafergeo.compare.schema_types import ViewSpec
from wafergeo.core.types import LabelVolume


def _feature_axes(feature: str) -> tuple[str, str, str]:
    item = classify_feature(feature)
    return item.method, item.target_shape, item.output_kind


def _figure_axis_for_feature(feature: str) -> tuple[str, str, str, str]:
    item = classify_feature(feature)
    if item.role == "derived_relation":
        return item.target_shape, "relations", item.method, item.role
    return item.target_shape, "methods", item.method, item.role


def _variation_lookup(*row_groups: list[dict[str, object]]) -> set[tuple[str, str]]:
    lookup: set[tuple[str, str]] = set()
    for rows in row_groups:
        for row in rows:
            if str(row.get("varies", "")).lower() != "true":
                continue
            execution_label = str(row.get("execution_label", ""))
            code_name = str(row.get("code_name", ""))
            if execution_label and code_name:
                lookup.add((execution_label, code_name))
    return lookup


def _material_boundary_mask(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    boundary = np.zeros(arr.shape, dtype=bool)
    for axis in range(arr.ndim):
        left = [slice(None)] * arr.ndim
        right = [slice(None)] * arr.ndim
        left[axis] = slice(1, None)
        right[axis] = slice(None, -1)
        diff = arr[tuple(left)] != arr[tuple(right)]
        boundary[tuple(left)] |= diff
        boundary[tuple(right)] |= diff
    return boundary


def _feature_alignment_scores(
    *,
    feature_name: str,
    feature_path: Path,
    label: LabelVolume,
    view_mask: np.ndarray,
    reference_label: LabelVolume | None,
) -> dict[str, float | str]:
    scores: dict[str, float | str] = {
        "shape_match": SCORE_EMPTY,
        "boundary_match": SCORE_EMPTY,
        "interface_match": SCORE_EMPTY,
        "transition_match": SCORE_EMPTY,
    }
    if feature_path.suffix != ".npz" or not feature_path.exists():
        return scores
    with np.load(feature_path, allow_pickle=False) as data:
        if feature_name in {"material_sdf", "material_tsdf_views"} and "sdf_nm" in data.files:
            sdf = np.asarray(data["sdf_nm"])
            material_ids = [int(v) for v in np.asarray(data["material_ids"]).tolist()]
            per_material: list[float] = []
            for idx, material_id in enumerate(material_ids):
                per_material.append(_iou(sdf[idx] <= 0.0, label.material_id == material_id))
            scores["shape_match"] = float(np.mean(per_material)) if per_material else SCORE_EMPTY
            return scores
        if feature_name == "material_udf" and "udf_nm" in data.files:
            udf = np.asarray(data["udf_nm"])
            material_ids = [int(v) for v in np.asarray(data["material_ids"]).tolist()]
            threshold = min(float(v) for v in label.grid.spacing)
            per_material_boundary: list[float] = []
            for idx, material_id in enumerate(material_ids):
                target = label.material_id == material_id
                per_material_boundary.append(_iou(udf[idx] <= threshold, _boundary_mask(target)))
            scores["boundary_match"] = (
                float(np.mean(per_material_boundary)) if per_material_boundary else SCORE_EMPTY
            )
            return scores
        if feature_name == "material_interface_relation" and "interface_distance_nm" in data.files:
            threshold = min(float(v) for v in label.grid.spacing)
            derived = np.asarray(data["interface_distance_nm"]) <= threshold
            scores["interface_match"] = _iou(derived, _material_boundary_mask(label.material_id))
            return scores
        if feature_name in {"process_delta_sdf", "process_delta_tsdf_views"} and (
            "changed_mask" in data.files
        ):
            if reference_label is not None:
                changed = np.asarray(label.material_id) != np.asarray(reference_label.material_id)
                scores["shape_match"] = _iou(
                    np.asarray(data["changed_mask"], dtype=bool),
                    changed,
                )
            return scores
        if feature_name == "process_delta_udf" and "udf_nm" in data.files:
            if reference_label is not None:
                changed = np.asarray(label.material_id) != np.asarray(reference_label.material_id)
                threshold = min(float(v) for v in label.grid.spacing)
                udf = np.asarray(data["udf_nm"])
                scores["boundary_match"] = _iou(udf[0] <= threshold, _boundary_mask(changed))
            return scores
        if feature_name == "process_transition_relation" and "changed_mask" in data.files:
            if reference_label is not None:
                changed = np.asarray(label.material_id) != np.asarray(reference_label.material_id)
                scores["transition_match"] = _iou(
                    np.asarray(data["changed_mask"], dtype=bool),
                    changed,
                )
            return scores
        if feature_name == "udf" and "udf_nm" in data.files:
            udf = np.asarray(data["udf_nm"])
            threshold = min(float(v) for v in label.grid.spacing)
            derived = udf <= threshold
            target_mask = _mask_for_array(label, view_mask, tuple(udf.shape))
            scores["boundary_match"] = _iou(derived, _boundary_mask(target_mask))
            return scores

        array_name = "sdf_nm" if "sdf_nm" in data.files else ""
        if not array_name and "tsdf" in data.files:
            array_name = "tsdf"
        if not array_name:
            return scores
        field = np.asarray(data[array_name])
        derived = field <= 0.0
        target = _mask_for_array(label, view_mask, tuple(field.shape))
        scores["shape_match"] = _mean(
            (
                _iou(derived, target),
                _iou(_boundary_mask(derived), _boundary_mask(target)),
            )
        )
        return scores


def _alignment_rows(
    *,
    out_dir: Path,
    view: ViewSpec,
    index_rows: list[dict[str, str]],
    case_summary_rows: list[dict[str, object]],
    variation_rows: list[dict[str, object]],
    scalar_variation_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows_by_case = {row["case_id"]: row for row in index_rows}
    varies = _variation_lookup(variation_rows, scalar_variation_rows)
    label_cache: dict[str, tuple[LabelVolume, LabelVolume | None, np.ndarray]] = {}
    score_rows: list[dict[str, object]] = []

    for case_row in case_summary_rows:
        execution_label = str(case_row["execution_label"])
        case_id = str(case_row["case_id"])
        index_row = rows_by_case.get(case_id)
        if index_row is None:
            continue
        if case_id not in label_cache:
            label = load_simulation_label(
                index_row["input_kind"],
                Path(index_row["input_path"]),
                void_id=_optional_int(index_row.get("void_id")),
            )
            reference_label = None
            if index_row.get("reference_path") and index_row.get("reference_kind"):
                reference_label = load_simulation_label(
                    index_row["reference_kind"],
                    Path(index_row["reference_path"]),
                    void_id=_optional_int(index_row.get("reference_void_id")),
                )
            view_feature = extract_view_feature(
                label,
                axes=view.axes,
                depth_axis=view.depth_axis,
                contour_mode="material",
            )
            label_cache[case_id] = (label, reference_label, view_feature.mask)
        label, reference_label, view_mask = label_cache[case_id]
        case_out = out_dir / str(case_row["output_dir"])
        for feature in _feature_summary_items(case_out):
            feature_name = str(feature.get("name", ""))
            rel_path = str(feature.get("path", ""))
            feature_path = case_out / "features" / rel_path
            scores = _feature_alignment_scores(
                feature_name=feature_name,
                feature_path=feature_path,
                label=label,
                view_mask=view_mask,
                reference_label=reference_label,
            )
            alignment_values = [
                value for value in scores.values() if isinstance(value, int | float)
            ]
            alignment_score = _mean(float(value) for value in alignment_values)
            size_mb = _as_float(feature.get("size_mb"))
            score_rows.append(
                {
                    "execution_label": execution_label,
                    "case_id": case_id,
                    "code_name": feature_name,
                    "path": rel_path,
                    **scores,
                    "alignment_score": alignment_score,
                    "variation_capture": (
                        1.0 if (execution_label, feature_name) in varies else 0.0
                    ),
                    "compactness": _clamp01(1.0 / (1.0 + size_mb)),
                }
            )
    return score_rows


def _target_score_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    target_rows: list[dict[str, object]] = []
    for row in rows:
        code_name = str(row.get("code_name", ""))
        method, target_shape, output_kind = _feature_axes(code_name)
        taxonomy = classify_feature(code_name)
        relation = method if taxonomy.role == "derived_relation" else ""
        display_method = "" if taxonomy.role == "derived_relation" else method
        target_rows.append(
            {
                "execution_label": row.get("execution_label", ""),
                "case_id": row.get("case_id", ""),
                "code_name": code_name,
                "target_shape": target_shape,
                "method": display_method,
                "relation": relation,
                "role": taxonomy.role,
                "output_kind": output_kind,
                "shape_match": row.get("shape_match", SCORE_EMPTY),
                "boundary_match": row.get("boundary_match", SCORE_EMPTY),
                "interface_match": row.get("interface_match", SCORE_EMPTY),
                "transition_match": row.get("transition_match", SCORE_EMPTY),
                "case_sensitivity": row.get("variation_capture", SCORE_EMPTY),
                "data_cost": row.get("compactness", SCORE_EMPTY),
            }
        )
    return target_rows


def _field_sections_xz_yz(array: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    arr = np.asarray(array)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    return arr[:, arr.shape[1] // 2, :], arr[:, :, arr.shape[2] // 2]


def _decoded_material_sdf(
    feature_path: Path,
    label: LabelVolume,
) -> tuple[np.ndarray, list[int]] | None:
    with np.load(feature_path, allow_pickle=False) as data:
        if "sdf_nm" not in data.files or "material_ids" not in data.files:
            return None
        sdf = np.asarray(data["sdf_nm"])
        material_ids = [int(value) for value in np.asarray(data["material_ids"]).tolist()]
    if sdf.ndim != 4 or len(material_ids) != sdf.shape[0]:
        return None
    inside = sdf <= 0.0
    any_inside = np.any(inside, axis=0)
    inside_distance = np.where(inside, sdf, np.inf)
    winner = np.argmin(inside_distance, axis=0)
    decoded = np.full(sdf.shape[1:], int(label.material.void_id), dtype=label.material_id.dtype)
    material_id_array = np.asarray(material_ids, dtype=label.material_id.dtype)
    decoded[any_inside] = material_id_array[winner[any_inside]]
    return decoded, material_ids


def _plot_input_shape_sections(
    *,
    plt: Any,
    path: Path,
    index_rows: list[dict[str, str]],
    max_cases: int = 12,
) -> bool:
    rows = index_rows[:max_cases]
    if not rows:
        return False
    fig, axes = plt.subplots(
        len(rows),
        2,
        figsize=(7.0, max(3.5, 2.0 * len(rows))),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, row in enumerate(rows):
        label = load_simulation_label(
            row["input_kind"],
            Path(row["input_path"]),
            void_id=_optional_int(row.get("void_id")),
        )
        case_id = row["case_id"]
        axes[row_idx][0].imshow(_shape_section_xz(label), cmap="tab20", origin="lower")
        axes[row_idx][0].set_title(f"{case_id} [x,z] mid-y")
        axes[row_idx][1].imshow(_shape_section_yz(label), cmap="tab20", origin="lower")
        axes[row_idx][1].set_title(f"{case_id} [y,z] mid-x")
        for axis in axes[row_idx]:
            axis.set_xlabel("horizontal index")
            axis.set_ylabel("z index")
    if len(index_rows) > max_cases:
        fig.suptitle(f"input shape sections, first {max_cases} of {len(index_rows)} cases")
    else:
        fig.suptitle("input shape sections")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _representative_feature_row(rows: list[dict[str, object]]) -> dict[str, object]:
    scored = [
        row
        for row in rows
        if isinstance(row.get("alignment_score"), int | float)
    ]
    if scored:
        return min(scored, key=lambda row: _as_float(row.get("alignment_score"), 1.0))
    return max(rows, key=lambda row: _as_float(row.get("variation_capture")))


def _numeric_npz_vector(feature_name: str, path: Path) -> np.ndarray | None:
    arrays_by_feature = {
        "sdf_raw": ("sdf_nm",),
        "tsdf_views": ("tsdf_10nm", "tsdf_30nm", "tsdf_100nm"),
        "udf": ("udf_nm",),
        "material_sdf": ("sdf_nm",),
        "material_tsdf_views": ("tsdf_10nm", "tsdf_30nm", "tsdf_100nm"),
        "material_udf": ("udf_nm",),
        "material_interface_relation": (
            "interface_distance_nm",
            "pair_code",
            "distance_gap_nm",
        ),
        "process_delta_sdf": (
            "changed_sdf_nm",
            "etched_sdf_nm",
            "deposited_sdf_nm",
            "material_changed_sdf_nm",
        ),
        "process_delta_tsdf_views": ("tsdf_10nm", "tsdf_30nm", "tsdf_100nm"),
        "process_delta_udf": ("udf_nm",),
        "process_transition_relation": (
            "transition_code",
            "transition_distance_nm",
        ),
    }
    with np.load(path, allow_pickle=False) as data:
        names = arrays_by_feature.get(feature_name)
        if names is None:
            names = tuple(name for name in data.files if np.issubdtype(data[name].dtype, np.number))
        parts: list[np.ndarray] = []
        for name in names:
            if name not in data.files:
                continue
            array = np.asarray(data[name], dtype=np.float32)
            if "sdf" in name:
                array = np.clip(array, -60.0, 60.0)
            elif name == "udf_nm":
                array = np.clip(array, 0.0, 60.0)
            parts.append(np.nan_to_num(array, nan=0.0, posinf=60.0, neginf=-60.0).ravel())
    if not parts:
        return None
    return np.concatenate(parts).astype(np.float32, copy=False)


def _numeric_feature_vector(feature_name: str, path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    if path.suffix == ".npz":
        return _numeric_npz_vector(feature_name, path)
    return None


def _feature_vectors_by_axis(
    *,
    out_dir: Path,
    case_summary_rows: list[dict[str, object]],
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    vectors: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for row in case_summary_rows:
        execution_label = str(row.get("execution_label", ""))
        case_id = str(row.get("case_id", ""))
        case_out = out_dir / str(row.get("output_dir", ""))
        if not execution_label or not case_id:
            continue
        for feature in _feature_summary_items(case_out):
            feature_name = str(feature.get("name", ""))
            feature_path = case_out / "features" / str(feature.get("path", ""))
            vector = _numeric_feature_vector(feature_name, feature_path)
            if vector is None:
                continue
            vectors.setdefault((execution_label, feature_name), {})[case_id] = vector
    return vectors


def _distance_matrix_for_vectors(
    vectors: Mapping[str, np.ndarray],
    case_order: Sequence[str],
) -> np.ndarray | None:
    present = [case_id for case_id in case_order if case_id in vectors]
    if len(present) < 2:
        return None
    max_len = max(int(vectors[case_id].size) for case_id in present)
    if max_len == 0:
        return None
    matrix = np.zeros((len(present), max_len), dtype=np.float32)
    for row_idx, case_id in enumerate(present):
        vector = vectors[case_id].astype(np.float32, copy=False)
        matrix[row_idx, : vector.size] = vector
    scale = float(np.std(matrix))
    if scale > 0.0 and math.isfinite(scale):
        matrix = matrix / scale
    distances = np.zeros((len(case_order), len(case_order)), dtype=np.float32)
    present_index = {case_id: idx for idx, case_id in enumerate(present)}
    for left_idx, left_case in enumerate(case_order):
        if left_case not in present_index:
            continue
        left = matrix[present_index[left_case]]
        for right_idx, right_case in enumerate(case_order):
            if right_case not in present_index:
                continue
            diff = left - matrix[present_index[right_case]]
            distances[left_idx, right_idx] = float(np.sqrt(np.mean(diff * diff)))
    return distances


def _case_distance_outputs(
    *,
    out_dir: Path,
    index_rows: list[dict[str, str]],
    case_summary_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[str], list[str], dict[str, np.ndarray]]:
    case_order = [row["case_id"] for row in index_rows]
    method_vectors = _feature_vectors_by_axis(
        out_dir=out_dir,
        case_summary_rows=case_summary_rows,
    )
    rows: list[dict[str, object]] = []
    labels: list[str] = []
    distance_vectors: dict[str, np.ndarray] = {}
    for (execution_label, feature), vectors in method_vectors.items():
        distances = _distance_matrix_for_vectors(vectors, case_order)
        if distances is None:
            continue
        method, target_shape, _ = _feature_axes(feature)
        taxonomy = classify_feature(feature)
        relation = method if taxonomy.role == "derived_relation" else ""
        display_method = "" if taxonomy.role == "derived_relation" else method
        label_axis = f"relation={relation}" if relation else f"method={method}"
        label = f"target_shape={target_shape}\n{label_axis}\ncode={feature}"
        labels.append(label)
        upper = distances[np.triu_indices(len(case_order), k=1)]
        distance_vectors[label] = upper.astype(np.float32, copy=False)
        for left_idx, left_case in enumerate(case_order):
            for right_idx, right_case in enumerate(case_order):
                if right_idx <= left_idx:
                    continue
                rows.append(
                    {
                        "execution_label": execution_label,
                        "code_name": feature,
                        "target_shape": target_shape,
                        "method": display_method,
                        "relation": relation,
                        "role": taxonomy.role,
                        "case_a": left_case,
                        "case_b": right_case,
                        "normalized_rmse": float(distances[left_idx, right_idx]),
                    }
                )
    return rows, case_order, labels, distance_vectors


def _distance_correlation_outputs(
    distance_vectors: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, object]], list[str], np.ndarray]:
    labels = list(distance_vectors)
    matrix = np.eye(len(labels), dtype=np.float32)
    rows: list[dict[str, object]] = []
    for left_idx, left_label in enumerate(labels):
        left = distance_vectors[left_label]
        for right_idx, right_label in enumerate(labels):
            right = distance_vectors[right_label]
            corr = 0.0
            has_variation = (
                left.size
                and right.size
                and float(np.std(left)) > 0.0
                and float(np.std(right)) > 0.0
            )
            if has_variation:
                corr = float(np.corrcoef(left, right)[0, 1])
            if left_idx == right_idx:
                corr = 1.0
            matrix[left_idx, right_idx] = corr
            rows.append(
                {
                    "left": left_label.replace("\n", " / "),
                    "right": right_label.replace("\n", " / "),
                    "distance_correlation": corr,
                }
            )
    return rows, labels, matrix


def _mean_score(rows: list[dict[str, object]], column: str) -> float | str:
    values = [
        _as_float(row.get(column))
        for row in rows
        if isinstance(row.get(column), int | float)
    ]
    if not values:
        return SCORE_EMPTY
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _feature_sections_for_report(
    *,
    feature_name: str,
    feature_path: Path,
    label: LabelVolume,
) -> tuple[np.ndarray | None, np.ndarray | None, str, str]:
    if not feature_path.exists():
        raise FileNotFoundError(f"feature file does not exist: {feature_path}")
    if feature_path.suffix != ".npz":
        raise ValueError(f"feature figure requires npz, got: {feature_path}")
    with np.load(feature_path, allow_pickle=False) as data:
        if feature_name == "material_sdf":
            decoded_result = _decoded_material_sdf(feature_path, label)
            if decoded_result is None:
                raise ValueError(f"material_sdf cannot be decoded: {feature_path}")
            decoded, _ = decoded_result
            decoded_label = LabelVolume(
                grid=label.grid,
                material=label.material,
                material_id=decoded.astype(label.material_id.dtype, copy=False),
                meta=label.meta,
            )
            return (
                _shape_section_xz(decoded_label),
                _shape_section_yz(decoded_label),
                "tab20",
                "decoded material labels",
            )
        if feature_name == "material_interface_relation" and "interface_distance_nm" in data.files:
            field = np.clip(
                np.asarray(data["interface_distance_nm"], dtype=np.float32),
                0.0,
                60.0,
            )
            sections = _field_sections_xz_yz(field)
            if sections is None:
                raise ValueError(
                    f"material_interface_relation is not a 3D field: {feature_path}"
                )
            return sections[0], sections[1], "viridis", "nearest material-interface distance"
        if feature_name == "process_delta_sdf" and "changed_sdf_nm" in data.files:
            field = np.asarray(data["changed_sdf_nm"], dtype=np.float32)
            clipped = np.clip(field, -60.0, 60.0)
            sections = _field_sections_xz_yz(clipped)
            if sections is None:
                raise ValueError(f"process_delta_sdf is not a 3D field: {feature_path}")
            return sections[0], sections[1], "coolwarm", "process delta SDF"
        if feature_name == "process_transition_relation" and "transition_code" in data.files:
            field = np.asarray(data["transition_code"], dtype=np.float32)
            sections = _field_sections_xz_yz(field)
            if sections is None:
                raise ValueError(
                    f"process_transition_relation is not a 3D field: {feature_path}"
                )
            return sections[0], sections[1], "tab20", "reference-to-final material transition"
        if feature_name == "udf" and "udf_nm" in data.files:
            field = np.clip(np.asarray(data["udf_nm"], dtype=np.float32), 0.0, 60.0)
            sections = _field_sections_xz_yz(field)
            if sections is None:
                raise ValueError(f"udf is not a 3D field: {feature_path}")
            return sections[0], sections[1], "viridis", "boundary distance"
        if feature_name in {"material_udf", "process_delta_udf"} and "udf_nm" in data.files:
            field = np.clip(np.asarray(data["udf_nm"], dtype=np.float32), 0.0, 60.0)
            sections = _field_sections_xz_yz(field)
            if sections is None:
                raise ValueError(f"{feature_name} is not a drawable field: {feature_path}")
            return sections[0], sections[1], "viridis", "boundary distance"
        array_name = "sdf_nm" if "sdf_nm" in data.files else ""
        if not array_name:
            raise ValueError(f"feature npz does not contain a drawable field: {feature_path}")
        field = np.clip(np.asarray(data[array_name], dtype=np.float32), -60.0, 60.0)
        sections = _field_sections_xz_yz(field)
        if sections is None:
            raise ValueError(f"SDF feature is not a 3D field: {feature_path}")
        return sections[0], sections[1], "coolwarm", "SDF field"


def _target_sections(
    *,
    feature_name: str,
    label: LabelVolume,
    reference_label: LabelVolume | None,
) -> tuple[np.ndarray | None, np.ndarray | None, str, str]:
    taxonomy = classify_feature(feature_name)
    if taxonomy.target_shape == "material_shape":
        return (
            _shape_section_xz(label),
            _shape_section_yz(label),
            "tab20",
            "target: material shapes",
        )
    if taxonomy.target_shape == "process_delta_shape":
        if reference_label is None:
            raise ValueError(
                "process-delta figures require reference_kind and reference_path"
            )
        changed = np.asarray(label.material_id) != np.asarray(reference_label.material_id)
        return (
            changed[:, changed.shape[1] // 2, :],
            changed[:, :, changed.shape[2] // 2],
            "gray",
            "target: process-delta shape",
        )
    mask = np.asarray(label.material_id) != int(label.material.void_id)
    return (
        mask[:, mask.shape[1] // 2, :],
        mask[:, :, mask.shape[2] // 2],
        "gray",
        "target: full shape",
    )


def _material_feature_arrays(
    *,
    feature_name: str,
    feature_path: Path,
) -> tuple[list[int], list[tuple[str, np.ndarray, str, float, float]]]:
    with np.load(feature_path, allow_pickle=False) as data:
        material_ids = [int(value) for value in np.asarray(data["material_ids"]).tolist()]
        if feature_name == "material_sdf":
            return material_ids, [
                ("sdf_nm", np.asarray(data["sdf_nm"], dtype=np.float32), "coolwarm", -60.0, 60.0)
            ]
        if feature_name == "material_tsdf_views":
            return material_ids, [
                (
                    "tsdf_10nm",
                    np.asarray(data["tsdf_10nm"], dtype=np.float32),
                    "coolwarm",
                    -1.0,
                    1.0,
                ),
                (
                    "tsdf_30nm",
                    np.asarray(data["tsdf_30nm"], dtype=np.float32),
                    "coolwarm",
                    -1.0,
                    1.0,
                ),
                (
                    "tsdf_100nm",
                    np.asarray(data["tsdf_100nm"], dtype=np.float32),
                    "coolwarm",
                    -1.0,
                    1.0,
                ),
            ]
        if feature_name == "material_udf":
            return material_ids, [
                ("udf_nm", np.asarray(data["udf_nm"], dtype=np.float32), "viridis", 0.0, 60.0)
            ]
    raise ValueError(f"not a material field feature: {feature_name}")


def _plot_material_field_report(
    *,
    plt: Any,
    path: Path,
    feature_name: str,
    feature_path: Path,
    label: LabelVolume,
    case_id: str,
) -> bool:
    material_ids, arrays = _material_feature_arrays(
        feature_name=feature_name,
        feature_path=feature_path,
    )
    if not material_ids or not arrays:
        return False
    column_count = 2 + (2 * len(arrays))
    row_count = len(material_ids)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(max(8.0, 2.0 * column_count), max(2.8, 2.0 * row_count)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, material_id in enumerate(material_ids):
        target_mask = np.asarray(label.material_id) == material_id
        target_xz = target_mask[:, target_mask.shape[1] // 2, :]
        target_yz = target_mask[:, :, target_mask.shape[2] // 2]
        _imshow_or_note(
            axes[row_idx][0],
            target_xz,
            f"material {material_id} target [x,z]",
            cmap="gray",
        )
        _imshow_or_note(
            axes[row_idx][1],
            target_yz,
            f"material {material_id} target [y,z]",
            cmap="gray",
        )
        col = 2
        for array_name, values, cmap, vmin, vmax in arrays:
            if row_idx >= values.shape[0]:
                continue
            field = np.clip(values[row_idx], vmin, vmax)
            xz = field[:, field.shape[1] // 2, :]
            yz = field[:, :, field.shape[2] // 2]
            _imshow_or_note(
                axes[row_idx][col],
                xz,
                f"{array_name} [x,z]",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            _imshow_or_note(
                axes[row_idx][col + 1],
                yz,
                f"{array_name} [y,z]",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            col += 2
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"target_shape=material_shape / code_name={feature_name} / case={case_id}: "
        "all material channels"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_method_field_report(
    *,
    plt: Any,
    path: Path,
    row: dict[str, object],
    out_dir: Path,
    index_rows: dict[str, dict[str, str]],
    case_summary_rows: list[dict[str, object]],
) -> bool:
    case_id = str(row["case_id"])
    index_row = index_rows.get(case_id)
    if index_row is None:
        return False
    case_summary = next(
        (
            item
            for item in case_summary_rows
            if item.get("execution_label") == row.get("execution_label")
            and item.get("case_id") == case_id
        ),
        None,
    )
    if case_summary is None:
        return False
    label = load_simulation_label(
        index_row["input_kind"],
        Path(index_row["input_path"]),
        void_id=_optional_int(index_row.get("void_id")),
    )
    reference_label = None
    if index_row.get("reference_path") and index_row.get("reference_kind"):
        reference_label = load_simulation_label(
            index_row["reference_kind"],
            Path(index_row["reference_path"]),
            void_id=_optional_int(index_row.get("reference_void_id")),
        )

    case_out = out_dir / str(case_summary["output_dir"])
    feature_name = str(row.get("code_name", ""))
    method, target_shape, _ = _feature_axes(feature_name)
    taxonomy = classify_feature(feature_name)
    axis_label = (
        f"relation={method}" if taxonomy.role == "derived_relation" else f"method={method}"
    )
    feature_path = case_out / "features" / str(row.get("path", ""))
    if feature_name in {"material_sdf", "material_tsdf_views", "material_udf"}:
        return _plot_material_field_report(
            plt=plt,
            path=path,
            feature_name=feature_name,
            feature_path=feature_path,
            label=label,
            case_id=case_id,
        )
    feature_xz, feature_yz, cmap, feature_note = _feature_sections_for_report(
        feature_name=feature_name,
        feature_path=feature_path,
        label=label,
    )
    target_xz, target_yz, target_cmap, target_note = _target_sections(
        feature_name=feature_name,
        label=label,
        reference_label=reference_label,
    )

    fig, axes = plt.subplots(3, 2, figsize=(8.0, 10.0), constrained_layout=True)
    _imshow_or_note(axes[0][0], _shape_section_xz(label), "final material [x,z]", cmap="tab20")
    _imshow_or_note(axes[0][1], _shape_section_yz(label), "final material [y,z]", cmap="tab20")
    _imshow_or_note(axes[1][0], target_xz, f"{target_shape} [x,z]", cmap=target_cmap)
    _imshow_or_note(axes[1][1], target_yz, f"{target_shape} [y,z]", cmap=target_cmap)
    _imshow_or_note(axes[2][0], feature_xz, f"{method} field [x,z]", cmap=cmap)
    _imshow_or_note(axes[2][1], feature_yz, f"{method} field [y,z]", cmap=cmap)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"target_shape={target_shape} / {axis_label} / "
        f"code_name={feature_name} / case={case_id}: "
        f"{feature_note}; {target_note}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_method_scores(
    *,
    plt: Any,
    path: Path,
    rows: list[dict[str, object]],
    title: str,
) -> bool:
    score_columns = [
        "shape_match",
        "boundary_match",
        "interface_match",
        "transition_match",
        "case_sensitivity",
        "data_cost",
    ]
    labels: list[str] = []
    values: list[float] = []
    for column in score_columns:
        value = _mean_score(rows, column)
        if isinstance(value, int | float):
            labels.append(column)
            values.append(float(value))
    if not values:
        return False
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    ax.bar(np.arange(len(values)), values, color="#4C78A8")
    ax.set_ylim(0.0, max(1.05, max(values) * 1.15))
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=30, ha="right")
    ax.set_ylabel("score / normalized distance")
    ax.set_title(title)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_method_case_distance(
    *,
    plt: Any,
    path: Path,
    rows: list[dict[str, object]],
    case_order: Sequence[str],
    title: str,
) -> bool:
    if not rows or len(case_order) < 2:
        return False
    fig, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)
    case_index = {case_id: idx for idx, case_id in enumerate(case_order)}
    matrix = np.zeros((len(case_order), len(case_order)), dtype=np.float32)
    for row in rows:
        left = str(row.get("case_a", ""))
        right = str(row.get("case_b", ""))
        if left not in case_index or right not in case_index:
            continue
        value = _as_float(row.get("normalized_rmse"))
        matrix[case_index[left], case_index[right]] = value
        matrix[case_index[right], case_index[left]] = value
    vmax = float(np.max(matrix))
    vmax = vmax if vmax > 0.0 else 1.0
    image = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(case_order)), labels=case_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(case_order)), labels=case_order)
    fig.colorbar(image, ax=ax, shrink=0.8, label="normalized RMSE")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def write_transform_eval_figures(
    *,
    out_dir: Path,
    view: ViewSpec,
    index_rows: list[dict[str, str]],
    eval_feature_summary_rows: list[dict[str, object]],
    eval_feature_signal_rows: list[dict[str, object]],
    variation_rows: list[dict[str, object]],
    scalar_variation_rows: list[dict[str, object]],
    case_summary_rows: list[dict[str, object]],
) -> dict[str, object]:
    del eval_feature_summary_rows
    plt = _load_matplotlib_pyplot()
    figures_dir = out_dir / "figures"

    figures: list[str] = []
    data: list[str] = []
    score_rows = _alignment_rows(
        out_dir=out_dir,
        view=view,
        index_rows=index_rows,
        case_summary_rows=case_summary_rows,
        variation_rows=variation_rows,
        scalar_variation_rows=scalar_variation_rows,
    )
    target_rows = _target_score_rows(score_rows)
    target_csv = figures_dir / "feature_scores.csv"
    target_fields = [
        "execution_label",
        "case_id",
        "code_name",
        "target_shape",
        "method",
        "relation",
        "role",
        "output_kind",
        "shape_match",
        "boundary_match",
        "interface_match",
        "transition_match",
        "case_sensitivity",
        "data_cost",
    ]
    _write_csv(target_csv, target_rows, target_fields)
    data.append(_rel(target_csv, out_dir))

    shape_sections = figures_dir / "input_shape_sections.png"
    if _plot_input_shape_sections(plt=plt, path=shape_sections, index_rows=index_rows):
        figures.append(_rel(shape_sections, out_dir))
    distance_rows, case_order, method_labels, distance_vectors = _case_distance_outputs(
        out_dir=out_dir,
        index_rows=index_rows,
        case_summary_rows=case_summary_rows,
    )
    if distance_rows:
        distance_csv = figures_dir / "case_distance.csv"
        _write_csv(
            distance_csv,
            distance_rows,
            [
                "execution_label",
                "code_name",
                "target_shape",
                "method",
                "relation",
                "role",
                "case_a",
                "case_b",
                "normalized_rmse",
            ],
        )
        data.append(_rel(distance_csv, out_dir))
    del method_labels
    corr_rows, corr_labels, corr_matrix = _distance_correlation_outputs(distance_vectors)
    if corr_rows:
        corr_csv = figures_dir / "distance_correlation.csv"
        _write_csv(
            corr_csv,
            corr_rows,
            ["left", "right", "distance_correlation"],
        )
        data.append(_rel(corr_csv, out_dir))
    del corr_labels, corr_matrix
    del eval_feature_signal_rows, variation_rows, scalar_variation_rows

    index_by_case = {row["case_id"]: row for row in index_rows}
    target_rows_by_axis: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in target_rows:
        feature = str(row.get("code_name", ""))
        target_shape, group_kind, axis_name, _ = _figure_axis_for_feature(feature)
        key = (target_shape, group_kind, axis_name)
        target_rows_by_axis.setdefault(key, []).append(row)
    distance_rows_by_axis: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in distance_rows:
        feature = str(row.get("code_name", ""))
        target_shape, group_kind, axis_name, _ = _figure_axis_for_feature(feature)
        key = (target_shape, group_kind, axis_name)
        distance_rows_by_axis.setdefault(key, []).append(row)
    feature_groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in score_rows:
        feature = str(row.get("code_name", ""))
        target_shape, group_kind, axis_name, _ = _figure_axis_for_feature(feature)
        key = (target_shape, group_kind, axis_name)
        feature_groups.setdefault(key, []).append(row)
    for key, rows in feature_groups.items():
        representative = _representative_feature_row(rows)
        target_shape, group_kind, axis_name = key
        method_dir = figures_dir / "by_target_shape" / _safe_file_stem(target_shape)
        if group_kind == "relations":
            method_dir = method_dir / "relations" / _safe_file_stem(axis_name)
            title_axis = f"target_shape={target_shape} / relation={axis_name}"
        else:
            method_dir = method_dir / _safe_file_stem(axis_name)
            title_axis = f"target_shape={target_shape} / method={axis_name}"
        field_path = method_dir / "field.png"
        if _plot_method_field_report(
            plt=plt,
            path=field_path,
            row=representative,
            out_dir=out_dir,
            index_rows=index_by_case,
            case_summary_rows=case_summary_rows,
        ):
            figures.append(_rel(field_path, out_dir))
        score_path = method_dir / "scores.png"
        if _plot_method_scores(
            plt=plt,
            path=score_path,
            rows=target_rows_by_axis.get(key, []),
            title=f"{title_axis} scores",
        ):
            figures.append(_rel(score_path, out_dir))
        distance_path = method_dir / "case_distance.png"
        if _plot_method_case_distance(
            plt=plt,
            path=distance_path,
            rows=distance_rows_by_axis.get(key, []),
            case_order=case_order,
            title=f"{title_axis} case distance",
        ):
            figures.append(_rel(distance_path, out_dir))

    return _ok_figure_index(out_dir, figures, data, TRANSFORM_FIGURE_NOTES)


