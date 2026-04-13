from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from wafergeo.compare.features import contour_feature_on_grid, extract_view_feature
from wafergeo.compare.loader import CONTOUR_LOADERS, load_simulation_label
from wafergeo.compare.runtime_io import resolve_path
from wafergeo.compare.schema_types import CompareEvalSpec, ViewSpec
from wafergeo.core.types import LabelVolume

FIGURE_MANIFEST = "figures/figure_manifest.json"
SCORE_EMPTY = ""
TRANSFORM_FIGURE_NOTES = {
    "input_shape_sections.png": (
        "Rows are cases. The left panel is the [x,z] mid-y section and the right panel "
        "is the [y,z] mid-x section of the original label volume."
    ),
    "feature_method_overview.png": (
        "Feature methods are grouped by average representation, variation, and compactness. "
        "Use it as the first-pass map of what each method contributes."
    ),
    "feature_representation_score.png": (
        "Higher bars mean the feature better preserves the input geometry or material/profile "
        "content. Empty alignment scores are excluded from the representation average."
    ),
    "candidate_signal_heatmap.png": (
        "Rows are candidates and columns are feature outputs or summary values. Bright cells "
        "mean that output changed across cases."
    ),
    "candidate_signal_cost.png": (
        "Points farther up have more varying outputs; points farther right write larger outputs."
    ),
    "representative_feature_slices": (
        "Each image shows the diagnostic view that matches the feature type. "
        "Material SDF panels decode per-material channels back into labels and show IoU "
        "by material. Profile panels compare CSV fractions and bounding ranges with the "
        "input label volume. Red/green/blue in alignment panels indicate mismatch types."
    ),
}
COMPARE_FIGURE_NOTES = {
    "objective_heatmap.png": (
        "Rows are metric candidates and columns are cases. Lower normalized_total_score is better."
    ),
    "rank_delta_heatmap.png": (
        "Color shows how each candidate changes case ranking relative to the first candidate."
    ),
    "metric_contribution_heatmap.png": (
        "Bright cells show which metrics contribute larger normalized loss for each candidate."
    ),
    "metric_evaluation_score.png": (
        "Higher bars mean more complete coverage, more stable ranking, stronger case separation, "
        "or better aggregate metric-set quality."
    ),
    "representative_differences": (
        "Representative cases show target, simulation, overlay+diff, and [x,z]/[y,z] "
        "sections for simulation and label targets when available. Yellow is overlap, "
        "red is simulation-only, green is target-only, blue is label mismatch."
    ),
}


def _load_matplotlib_pyplot() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _safe_file_stem(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if not safe or safe in {".", ".."}:
        return "figure"
    return safe


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _skip_manifest(out_dir: Path, reason: str) -> dict[str, object]:
    manifest: dict[str, object] = {
        "status": "SKIPPED",
        "reason": reason,
        "figures": [],
        "data": [],
    }
    _write_json(out_dir / FIGURE_MANIFEST, manifest)
    return {"status": "SKIPPED", "manifest": FIGURE_MANIFEST, "reason": reason}


def _write_figure_readme(out_dir: Path, notes: Mapping[str, object]) -> str:
    path = out_dir / "figures" / "README.md"
    lines = [
        "# Eval Figures",
        "",
        "These PNG files are diagnostic views. CSV/JSON/NPZ outputs remain the authoritative data.",
        "",
        "## How to read",
        "",
    ]
    for name, note in sorted(notes.items()):
        lines.append(f"- `{name}`: {note}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return _rel(path, out_dir)


def _ok_manifest(
    out_dir: Path,
    figures: list[str],
    data: list[str],
    notes: Mapping[str, object],
) -> dict[str, object]:
    guide = _write_figure_readme(out_dir, notes)
    manifest: dict[str, object] = {
        "status": "OK",
        "figures": figures,
        "data": data,
        "guide": guide,
        "how_to_read": dict(notes),
    }
    _write_json(out_dir / FIGURE_MANIFEST, manifest)
    return {
        "status": "OK",
        "manifest": FIGURE_MANIFEST,
        "figures": figures,
        "data": data,
        "guide": guide,
        "how_to_read": dict(notes),
    }


def _rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _optional_int(value: object) -> int | None:
    if value in ("", None):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    return None


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _mean(values: Iterable[float]) -> float | str:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return SCORE_EMPTY
    return float(np.mean(np.asarray(clean, dtype=np.float64)))


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    if a.shape != b.shape:
        return 0.0
    union = np.logical_or(a, b)
    if not np.any(union):
        return 1.0
    return float(np.logical_and(a, b).sum() / union.sum())


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.size == 0:
        return binary
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    center = padded[(slice(1, -1),) * binary.ndim]
    neighbor_all = center.copy()
    for axis in range(binary.ndim):
        before = [slice(1, -1)] * binary.ndim
        after = [slice(1, -1)] * binary.ndim
        before[axis] = slice(0, -2)
        after[axis] = slice(2, None)
        neighbor_all &= padded[tuple(before)] & padded[tuple(after)]
    return center & ~neighbor_all


def _mask_for_array(
    label: LabelVolume,
    view_mask: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    volume_mask = np.asarray(label.material_id) != int(label.material.void_id)
    if shape == tuple(view_mask.shape):
        return np.asarray(view_mask, dtype=bool)
    if shape == tuple(volume_mask.shape):
        return volume_mask
    return np.zeros(shape, dtype=bool)


def _mid_slice(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        return np.asarray(arr[arr.shape[0] // 2])
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    return np.asarray(arr).reshape(1, -1)


def _difference_rgb(sim: np.ndarray, target: np.ndarray) -> np.ndarray:
    sim_solid = np.asarray(sim) != 0
    target_solid = np.asarray(target) != 0
    rgb = np.full(sim_solid.shape + (3,), 245, dtype=np.uint8)
    same = sim_solid & target_solid & (np.asarray(sim) == np.asarray(target))
    mismatch = sim_solid & target_solid & (np.asarray(sim) != np.asarray(target))
    only_sim = sim_solid & ~target_solid
    only_target = target_solid & ~sim_solid
    rgb[same] = np.array([230, 190, 40], dtype=np.uint8)
    rgb[mismatch] = np.array([70, 110, 220], dtype=np.uint8)
    rgb[only_sim] = np.array([220, 70, 70], dtype=np.uint8)
    rgb[only_target] = np.array([70, 160, 80], dtype=np.uint8)
    return rgb


def _feature_summary_items(case_out: Path) -> list[dict[str, object]]:
    path = case_out / "feature_summary.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if not isinstance(features, list):
        return []
    return [item for item in features if isinstance(item, dict)]


def _plot_heatmap(
    *,
    plt: Any,
    path: Path,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    values: np.ndarray,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
) -> bool:
    if not row_labels or not col_labels:
        return False
    width = max(6.0, min(18.0, 1.0 + len(col_labels) * 0.45))
    height = max(3.5, min(12.0, 1.0 + len(row_labels) * 0.45))
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels, rotation=45, ha="right")
    fig.colorbar(image, ax=ax, shrink=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_grouped_bars(
    *,
    plt: Any,
    path: Path,
    labels: Sequence[str],
    series: Mapping[str, Sequence[float]],
    title: str,
    ylabel: str,
) -> bool:
    if not labels or not series:
        return False
    fig, ax = plt.subplots(
        figsize=(max(7.0, len(labels) * 0.8), 4.5),
        constrained_layout=True,
    )
    x = np.arange(len(labels), dtype=np.float64)
    names = list(series)
    width = min(0.8 / max(len(names), 1), 0.22)
    for idx, name in enumerate(names):
        offset = (idx - (len(names) - 1) / 2.0) * width
        ax.bar(x + offset, series[name], width=width, label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x, labels=labels, rotation=30, ha="right")
    ax.legend(loc="best", fontsize="small")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_signal_cost(
    *,
    plt: Any,
    path: Path,
    rows: list[dict[str, object]],
) -> bool:
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    for row in rows:
        x = _as_float(row.get("mean_size_mb"))
        y = _as_float(row.get("varying_output_count"))
        ax.scatter([x], [y], s=60)
        ax.annotate(
            str(row.get("candidate", "")),
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_title("candidate signal vs size")
    ax.set_xlabel("mean_size_mb")
    ax.set_ylabel("varying_output_count")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _variation_lookup(*row_groups: list[dict[str, object]]) -> set[tuple[str, str]]:
    lookup: set[tuple[str, str]] = set()
    for rows in row_groups:
        for row in rows:
            if str(row.get("varies", "")).lower() != "true":
                continue
            candidate = str(row.get("candidate", ""))
            feature = str(row.get("feature", ""))
            if candidate and feature:
                lookup.add((candidate, feature))
    return lookup


def _material_profile_match(profile_path: Path, label: LabelVolume) -> float | str:
    if not profile_path.exists():
        return SCORE_EMPTY
    labels = np.asarray(label.material_id)
    total = float(labels.size)
    expected: dict[int, float] = {
        int(material_id): float(np.count_nonzero(labels == int(material_id)) / total)
        for material_id in np.unique(labels)
    }
    observed: dict[int, float] = {}
    with profile_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            material_id = _optional_int(row.get("material_id"))
            if material_id is None:
                continue
            observed[material_id] = _as_float(row.get("voxel_fraction"))
    keys = set(expected) | set(observed)
    l1 = sum(abs(expected.get(key, 0.0) - observed.get(key, 0.0)) for key in keys)
    return _clamp01(1.0 - 0.5 * l1)


def _feature_alignment_scores(
    *,
    feature_name: str,
    feature_path: Path,
    label: LabelVolume,
    view_mask: np.ndarray,
    reference_label: LabelVolume | None,
) -> dict[str, float | str]:
    scores: dict[str, float | str] = {
        "occupancy_coverage": SCORE_EMPTY,
        "boundary_alignment": SCORE_EMPTY,
        "material_separation": SCORE_EMPTY,
        "profile_match": SCORE_EMPTY,
        "process_delta_alignment": SCORE_EMPTY,
    }
    if feature_name == "material_profile":
        scores["profile_match"] = _material_profile_match(feature_path, label)
        return scores
    if feature_path.suffix != ".npz" or not feature_path.exists():
        return scores
    with np.load(feature_path, allow_pickle=False) as data:
        if feature_name == "material_sdf" and "sdf_nm" in data.files:
            sdf = np.asarray(data["sdf_nm"])
            material_ids = [int(v) for v in np.asarray(data["material_ids"]).tolist()]
            per_material: list[float] = []
            for idx, material_id in enumerate(material_ids):
                per_material.append(_iou(sdf[idx] <= 0.0, label.material_id == material_id))
            scores["material_separation"] = (
                float(np.mean(per_material)) if per_material else SCORE_EMPTY
            )
            scores["occupancy_coverage"] = _iou(
                np.any(sdf <= 0.0, axis=0),
                label.material_id != label.material.void_id,
            )
            return scores
        if feature_name == "process_delta_sdf" and "changed_mask" in data.files:
            if reference_label is not None:
                changed = np.asarray(label.material_id) != np.asarray(reference_label.material_id)
                scores["process_delta_alignment"] = _iou(
                    np.asarray(data["changed_mask"], dtype=bool),
                    changed,
                )
            return scores
        if feature_name == "udf" and "udf_nm" in data.files:
            udf = np.asarray(data["udf_nm"])
            threshold = min(float(v) for v in label.grid.spacing)
            derived = udf <= threshold
            target_mask = _mask_for_array(label, view_mask, tuple(udf.shape))
            scores["boundary_alignment"] = _iou(derived, _boundary_mask(target_mask))
            return scores

        array_name = "sdf_nm" if "sdf_nm" in data.files else ""
        if not array_name and "tsdf" in data.files:
            array_name = "tsdf"
        if not array_name:
            return scores
        field = np.asarray(data[array_name])
        derived = field <= 0.0
        target = _mask_for_array(label, view_mask, tuple(field.shape))
        scores["occupancy_coverage"] = _iou(derived, target)
        scores["boundary_alignment"] = _iou(_boundary_mask(derived), _boundary_mask(target))
        return scores


def _representation_rows(
    *,
    out_dir: Path,
    view: ViewSpec,
    index_rows: list[dict[str, str]],
    case_summary_rows: list[dict[str, object]],
    variation_rows: list[dict[str, object]],
    scalar_variation_rows: list[dict[str, object]],
    profile_variation_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows_by_case = {row["case_id"]: row for row in index_rows}
    varies = _variation_lookup(variation_rows, scalar_variation_rows, profile_variation_rows)
    label_cache: dict[str, tuple[LabelVolume, LabelVolume | None, np.ndarray]] = {}
    score_rows: list[dict[str, object]] = []

    for case_row in case_summary_rows:
        candidate = str(case_row["candidate"])
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
            representation_score = _mean(float(value) for value in alignment_values)
            size_mb = _as_float(feature.get("size_mb"))
            score_rows.append(
                {
                    "candidate": candidate,
                    "case_id": case_id,
                    "feature": feature_name,
                    "path": rel_path,
                    **scores,
                    "representation_score": representation_score,
                    "variation_capture": 1.0 if (candidate, feature_name) in varies else 0.0,
                    "compactness": _clamp01(1.0 / (1.0 + size_mb)),
                }
            )
    return score_rows


def _signal_heatmap(
    *,
    candidate_eval_rows: list[dict[str, object]],
    variation_rows: list[dict[str, object]],
    scalar_variation_rows: list[dict[str, object]],
    profile_variation_rows: list[dict[str, object]],
) -> tuple[list[str], list[str], np.ndarray]:
    candidates = [str(row["candidate"]) for row in candidate_eval_rows]
    columns: list[str] = []
    values_by_key: dict[tuple[str, str], float] = {}
    for row in variation_rows:
        label = f"{row.get('feature', '')}.{row.get('array_name', '')}"
        if label not in columns:
            columns.append(label)
        values_by_key[(str(row.get("candidate", "")), label)] = (
            1.0 if str(row.get("varies", "")).lower() == "true" else 0.0
        )
    for row in scalar_variation_rows:
        label = f"{row.get('feature', '')}.{row.get('scalar', '')}"
        if label not in columns:
            columns.append(label)
        values_by_key[(str(row.get("candidate", "")), label)] = (
            1.0 if str(row.get("varies", "")).lower() == "true" else 0.0
        )
    for row in profile_variation_rows:
        label = f"{row.get('feature', '')}.{row.get('key', '')}.{row.get('scalar', '')}"
        if label not in columns:
            columns.append(label)
        values_by_key[(str(row.get("candidate", "")), label)] = (
            1.0 if str(row.get("varies", "")).lower() == "true" else 0.0
        )
    columns = columns[:40]
    matrix = np.zeros((len(candidates), len(columns)), dtype=np.float32)
    for row_idx, candidate in enumerate(candidates):
        for col_idx, column in enumerate(columns):
            matrix[row_idx, col_idx] = values_by_key.get((candidate, column), 0.0)
    return candidates, columns, matrix


def _plot_representation_score(
    *,
    plt: Any,
    path: Path,
    rows: list[dict[str, object]],
    title: str = "feature representation score",
) -> bool:
    grouped: dict[str, dict[str, list[float]]] = {}
    score_names = ["representation_score", "variation_capture", "compactness"]
    for row in rows:
        feature = str(row.get("feature", ""))
        if not feature:
            continue
        feature_scores = grouped.setdefault(feature, {name: [] for name in score_names})
        for name in score_names:
            value = row.get(name)
            if isinstance(value, int | float):
                feature_scores[name].append(float(value))
    labels = sorted(grouped)
    series = {
        name: [
            float(np.mean(grouped[label][name])) if grouped[label][name] else 0.0
            for label in labels
        ]
        for name in score_names
    }
    return _plot_grouped_bars(
        plt=plt,
        path=path,
        labels=labels,
        series=series,
        title=title,
        ylabel="normalized score",
    )


def _shape_section_xz(label: LabelVolume) -> np.ndarray:
    labels = np.asarray(label.material_id)
    y_index = labels.shape[1] // 2
    return labels[:, y_index, :]


def _shape_section_yz(label: LabelVolume) -> np.ndarray:
    labels = np.asarray(label.material_id)
    x_index = labels.shape[2] // 2
    return labels[:, :, x_index]


def _field_sections_xz_yz(array: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    arr = np.asarray(array)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    return arr[:, arr.shape[1] // 2, :], arr[:, :, arr.shape[2] // 2]


def _project_mask_view(
    mask_zyx: np.ndarray,
    *,
    axes: tuple[str, str],
    depth_axis: str,
) -> np.ndarray:
    axis_to_dim = {"z": 0, "y": 1, "x": 2}
    depth_dim = axis_to_dim[depth_axis]
    mask_any = np.any(np.asarray(mask_zyx, dtype=bool), axis=depth_dim)
    remaining_axes = [axis for axis in ("z", "y", "x") if axis != depth_axis]
    target_order = [axes[1], axes[0]]
    transpose_order = [remaining_axes.index(axis) for axis in target_order]
    return np.transpose(mask_any, transpose_order)


def _imshow_or_note(
    axis: Any,
    image: np.ndarray | None,
    title: str,
    *,
    cmap: str = "viridis",
) -> None:
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    if image is None:
        axis.text(0.5, 0.5, "not available", ha="center", va="center")
        return
    axis.imshow(image, cmap=cmap, origin="lower")


def _material_name_by_id(label: LabelVolume) -> dict[int, str]:
    return {
        int(material_id): str(name)
        for material_id, name in zip(label.material.ids, label.material.names, strict=False)
    }


def _material_axis_label(material_id: int, names: Mapping[int, str]) -> str:
    name = names.get(material_id, "")
    if not name:
        return str(material_id)
    if len(name) > 18:
        name = f"{name[:15]}..."
    return f"{material_id}:{name}"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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
    candidate_distance = np.where(inside, sdf, np.inf)
    winner = np.argmin(candidate_distance, axis=0)
    decoded = np.full(sdf.shape[1:], int(label.material.void_id), dtype=label.material_id.dtype)
    material_id_array = np.asarray(material_ids, dtype=label.material_id.dtype)
    decoded[any_inside] = material_id_array[winner[any_inside]]
    return decoded, material_ids


def _material_iou_rows(
    *,
    label: LabelVolume,
    decoded: np.ndarray,
    material_ids: Sequence[int],
    max_materials: int = 10,
) -> list[dict[str, object]]:
    labels = np.asarray(label.material_id)
    names = _material_name_by_id(label)
    rows: list[dict[str, object]] = []
    for material_id in material_ids:
        if material_id == int(label.material.void_id):
            continue
        target = labels == material_id
        predicted = decoded == material_id
        union = int(np.count_nonzero(target | predicted))
        score = 1.0 if union == 0 else float(np.count_nonzero(target & predicted) / union)
        rows.append(
            {
                "material_id": material_id,
                "label": _material_axis_label(material_id, names),
                "iou": score,
                "target_count": int(np.count_nonzero(target)),
                "predicted_count": int(np.count_nonzero(predicted)),
            }
        )
    rows.sort(
        key=lambda row: _as_float(row.get("target_count"))
        + _as_float(row.get("predicted_count")),
        reverse=True,
    )
    return rows[:max_materials]


def _plot_material_iou_bars(axis: Any, rows: list[dict[str, object]]) -> None:
    axis.set_title("material IoU (decoded SDF)")
    if not rows:
        axis.text(0.5, 0.5, "not available", ha="center", va="center")
        axis.set_xticks([])
        axis.set_yticks([])
        return
    labels = [str(row["label"]) for row in rows]
    values = [_as_float(row.get("iou")) for row in rows]
    x = np.arange(len(labels))
    axis.bar(x, values, color="#4C78A8")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("IoU")
    axis.set_xticks(x, labels=labels, rotation=30, ha="right")


def _plot_profile_fraction_bars(
    axis: Any,
    *,
    label: LabelVolume,
    profile_rows: list[dict[str, str]],
) -> None:
    labels_3d = np.asarray(label.material_id)
    total = float(labels_3d.size)
    names = _material_name_by_id(label)
    target = {
        int(material_id): float(np.count_nonzero(labels_3d == int(material_id)) / total)
        for material_id in np.unique(labels_3d)
    }
    observed: dict[int, float] = {}
    for row in profile_rows:
        material_id = _optional_int(row.get("material_id"))
        if material_id is None:
            continue
        observed[material_id] = _as_float(row.get("voxel_fraction"))
    material_ids = sorted(
        set(target) | set(observed),
        key=lambda material_id: target.get(material_id, 0.0) + observed.get(material_id, 0.0),
        reverse=True,
    )
    if not material_ids:
        axis.text(0.5, 0.5, "not available", ha="center", va="center")
        axis.set_xticks([])
        axis.set_yticks([])
        return
    x = np.arange(len(material_ids), dtype=np.float64)
    floor = 1e-8
    target_values = [max(target.get(material_id, 0.0), floor) for material_id in material_ids]
    observed_values = [max(observed.get(material_id, 0.0), floor) for material_id in material_ids]
    axis.bar(x - 0.18, target_values, width=0.36, label="input label", color="#4C78A8")
    axis.bar(x + 0.18, observed_values, width=0.36, label="profile CSV", color="#F58518")
    axis.set_title("material fraction: input vs profile")
    axis.set_ylabel("voxel_fraction (log)")
    axis.set_yscale("log")
    axis.set_xticks(
        x,
        labels=[_material_axis_label(material_id, names) for material_id in material_ids],
        rotation=30,
        ha="right",
    )
    axis.legend(loc="best", fontsize="small")


def _plot_profile_z_ranges(
    axis: Any,
    *,
    profile_rows: list[dict[str, str]],
    title: str,
) -> None:
    rows = [
        row
        for row in profile_rows
        if row.get("bbox_min_z_nm") not in ("", None)
        and row.get("bbox_max_z_nm") not in ("", None)
    ]
    rows.sort(key=lambda row: _as_float(row.get("voxel_fraction")), reverse=True)
    rows = rows[:10]
    axis.set_title(title)
    if not rows:
        axis.text(0.5, 0.5, "not available", ha="center", va="center")
        axis.set_xticks([])
        axis.set_yticks([])
        return
    labels = []
    for idx, row in enumerate(rows):
        material_id = _optional_int(row.get("material_id"))
        label_id = material_id if material_id is not None else idx
        labels.append(
            _material_axis_label(
                label_id,
                {label_id: str(row.get("material_name", row.get("transition_key", "")))},
            )
        )
    y = np.arange(len(rows))
    for idx, row in enumerate(rows):
        z_min = _as_float(row.get("bbox_min_z_nm"))
        z_max = _as_float(row.get("bbox_max_z_nm"))
        axis.hlines(idx, z_min, z_max, linewidth=5, color="#54A24B")
        axis.plot([z_min, z_max], [idx, idx], "o", color="#54A24B", markersize=4)
    axis.set_yticks(y, labels=labels)
    axis.set_xlabel("z range (nm)")
    axis.invert_yaxis()


def _plot_transition_bars(axis: Any, profile_rows: list[dict[str, str]]) -> None:
    rows = sorted(
        profile_rows,
        key=lambda row: _as_float(row.get("voxel_count")),
        reverse=True,
    )[:10]
    axis.set_title("process transitions")
    if not rows:
        axis.text(0.5, 0.5, "not available", ha="center", va="center")
        axis.set_xticks([])
        axis.set_yticks([])
        return
    labels = [str(row.get("transition_key", "")) for row in rows]
    values = [max(_as_float(row.get("voxel_count")), 1.0) for row in rows]
    x = np.arange(len(rows))
    axis.bar(x, values, color="#E45756")
    axis.set_yscale("log")
    axis.set_ylabel("changed voxels (log)")
    axis.set_xticks(x, labels=labels, rotation=30, ha="right")


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


def _representative_transform_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("candidate", "")), []).append(row)
    selected: list[dict[str, object]] = []
    for candidate_rows in groups.values():
        scored = [
            row
            for row in candidate_rows
            if isinstance(row.get("representation_score"), int | float)
        ]
        if scored:
            selected.append(
                min(scored, key=lambda row: _as_float(row.get("representation_score"), 1.0))
            )
        else:
            selected.append(
                max(candidate_rows, key=lambda row: _as_float(row.get("variation_capture")))
            )
    return selected


def _plot_material_sdf_representative(
    *,
    plt: Any,
    path: Path,
    row: dict[str, object],
    feature_path: Path,
    label: LabelVolume,
    view: ViewSpec,
    view_feature: Any,
) -> bool:
    decoded_result = _decoded_material_sdf(feature_path, label)
    if decoded_result is None:
        return False
    decoded, material_ids = decoded_result
    decoded_label = LabelVolume(
        grid=label.grid,
        material=label.material,
        material_id=decoded.astype(label.material_id.dtype, copy=False),
        meta=label.meta,
    )
    decoded_view = extract_view_feature(
        decoded_label,
        axes=view.axes,
        depth_axis=view.depth_axis,
        contour_mode="material",
    )
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.5), constrained_layout=True)
    axes[0][0].imshow(view_feature.label2d, cmap="tab20", origin="lower")
    axes[0][0].set_title("input label view")
    axes[0][1].imshow(decoded_view.label2d, cmap="tab20", origin="lower")
    axes[0][1].set_title("decoded material_sdf view")
    axes[0][2].imshow(_difference_rgb(decoded_view.label2d, view_feature.label2d), origin="lower")
    axes[0][2].set_title("decoded vs input")
    _plot_material_iou_bars(
        axes[0][3],
        _material_iou_rows(label=label, decoded=decoded, material_ids=material_ids),
    )
    _imshow_or_note(axes[1][0], _shape_section_xz(label), "input [x,z] mid-y", cmap="tab20")
    _imshow_or_note(
        axes[1][1],
        decoded[:, decoded.shape[1] // 2, :],
        "decoded [x,z] mid-y",
        cmap="tab20",
    )
    _imshow_or_note(axes[1][2], _shape_section_yz(label), "input [y,z] mid-x", cmap="tab20")
    _imshow_or_note(
        axes[1][3],
        decoded[:, :, decoded.shape[2] // 2],
        "decoded [y,z] mid-x",
        cmap="tab20",
    )
    for axis in axes.ravel()[:3].tolist() + axes.ravel()[4:].tolist():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"{row.get('candidate')} / {row.get('case_id')} / material_sdf "
        "(decoded labels from sdf<=0)"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_material_profile_representative(
    *,
    plt: Any,
    path: Path,
    row: dict[str, object],
    feature_path: Path,
    label: LabelVolume,
    view_feature: Any,
) -> bool:
    profile_rows = _read_csv_rows(feature_path)
    if not profile_rows:
        return False
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), constrained_layout=True)
    axes[0][0].imshow(view_feature.label2d, cmap="tab20", origin="lower")
    axes[0][0].set_title("input label view")
    axes[0][0].set_xticks([])
    axes[0][0].set_yticks([])
    _plot_profile_fraction_bars(axes[0][1], label=label, profile_rows=profile_rows)
    axes[0][2].axis("off")
    material_count = len(profile_rows)
    non_void = sum(str(row.get("is_void", "")).lower() != "true" for row in profile_rows)
    profile_match = _material_profile_match(feature_path, label)
    axes[0][2].text(
        0.0,
        0.5,
        "CSV summary, not a voxel field.\n\n"
        f"candidate: {row.get('candidate')}\n"
        f"case: {row.get('case_id')}\n"
        f"materials: {material_count} ({non_void} non-void)\n"
        f"profile_match: {_as_float(profile_match):.3f}\n\n"
        "Bars compare material fractions.\n"
        "Ranges show each material bbox in z.",
        va="center",
        fontsize="small",
    )
    _imshow_or_note(axes[1][0], _shape_section_xz(label), "input [x,z] mid-y", cmap="tab20")
    _imshow_or_note(axes[1][1], _shape_section_yz(label), "input [y,z] mid-x", cmap="tab20")
    _plot_profile_z_ranges(
        axes[1][2],
        profile_rows=profile_rows,
        title="profile bbox z ranges",
    )
    fig.suptitle(f"{row.get('candidate')} / {row.get('case_id')} / material_profile")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_process_delta_profile_representative(
    *,
    plt: Any,
    path: Path,
    row: dict[str, object],
    feature_path: Path,
    label: LabelVolume,
    reference_label: LabelVolume | None,
    view: ViewSpec,
    view_feature: Any,
) -> bool:
    profile_rows = _read_csv_rows(feature_path)
    if not profile_rows:
        return False
    changed: np.ndarray | None = None
    reference_view = None
    if reference_label is not None:
        changed = np.asarray(label.material_id) != np.asarray(reference_label.material_id)
        reference_view = extract_view_feature(
            reference_label,
            axes=view.axes,
            depth_axis=view.depth_axis,
            contour_mode="material",
        )
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), constrained_layout=True)
    axes[0][0].imshow(view_feature.label2d, cmap="tab20", origin="lower")
    axes[0][0].set_title("final label view")
    axes[0][0].set_xticks([])
    axes[0][0].set_yticks([])
    if reference_view is None:
        _imshow_or_note(axes[0][1], None, "reference label view")
    else:
        axes[0][1].imshow(reference_view.label2d, cmap="tab20", origin="lower")
        axes[0][1].set_title("reference label view")
        axes[0][1].set_xticks([])
        axes[0][1].set_yticks([])
    changed_view = (
        _project_mask_view(changed, axes=view.axes, depth_axis=view.depth_axis)
        if changed is not None
        else None
    )
    _imshow_or_note(axes[0][2], changed_view, "changed mask view", cmap="gray")
    _imshow_or_note(axes[1][0], _shape_section_xz(label), "final [x,z] mid-y", cmap="tab20")
    _imshow_or_note(
        axes[1][1],
        None if changed is None else changed[:, changed.shape[1] // 2, :],
        "changed [x,z] mid-y",
        cmap="gray",
    )
    _plot_transition_bars(axes[1][2], profile_rows)
    fig.suptitle(f"{row.get('candidate')} / {row.get('case_id')} / process_delta_profile")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_representative_feature(
    *,
    plt: Any,
    path: Path,
    row: dict[str, object],
    out_dir: Path,
    index_rows: dict[str, dict[str, str]],
    case_summary_rows: list[dict[str, object]],
    view: ViewSpec,
) -> bool:
    case_id = str(row["case_id"])
    index_row = index_rows.get(case_id)
    if index_row is None:
        return False
    case_summary = next(
        (
            item
            for item in case_summary_rows
            if item.get("candidate") == row.get("candidate") and item.get("case_id") == case_id
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
    view_feature = extract_view_feature(
        label,
        axes=view.axes,
        depth_axis=view.depth_axis,
        contour_mode="material",
    )
    case_out = out_dir / str(case_summary["output_dir"])
    feature_path = case_out / "features" / str(row.get("path", ""))
    if not feature_path.exists():
        return False

    feature_name = str(row.get("feature", ""))
    reference_label = None
    if index_row.get("reference_path") and index_row.get("reference_kind"):
        reference_label = load_simulation_label(
            index_row["reference_kind"],
            Path(index_row["reference_path"]),
            void_id=_optional_int(index_row.get("reference_void_id")),
        )
    if feature_name == "material_sdf":
        return _plot_material_sdf_representative(
            plt=plt,
            path=path,
            row=row,
            feature_path=feature_path,
            label=label,
            view=view,
            view_feature=view_feature,
        )
    if feature_name == "material_profile":
        return _plot_material_profile_representative(
            plt=plt,
            path=path,
            row=row,
            feature_path=feature_path,
            label=label,
            view_feature=view_feature,
        )
    if feature_name == "process_delta_profile":
        return _plot_process_delta_profile_representative(
            plt=plt,
            path=path,
            row=row,
            feature_path=feature_path,
            label=label,
            reference_label=reference_label,
            view=view,
            view_feature=view_feature,
        )

    input_image = view_feature.label2d
    feature_xz: np.ndarray | None = None
    feature_yz: np.ndarray | None = None
    if feature_path.suffix == ".npz":
        with np.load(feature_path, allow_pickle=False) as data:
            if feature_name == "material_sdf" and "sdf_nm" in data.files:
                field = np.asarray(data["sdf_nm"])[0]
            elif "sdf_nm" in data.files:
                field = np.asarray(data["sdf_nm"])
            elif "udf_nm" in data.files:
                field = np.asarray(data["udf_nm"])
            elif "changed_mask" in data.files:
                field = np.asarray(data["changed_mask"])
            else:
                field = np.asarray(data[data.files[0]])
        feature_image = _mid_slice(field)
        feature_sections = _field_sections_xz_yz(field)
        if feature_sections is not None:
            feature_xz, feature_yz = feature_sections
        if np.issubdtype(field.dtype, np.floating):
            derived = _mid_slice(field <= 0.0)
        else:
            derived = _mid_slice(field != 0)
        target = (
            view_feature.mask
            if derived.shape == view_feature.mask.shape
            else _mid_slice(label.material_id != label.material.void_id)
        )
        align_image = _difference_rgb(derived.astype(np.int32), target.astype(np.int32))
        if feature_image.shape != view_feature.label2d.shape:
            input_image = _mid_slice(label.material_id)
    else:
        profile_values: list[float] = []
        with feature_path.open("r", encoding="utf-8-sig", newline="") as f:
            for profile_row in csv.DictReader(f):
                profile_values.append(_as_float(profile_row.get("voxel_fraction")))
        feature_image = np.asarray(profile_values, dtype=np.float32)[np.newaxis, :]
        width = max(len(profile_values), 1)
        align_image = np.tile(np.asarray([[[230, 190, 40]]], dtype=np.uint8), (1, width, 1))

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.0), constrained_layout=True)
    axes[0][0].imshow(input_image, cmap="tab20", origin="lower")
    axes[0][0].set_title("input view")
    axes[0][1].imshow(feature_image, cmap="viridis", origin="lower")
    axes[0][1].set_title(feature_name)
    axes[0][2].imshow(align_image, origin="lower")
    axes[0][2].set_title("alignment")
    axes[0][3].axis("off")
    axes[0][3].text(
        0.0,
        0.5,
        f"candidate: {row.get('candidate')}\ncase: {case_id}\nfeature: {feature_name}",
        va="center",
    )
    _imshow_or_note(axes[1][0], _shape_section_xz(label), "input [x,z] mid-y", cmap="tab20")
    _imshow_or_note(axes[1][1], _shape_section_yz(label), "input [y,z] mid-x", cmap="tab20")
    _imshow_or_note(axes[1][2], feature_xz, f"{feature_name} [x,z] mid-y")
    _imshow_or_note(axes[1][3], feature_yz, f"{feature_name} [y,z] mid-x")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(f"{row.get('candidate')} / {case_id} / {feature_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def write_transform_eval_figures(
    *,
    out_dir: Path,
    view: ViewSpec,
    index_rows: list[dict[str, str]],
    candidate_summary_rows: list[dict[str, object]],
    candidate_eval_rows: list[dict[str, object]],
    variation_rows: list[dict[str, object]],
    scalar_variation_rows: list[dict[str, object]],
    profile_variation_rows: list[dict[str, object]],
    case_summary_rows: list[dict[str, object]],
) -> dict[str, object]:
    del candidate_summary_rows
    plt = _load_matplotlib_pyplot()
    figures_dir = out_dir / "figures"
    if plt is None:
        return _skip_manifest(out_dir, "matplotlib is not installed")

    figures: list[str] = []
    data: list[str] = []
    score_rows = _representation_rows(
        out_dir=out_dir,
        view=view,
        index_rows=index_rows,
        case_summary_rows=case_summary_rows,
        variation_rows=variation_rows,
        scalar_variation_rows=scalar_variation_rows,
        profile_variation_rows=profile_variation_rows,
    )
    score_csv = figures_dir / "feature_representation_scores.csv"
    score_fields = [
        "candidate",
        "case_id",
        "feature",
        "path",
        "occupancy_coverage",
        "boundary_alignment",
        "material_separation",
        "profile_match",
        "process_delta_alignment",
        "representation_score",
        "variation_capture",
        "compactness",
    ]
    _write_csv(score_csv, score_rows, score_fields)
    data.append(_rel(score_csv, out_dir))

    shape_sections = figures_dir / "input_shape_sections.png"
    if _plot_input_shape_sections(plt=plt, path=shape_sections, index_rows=index_rows):
        figures.append(_rel(shape_sections, out_dir))
    method_overview = figures_dir / "feature_method_overview.png"
    if _plot_representation_score(
        plt=plt,
        path=method_overview,
        rows=score_rows,
        title="feature method overview",
    ):
        figures.append(_rel(method_overview, out_dir))
    representation = figures_dir / "feature_representation_score.png"
    if _plot_representation_score(plt=plt, path=representation, rows=score_rows):
        figures.append(_rel(representation, out_dir))
    signal_rows, signal_cols, signal_values = _signal_heatmap(
        candidate_eval_rows=candidate_eval_rows,
        variation_rows=variation_rows,
        scalar_variation_rows=scalar_variation_rows,
        profile_variation_rows=profile_variation_rows,
    )
    signal_heatmap = figures_dir / "candidate_signal_heatmap.png"
    if _plot_heatmap(
        plt=plt,
        path=signal_heatmap,
        row_labels=signal_rows,
        col_labels=signal_cols,
        values=signal_values,
        title="candidate signal heatmap",
    ):
        figures.append(_rel(signal_heatmap, out_dir))
    signal_cost = figures_dir / "candidate_signal_cost.png"
    if _plot_signal_cost(plt=plt, path=signal_cost, rows=candidate_eval_rows):
        figures.append(_rel(signal_cost, out_dir))

    index_by_case = {row["case_id"]: row for row in index_rows}
    for row in _representative_transform_rows(score_rows):
        stem = _safe_file_stem(f"{row.get('candidate')}_{row.get('case_id')}_{row.get('feature')}")
        path = figures_dir / "representative_feature_slices" / f"{stem}.png"
        if _plot_representative_feature(
            plt=plt,
            path=path,
            row=row,
            out_dir=out_dir,
            index_rows=index_by_case,
            case_summary_rows=case_summary_rows,
            view=view,
        ):
            figures.append(_rel(path, out_dir))

    return _ok_manifest(out_dir, figures, data, TRANSFORM_FIGURE_NOTES)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _matrix_from_rows(
    rows: list[dict[str, object]],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
) -> tuple[list[str], list[str], np.ndarray]:
    row_labels = _ordered_unique(str(row.get(row_key, "")) for row in rows if row.get(row_key))
    col_labels = _ordered_unique(str(row.get(col_key, "")) for row in rows if row.get(col_key))
    matrix = np.zeros((len(row_labels), len(col_labels)), dtype=np.float32)
    row_idx = {label: idx for idx, label in enumerate(row_labels)}
    col_idx = {label: idx for idx, label in enumerate(col_labels)}
    for row in rows:
        r = str(row.get(row_key, ""))
        c = str(row.get(col_key, ""))
        if r in row_idx and c in col_idx:
            matrix[row_idx[r], col_idx[c]] = _as_float(row.get(value_key))
    return row_labels, col_labels, matrix


def _metric_evaluation_rows(candidate_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    max_std = max(
        (_as_float(row.get("std_normalized_total_score")) for row in candidate_rows),
        default=0.0,
    )
    rows: list[dict[str, object]] = []
    for row in candidate_rows:
        metrics = [value for value in str(row.get("metrics", "")).split("|") if value]
        case_count = max(_as_float(row.get("case_count")), 1.0)
        metric_count = max(float(len(metrics)), 1.0)
        case_coverage = _clamp01(_as_float(row.get("ok_case_count")) / case_count)
        skipped = _as_float(row.get("skipped_metric_count"))
        metric_coverage = _clamp01(1.0 - skipped / (case_count * metric_count))
        ranking_stability = _clamp01(1.0 / (1.0 + _as_float(row.get("mean_abs_rank_delta"))))
        objective_spread = (
            _clamp01(_as_float(row.get("std_normalized_total_score")) / max_std)
            if max_std > 0.0
            else 0.0
        )
        runtime_efficiency = _clamp01(1.0 / (1.0 + _as_float(row.get("mean_runtime_sec"))))
        metric_evaluation_score = float(
            np.mean(
                np.asarray(
                    [case_coverage, metric_coverage, ranking_stability, objective_spread],
                    dtype=np.float64,
                )
            )
        )
        rows.append(
            {
                "candidate": row.get("candidate", ""),
                "case_coverage": case_coverage,
                "metric_coverage": metric_coverage,
                "ranking_stability": ranking_stability,
                "objective_spread": objective_spread,
                "runtime_efficiency": runtime_efficiency,
                "metric_evaluation_score": metric_evaluation_score,
            }
        )
    return rows


def _plot_metric_evaluation(
    *,
    plt: Any,
    path: Path,
    rows: list[dict[str, object]],
) -> bool:
    labels = [str(row.get("candidate", "")) for row in rows]
    series = {
        "case_coverage": [_as_float(row.get("case_coverage")) for row in rows],
        "metric_coverage": [_as_float(row.get("metric_coverage")) for row in rows],
        "ranking_stability": [_as_float(row.get("ranking_stability")) for row in rows],
        "objective_spread": [_as_float(row.get("objective_spread")) for row in rows],
        "metric_evaluation_score": [_as_float(row.get("metric_evaluation_score")) for row in rows],
    }
    return _plot_grouped_bars(
        plt=plt,
        path=path,
        labels=labels,
        series=series,
        title="metric evaluation score",
        ylabel="normalized score",
    )


def _representative_compare_keys(
    *,
    spec: CompareEvalSpec,
    case_rows: list[dict[str, object]],
    ranking_rows: list[dict[str, object]],
) -> list[tuple[str, str]]:
    baseline = next(iter(spec.candidates), "")
    selected: list[tuple[str, str]] = []
    baseline_rows = [row for row in case_rows if row.get("candidate") == baseline]
    if baseline_rows:
        best = min(baseline_rows, key=lambda row: _as_float(row.get("normalized_total_score")))
        worst = max(baseline_rows, key=lambda row: _as_float(row.get("normalized_total_score")))
        selected.extend([(baseline, str(best["case_id"])), (baseline, str(worst["case_id"]))])
    if ranking_rows:
        rank_row = max(ranking_rows, key=lambda row: abs(int(_as_float(row.get("rank_delta")))))
        selected.append((str(rank_row.get("candidate", "")), str(rank_row.get("case_id", ""))))

    unique: list[tuple[str, str]] = []
    for item in selected:
        if item[0] and item[1] and item not in unique:
            unique.append(item)
    return unique[:3]


def _compare_view_features(
    *,
    spec: CompareEvalSpec,
    index_row: dict[str, str],
    index_dir: Path,
) -> tuple[np.ndarray, np.ndarray, LabelVolume, LabelVolume | None]:
    sim_path = resolve_path(index_row["simulation_path"], base_dir=index_dir)
    target_path = resolve_path(index_row["target_path"], base_dir=index_dir)
    target_kind = index_row.get("target_kind") or "contour_json"
    sim_label = load_simulation_label(
        index_row["simulation_kind"],
        sim_path,
        void_id=_optional_int(index_row.get("simulation_void_id")),
    )
    sim_feature = extract_view_feature(
        sim_label,
        axes=spec.view.axes,
        depth_axis=spec.view.depth_axis,
        contour_mode="outer" if target_kind == "contour_json" else "material",
    )
    if target_kind in CONTOUR_LOADERS:
        target_data = CONTOUR_LOADERS[target_kind](
            target_path,
            units_override=index_row.get("target_units") or "nm",
            view_axes=spec.view.axes,
        )
        target_feature = contour_feature_on_grid(
            target_data,
            sim_feature.grid2d,
            sim_feature.mask.shape,
            axes=spec.view.axes,
        )
        target_label = None
    else:
        target_label = load_simulation_label(
            target_kind,
            target_path,
            void_id=_optional_int(index_row.get("target_void_id")),
        )
        target_feature = extract_view_feature(
            target_label,
            axes=spec.view.axes,
            depth_axis=spec.view.depth_axis,
            contour_mode="material",
        )
    return sim_feature.label2d, target_feature.label2d, sim_label, target_label


def _plot_representative_difference(
    *,
    plt: Any,
    path: Path,
    spec: CompareEvalSpec,
    index_row: dict[str, str],
    index_dir: Path,
    score_row: dict[str, object],
) -> bool:
    sim_label, target_label, sim_volume, target_volume = _compare_view_features(
        spec=spec,
        index_row=index_row,
        index_dir=index_dir,
    )
    loss_items = [
        f"{key[:-5]}={_as_float(value):.3g}"
        for key, value in score_row.items()
        if key.endswith("_loss") and value != ""
    ][:3]
    title = (
        f"{score_row.get('candidate')} / {score_row.get('case_id')} / "
        f"score={_as_float(score_row.get('normalized_total_score')):.3g}"
    )
    if loss_items:
        title = f"{title} / {', '.join(loss_items)}"

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.0), constrained_layout=True)
    axes[0][0].imshow(target_label, cmap="tab20", origin="lower")
    axes[0][0].set_title("target view")
    axes[0][1].imshow(sim_label, cmap="tab20", origin="lower")
    axes[0][1].set_title("simulation view")
    axes[0][2].imshow(_difference_rgb(sim_label, target_label), origin="lower")
    axes[0][2].set_title("overlay + diff")
    axes[0][3].axis("off")
    axes[0][3].text(0.0, 0.5, title, va="center")
    _imshow_or_note(axes[1][0], _shape_section_xz(sim_volume), "simulation [x,z]", cmap="tab20")
    _imshow_or_note(axes[1][1], _shape_section_yz(sim_volume), "simulation [y,z]", cmap="tab20")
    _imshow_or_note(
        axes[1][2],
        _shape_section_xz(target_volume) if target_volume is not None else None,
        "target [x,z]",
        cmap="tab20",
    )
    _imshow_or_note(
        axes[1][3],
        _shape_section_yz(target_volume) if target_volume is not None else None,
        "target [y,z]",
        cmap="tab20",
    )
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def write_compare_eval_figures(
    *,
    out_dir: Path,
    spec: CompareEvalSpec,
    index_rows: list[dict[str, str]],
    index_dir: Path,
    case_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    metric_summary_rows: list[dict[str, object]],
    ranking_rows: list[dict[str, object]],
) -> dict[str, object]:
    plt = _load_matplotlib_pyplot()
    figures_dir = out_dir / "figures"
    if plt is None:
        return _skip_manifest(out_dir, "matplotlib is not installed")

    figures: list[str] = []
    data: list[str] = []
    eval_rows = _metric_evaluation_rows(candidate_rows)
    eval_csv = figures_dir / "metric_evaluation_scores.csv"
    _write_csv(
        eval_csv,
        eval_rows,
        [
            "candidate",
            "case_coverage",
            "metric_coverage",
            "ranking_stability",
            "objective_spread",
            "runtime_efficiency",
            "metric_evaluation_score",
        ],
    )
    data.append(_rel(eval_csv, out_dir))

    row_labels, col_labels, matrix = _matrix_from_rows(
        case_rows,
        row_key="candidate",
        col_key="case_id",
        value_key="normalized_total_score",
    )
    objective_path = figures_dir / "objective_heatmap.png"
    if _plot_heatmap(
        plt=plt,
        path=objective_path,
        row_labels=row_labels,
        col_labels=col_labels,
        values=matrix,
        title="objective heatmap (lower is better)",
        vmax=None,
    ):
        figures.append(_rel(objective_path, out_dir))

    rank_rows, rank_cols, rank_matrix = _matrix_from_rows(
        ranking_rows,
        row_key="candidate",
        col_key="case_id",
        value_key="rank_delta",
    )
    max_delta = float(np.max(np.abs(rank_matrix))) if rank_matrix.size else 0.0
    rank_path = figures_dir / "rank_delta_heatmap.png"
    if _plot_heatmap(
        plt=plt,
        path=rank_path,
        row_labels=rank_rows,
        col_labels=rank_cols,
        values=rank_matrix,
        title="rank delta heatmap",
        cmap="coolwarm",
        vmin=-max(max_delta, 1.0),
        vmax=max(max_delta, 1.0),
    ):
        figures.append(_rel(rank_path, out_dir))

    contrib_rows, contrib_cols, contrib_matrix = _matrix_from_rows(
        metric_summary_rows,
        row_key="candidate",
        col_key="metric",
        value_key="mean_normalized_loss",
    )
    contrib_path = figures_dir / "metric_contribution_heatmap.png"
    if _plot_heatmap(
        plt=plt,
        path=contrib_path,
        row_labels=contrib_rows,
        col_labels=contrib_cols,
        values=contrib_matrix,
        title="metric contribution heatmap",
        vmax=None,
    ):
        figures.append(_rel(contrib_path, out_dir))

    eval_path = figures_dir / "metric_evaluation_score.png"
    if _plot_metric_evaluation(plt=plt, path=eval_path, rows=eval_rows):
        figures.append(_rel(eval_path, out_dir))

    case_by_key = {(str(row["candidate"]), str(row["case_id"])): row for row in case_rows}
    index_by_case = {row["case_id"]: row for row in index_rows}
    for candidate, case_id in _representative_compare_keys(
        spec=spec,
        case_rows=case_rows,
        ranking_rows=ranking_rows,
    ):
        index_row = index_by_case.get(case_id)
        score_row = case_by_key.get((candidate, case_id))
        if index_row is None or score_row is None:
            continue
        path = (
            figures_dir
            / "representative_differences"
            / f"{_safe_file_stem(candidate)}_{_safe_file_stem(case_id)}.png"
        )
        if _plot_representative_difference(
            plt=plt,
            path=path,
            spec=spec,
            index_row=index_row,
            index_dir=index_dir,
            score_row=score_row,
        ):
            figures.append(_rel(path, out_dir))

    return _ok_manifest(out_dir, figures, data, COMPARE_FIGURE_NOTES)
