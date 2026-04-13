from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from wafergeo.core.types import LabelVolume

FIGURE_INDEX = "figures/index.json"
SCORE_EMPTY = ""
TRANSFORM_FIGURE_NOTES = {
    "input_shape_sections.png": (
        "Rows are cases. The left panel is the [x,z] mid-y section and the right panel "
        "is the [y,z] mid-x section of the original label volume."
    ),
    "by_target_shape": (
        "SDF-method figures are grouped as by_target_shape/<target_shape>/<method>/. "
        "SDF-derived relation figures are grouped as "
        "by_target_shape/<target_shape>/relations/<relation>/."
    ),
}
COMPARE_FIGURE_NOTES = {
    "comparison_loss_heatmap.png": (
        "Rows are evaluation axes and columns are cases. Lower comparison_loss is better."
    ),
    "ranking_shift_heatmap.png": (
        "Color shows how each evaluation axis changes case ranking relative to the first axis."
    ),
    "metric_loss_breakdown.png": (
        "Bright cells show which metrics contribute larger normalized loss."
    ),
    "cd_vs_sdf_scatter.png": (
        "Compares height-CD loss and SDF shape loss per case. Points far from the "
        "diagonal are cases where CD and SDF judge the geometry differently."
    ),
    "axis_agreement.csv": (
        "Pairwise agreement between evaluation axes, including loss correlation and rank agreement."
    ),
    "evaluation_axis_summary.png": (
        "Shows coverage, case separation, and mean ranking shift for each evaluation axis. "
        "It is a diagnostic view, not an optimization objective."
    ),
    "evaluation_axis_summary.csv": "Machine-readable data for evaluation_axis_summary.png.",
    "representative_differences": (
        "Representative cases show target, simulation, and diff for the view, [x,z], "
        "and [y,z] sections. Yellow is overlap, red is simulation-only, green is "
        "target-only, blue is label mismatch."
    ),
}


def _load_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "eval figure generation requires matplotlib; install the viz extra "
            "or run in an environment that provides matplotlib."
        ) from exc
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


def _ok_figure_index(
    out_dir: Path,
    figures: list[str],
    data: list[str],
    notes: Mapping[str, object],
) -> dict[str, object]:
    guide = _write_figure_readme(out_dir, notes)
    index: dict[str, object] = {
        "status": "OK",
        "figures": figures,
        "data": data,
        "guide": guide,
        "how_to_read": dict(notes),
    }
    _write_json(out_dir / FIGURE_INDEX, index)
    return {
        "status": "OK",
        "index": FIGURE_INDEX,
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


def _shape_section_xz(label: LabelVolume) -> np.ndarray:
    array = np.asarray(label.material_id)
    y_index = array.shape[1] // 2
    return array[:, y_index, :]


def _shape_section_yz(label: LabelVolume) -> np.ndarray:
    array = np.asarray(label.material_id)
    x_index = array.shape[2] // 2
    return array[:, :, x_index]


def _imshow_or_note(
    axis: Any,
    image: np.ndarray | None,
    title: str,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    axis.set_title(title)
    if image is None:
        raise ValueError(f"figure image is missing: {title}")
    axis.imshow(image, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    axis.set_xticks([])
    axis.set_yticks([])


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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


