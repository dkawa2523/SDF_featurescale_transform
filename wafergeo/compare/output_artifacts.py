from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np

from wafergeo.compare.features import ViewFeature
from wafergeo.compare.render import write_rgb_png


def _as_float(value: object, default: float = 0.0) -> float:
    if not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iter_metric_detail_rows(
    metric_details: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    rows = list(metric_details)
    if len(rows) == 1 and isinstance(rows[0].get("details"), list):
        nested = cast(list[object], rows[0]["details"])
        return [row for row in nested if isinstance(row, dict)]
    return rows


def write_per_material_sdf_csv(
    path: str | Path,
    metric_details: Iterable[dict[str, object]],
    *,
    case_id: str | None = None,
) -> bool:
    rows: list[dict[str, object]] = []
    for detail in _iter_metric_detail_rows(metric_details):
        if detail.get("metric") != "sdf_material":
            continue
        per_material = detail.get("per_material", [])
        if not isinstance(per_material, list):
            continue
        for row in per_material:
            if not isinstance(row, dict):
                continue
            output_row: dict[str, object] = {
                "material_id": row.get("material_id", ""),
                "sdf_loss_nm": row.get("sdf_loss_nm", ""),
                "sim_pixels": row.get("sim_pixels", ""),
                "target_pixels": row.get("target_pixels", ""),
                "union_pixels": row.get("union_pixels", ""),
            }
            if case_id is not None:
                output_row = {"case_id": case_id, **output_row}
            elif "case_id" in row:
                output_row = {"case_id": row["case_id"], **output_row}
            rows.append(output_row)
    if not rows:
        return False

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def write_metric_summary_csv(path: str | Path, metric_rows: Iterable[dict[str, object]]) -> bool:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        name = str(row.get("name", ""))
        if name:
            grouped[name].append(row)
    if not grouped:
        return False

    output_rows: list[dict[str, object]] = []
    for name in sorted(grouped):
        rows = grouped[name]
        losses = np.asarray([_as_float(row.get("loss")) for row in rows], dtype=np.float64)
        normalized = np.asarray(
            [_as_float(row.get("normalized_loss")) for row in rows],
            dtype=np.float64,
        )
        output_rows.append(
            {
                "name": name,
                "case_count": len(rows),
                "ok_count": sum(1 for row in rows if row.get("status") == "OK"),
                "skipped_count": sum(1 for row in rows if row.get("status") == "SKIPPED"),
                "loss_min": float(np.min(losses)),
                "loss_max": float(np.max(losses)),
                "loss_mean": float(np.mean(losses)),
                "loss_std": float(np.std(losses)),
                "normalized_loss_mean": float(np.mean(normalized)),
            }
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    return True


def _material_confusion_rows(
    sim_feature: ViewFeature,
    target_feature: ViewFeature,
) -> list[dict[str, object]]:
    sim_labels = np.asarray(sim_feature.label2d)
    target_labels = np.asarray(target_feature.label2d)
    total_pixels = int(sim_labels.size)
    sim_totals = {
        int(label): int((sim_labels == label).sum())
        for label in np.unique(sim_labels)
    }
    target_totals = {
        int(label): int((target_labels == label).sum())
        for label in np.unique(target_labels)
    }
    rows: list[dict[str, object]] = []
    for sim_id in sorted(sim_totals):
        sim_mask = sim_labels == sim_id
        for target_id in sorted(target_totals):
            pixels = int(np.logical_and(sim_mask, target_labels == target_id).sum())
            if pixels == 0:
                continue
            rows.append(
                {
                    "simulation_material_id": sim_id,
                    "target_material_id": target_id,
                    "pixels": pixels,
                    "fraction_of_total": float(pixels / total_pixels) if total_pixels else 0.0,
                    "fraction_of_simulation_material": float(pixels / sim_totals[sim_id])
                    if sim_totals[sim_id]
                    else 0.0,
                    "fraction_of_target_material": float(pixels / target_totals[target_id])
                    if target_totals[target_id]
                    else 0.0,
                }
            )
    return rows


def write_material_confusion_outputs(
    output_dir: str | Path,
    *,
    sim_feature: ViewFeature,
    target_feature: ViewFeature,
) -> bool:
    if sim_feature.source != "label_volume" or target_feature.source != "label_volume":
        return False
    if sim_feature.label2d.shape != target_feature.label2d.shape:
        return False

    output_path = Path(output_dir)
    rows = _material_confusion_rows(sim_feature, target_feature)
    if not rows:
        return False

    csv_path = output_path / "material_confusion.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total_pixels = int(sim_feature.label2d.size)
    matching_pixels = int(
        np.sum(np.asarray(sim_feature.label2d) == np.asarray(target_feature.label2d))
    )
    mismatching_pixels = total_pixels - matching_pixels
    off_diagonal = [
        row
        for row in rows
        if row["simulation_material_id"] != row["target_material_id"]
    ]
    major_pair = (
        max(off_diagonal, key=lambda row: _as_float(row["pixels"]))
        if off_diagonal
        else None
    )
    summary: dict[str, object] = {
        "total_pixels": total_pixels,
        "matching_pixels": matching_pixels,
        "mismatching_pixels": mismatching_pixels,
        "accuracy": float(matching_pixels / total_pixels) if total_pixels else 1.0,
        "simulation_material_ids": sorted(int(v) for v in np.unique(sim_feature.label2d)),
        "target_material_ids": sorted(int(v) for v in np.unique(target_feature.label2d)),
        "major_confusion_pair": major_pair,
    }
    (output_path / "material_confusion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return True


def _draw_line(
    rgb: np.ndarray,
    start: tuple[int, int],
    stop: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    y0, x0 = start
    y1, x1 = stop
    steps = max(abs(y1 - y0), abs(x1 - x0), 1) + 1
    yy = np.rint(np.linspace(y0, y1, steps)).astype(int)
    xx = np.rint(np.linspace(x0, x1, steps)).astype(int)
    valid = (yy >= 0) & (yy < rgb.shape[0]) & (xx >= 0) & (xx < rgb.shape[1])
    rgb[yy[valid], xx[valid]] = np.asarray(color, dtype=np.uint8)


def _scale_values(values: np.ndarray, *, low: float, high: float) -> np.ndarray:
    if not np.isfinite(high - low) or high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def write_cd_profile_png(path: str | Path, profile_rows: Iterable[dict[str, float]]) -> bool:
    rows = list(profile_rows)
    if not rows:
        return False
    z_values = np.asarray([_as_float(row.get("z_nm")) for row in rows], dtype=np.float32)
    losses = np.asarray([_as_float(row.get("edge_loss_nm")) for row in rows], dtype=np.float32)
    if z_values.size == 0:
        return False

    height, width, margin = 260, 520, 28
    rgb = np.full((height, width, 3), 250, dtype=np.uint8)
    rgb[height - margin, margin : width - margin] = np.array([80, 80, 80], dtype=np.uint8)
    rgb[margin : height - margin, margin] = np.array([80, 80, 80], dtype=np.uint8)

    x_norm = _scale_values(z_values, low=float(np.min(z_values)), high=float(np.max(z_values)))
    y_high = max(float(np.max(losses)), 1.0)
    y_norm = _scale_values(losses, low=0.0, high=y_high)
    x_pix = margin + np.rint(x_norm * (width - 2 * margin - 1)).astype(int)
    y_pix = height - margin - np.rint(y_norm * (height - 2 * margin - 1)).astype(int)

    for idx in range(1, len(x_pix)):
        _draw_line(
            rgb,
            (int(y_pix[idx - 1]), int(x_pix[idx - 1])),
            (int(y_pix[idx]), int(x_pix[idx])),
            (220, 70, 70),
        )
    for y, x in zip(y_pix, x_pix, strict=False):
        rgb[max(int(y) - 1, 0) : int(y) + 2, max(int(x) - 1, 0) : int(x) + 2] = np.array(
            [70, 110, 220],
            dtype=np.uint8,
        )
    write_rgb_png(path, rgb)
    return True


def write_ranking_top_png(
    path: str | Path,
    ranking_rows: Iterable[dict[str, str | float]],
    *,
    top_n: int = 10,
) -> bool:
    rows = list(ranking_rows)[:top_n]
    if not rows:
        return False
    values = np.asarray(
        [_as_float(row.get("normalized_total_score")) for row in rows],
        dtype=np.float32,
    )
    max_value = max(float(np.max(values)), 1.0)
    bar_h = 18
    gap = 8
    margin = 24
    width = 520
    height = max(120, margin * 2 + len(rows) * (bar_h + gap))
    rgb = np.full((height, width, 3), 250, dtype=np.uint8)
    axis_x = margin + 100
    rgb[margin : height - margin, axis_x] = np.array([80, 80, 80], dtype=np.uint8)
    usable_width = width - axis_x - margin

    for idx, value in enumerate(values):
        top = margin + idx * (bar_h + gap)
        length = int(math.ceil((float(value) / max_value) * usable_width))
        color = np.array([70, 160, 80], dtype=np.uint8) if idx == 0 else np.array(
            [70, 110, 220],
            dtype=np.uint8,
        )
        rgb[top : top + bar_h, axis_x : axis_x + max(length, 1)] = color
    write_rgb_png(path, rgb)
    return True


def write_label_image_png(path: str | Path, labels: np.ndarray, *, void_id: int = 0) -> None:
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError("label image expects a 2D label array")
    rgb = np.full(arr.shape + (3,), 245, dtype=np.uint8)
    palette = np.asarray(
        [
            [70, 110, 220],
            [70, 160, 80],
            [220, 70, 70],
            [230, 190, 40],
            [150, 90, 180],
            [70, 170, 180],
            [210, 120, 50],
        ],
        dtype=np.uint8,
    )
    for idx, label_id in enumerate(sorted(int(v) for v in np.unique(arr))):
        if label_id == int(void_id):
            continue
        rgb[arr == label_id] = palette[idx % len(palette)]
    smallest = max(min(arr.shape), 1)
    scale = max(1, min(8, int(math.ceil(384 / smallest)))) if smallest < 384 else 1
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    write_rgb_png(path, rgb)
