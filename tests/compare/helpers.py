from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_npz(
    path: Path,
    *,
    shift_x: int = 0,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Path:
    labels = np.zeros((8, 8, 2), dtype=np.uint8)
    labels[2 + shift_x : 6 + shift_x, 2:6, :] = 1
    np.savez(
        path,
        labels=labels,
        spacing=np.array(spacing, dtype=np.float32),
        origin=np.array(origin, dtype=np.float32),
        material_ids=np.array([0, 1], dtype=np.int32),
    )
    return path


def write_internal_boundary_npz(path: Path, *, split_x: int = 4) -> Path:
    labels = np.zeros((8, 8, 2), dtype=np.uint8)
    labels[1:7, 1:7, :] = 1
    labels[split_x:7, 1:7, :] = 2
    np.savez(
        path,
        labels=labels,
        spacing=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        material_ids=np.array([0, 1, 2], dtype=np.int32),
    )
    return path


def write_swapped_material_npz(path: Path, *, swap: bool = False) -> Path:
    labels = np.zeros((8, 8, 2), dtype=np.uint8)
    labels[1:4, 1:7, :] = 1 if not swap else 2
    labels[4:7, 1:7, :] = 2 if not swap else 1
    np.savez(
        path,
        labels=labels,
        spacing=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        material_ids=np.array([0, 1, 2], dtype=np.int32),
    )
    return path


def write_hidden_material_npz(path: Path, *, shift_x: int = 0) -> Path:
    labels = np.ones((8, 5, 4), dtype=np.uint8)
    labels[2 + shift_x : 5 + shift_x, 1:2, :] = 2
    np.savez(
        path,
        labels=labels,
        spacing=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        material_ids=np.array([0, 1, 2], dtype=np.int32),
    )
    return path


def write_cd_opening_npz(path: Path, *, half_width: int) -> Path:
    labels = np.ones((9, 5, 6), dtype=np.uint8)
    center = labels.shape[0] // 2
    labels[center - half_width : center + half_width + 1, :, :] = 0
    np.savez(
        path,
        labels=labels,
        spacing=np.array([1.0, 1.0, 2.0], dtype=np.float32),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        material_ids=np.array([0, 1], dtype=np.int32),
    )
    return path


def write_cd_material_feature_npz(
    path: Path,
    *,
    half_width: int,
    center_offset: int = 0,
) -> Path:
    labels = np.ones((9, 5, 6), dtype=np.uint8)
    center = labels.shape[0] // 2 + center_offset
    labels[center - half_width : center + half_width + 1, :, :] = 2
    np.savez(
        path,
        labels=labels,
        spacing=np.array([1.0, 1.0, 2.0], dtype=np.float32),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        material_ids=np.array([0, 1, 2], dtype=np.int32),
    )
    return path


def write_contour(path: Path, *, shift_x: float = 0.0, units: str = "nm") -> Path:
    payload = {
        "schema_version": "contour/v1",
        "units": units,
        "coordinate_axes": ["x", "y", "z"],
        "contours": [
            {
                "id": "outer",
                "label": "global",
                "material_id": None,
                "closed": True,
                "points": [
                    [1.5 + shift_x, 1.5, 0.0],
                    [5.5 + shift_x, 1.5, 0.0],
                    [5.5 + shift_x, 5.5, 0.0],
                    [1.5 + shift_x, 5.5, 0.0],
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_compare_config(
    path: Path,
    *,
    sim_path: Path,
    target_path: Path,
    out_dir: Path,
) -> Path:
    path.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {sim_path}
  target:
    kind: contour_json
    path: {target_path}
    units: nm
view:
  kind: topview
  axes: [x, y]
  depth_axis: z
features:
  use: [sdf, contour]
metrics:
  use: [cd, chamfer, sdf, iou]
output:
  dir: {out_dir}
  difference_image: true
""",
        encoding="utf-8",
    )
    return path


def write_label_target_compare_config(
    path: Path,
    *,
    sim_path: Path,
    target_path: Path,
    out_dir: Path,
    axes: str = "[x, y]",
    depth_axis: str = "z",
) -> Path:
    path.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {sim_path}
  target:
    kind: npz_label
    path: {target_path}
view:
  kind: topview
  axes: {axes}
  depth_axis: {depth_axis}
features:
  use: [sdf, contour]
metrics:
  use: [cd, chamfer, sdf, iou]
output:
  dir: {out_dir}
  difference_image: true
""",
        encoding="utf-8",
    )
    return path
