from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wafergeo.compare.features import ViewFeature
from wafergeo.core.types import LabelVolume


def _bbox_1d(indices: np.ndarray, *, origin: float, spacing: float) -> dict[str, float]:
    start = float(origin + float(indices.min()) * spacing)
    stop = float(origin + float(indices.max()) * spacing)
    return {"min_nm": start, "max_nm": stop}


def summarize_label_volume(label: LabelVolume) -> dict[str, object]:
    labels = np.asarray(label.material_id)
    shape_zyx = tuple(int(v) for v in labels.shape)
    spacing_zyx = tuple(float(v) for v in label.grid.spacing)
    origin_zyx = tuple(float(v) for v in label.grid.origin)
    materials: list[dict[str, object]] = []
    name_by_id = {
        int(material_id): str(name)
        for material_id, name in zip(label.material.ids, label.material.names, strict=True)
    }

    for material_id in sorted(int(v) for v in np.unique(labels)):
        mask = labels == material_id
        zz, yy, xx = np.nonzero(mask)
        material_row: dict[str, object] = {
            "material_id": material_id,
            "name": name_by_id.get(material_id),
            "voxel_count": int(mask.sum()),
            "is_void": material_id == int(label.material.void_id),
        }
        if zz.size:
            material_row["bbox_nm"] = {
                "x": _bbox_1d(xx, origin=origin_zyx[2], spacing=spacing_zyx[2]),
                "y": _bbox_1d(yy, origin=origin_zyx[1], spacing=spacing_zyx[1]),
                "z": _bbox_1d(zz, origin=origin_zyx[0], spacing=spacing_zyx[0]),
            }
        materials.append(material_row)

    non_void = labels != int(label.material.void_id)
    return {
        "source": "label_volume",
        "shape_zyx": list(shape_zyx),
        "spacing_zyx_nm": list(spacing_zyx),
        "origin_zyx_nm": list(origin_zyx),
        "axis_order": label.grid.axis_order,
        "units": label.grid.units,
        "void_id": int(label.material.void_id),
        "material_ids": [int(v) for v in label.material.ids],
        "non_void_voxels": int(non_void.sum()),
        "materials": materials,
    }


def summarize_view_feature(feature: ViewFeature) -> dict[str, object]:
    labels = np.asarray(feature.label2d)
    axes = tuple(str(v) for v in feature.axes)
    axis_y, axis_x = axes[1], axes[0]
    spacing_y, spacing_x = (float(v) for v in feature.grid2d.spacing)
    origin_y, origin_x = (float(v) for v in feature.grid2d.origin)
    label_rows: list[dict[str, object]] = []

    for label_id in sorted(int(v) for v in np.unique(labels)):
        mask = labels == label_id
        rows, cols = np.nonzero(mask)
        label_row: dict[str, object] = {
            "label_id": label_id,
            "pixel_count": int(mask.sum()),
        }
        if rows.size:
            label_row["bbox_nm"] = {
                axis_x: _bbox_1d(cols, origin=origin_x, spacing=spacing_x),
                axis_y: _bbox_1d(rows, origin=origin_y, spacing=spacing_y),
            }
        label_rows.append(label_row)

    boundary_pixels = (
        int(np.asarray(feature.boundary_mask, dtype=bool).sum())
        if feature.boundary_mask is not None
        else None
    )
    material_mask_rows = []
    for material_id, mask in sorted((feature.material_masks or {}).items()):
        rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
        row: dict[str, object] = {
            "material_id": int(material_id),
            "pixel_count": int(len(rows)),
        }
        if rows.size:
            row["bbox_nm"] = {
                axis_x: _bbox_1d(cols, origin=origin_x, spacing=spacing_x),
                axis_y: _bbox_1d(rows, origin=origin_y, spacing=spacing_y),
            }
        material_mask_rows.append(row)
    return {
        "source": feature.source,
        "axes": list(axes),
        "shape_yx": [int(v) for v in labels.shape],
        "spacing_yx_nm": [spacing_y, spacing_x],
        "origin_yx_nm": [origin_y, origin_x],
        "non_void_pixels": int(np.asarray(feature.mask, dtype=bool).sum()),
        "boundary_pixels": boundary_pixels,
        "material_masks": material_mask_rows,
        "labels": label_rows,
    }


def write_json_summary(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
