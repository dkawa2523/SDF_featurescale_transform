from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np

from wafergeo.compare.loader_types import AxisName, ContourData, ContourItem


def _unit_to_nm_factor(units: str) -> float:
    key = units.strip().lower()
    if key == "nm":
        return 1.0
    if key in {"um", "micron", "micrometer", "micrometre"}:
        return 1000.0
    if key in {"mm"}:
        return 1_000_000.0
    raise ValueError(f"unsupported contour units: {units}")


def _project_points(
    points_raw: object,
    *,
    coordinate_axes: list[str],
    view_axes: tuple[AxisName, AxisName],
) -> np.ndarray:
    points = np.asarray(points_raw, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError("contour points must be shape (N,2) or (N,3)")
    if points.shape[1] == 2:
        return points.astype(np.float32, copy=False)
    if points.shape[1] != len(coordinate_axes):
        raise ValueError("3D contour points must match coordinate_axes length")
    axis_index = {name: idx for idx, name in enumerate(coordinate_axes)}
    try:
        first = axis_index[view_axes[0]]
        second = axis_index[view_axes[1]]
    except KeyError as exc:
        raise ValueError(f"view axis is not present in contour coordinate_axes: {exc}") from exc
    return points[:, [first, second]].astype(np.float32, copy=False)


def load_contour_json(
    path: str | Path,
    *,
    units_override: str | None = None,
    view_axes: tuple[AxisName, AxisName] = ("x", "y"),
) -> ContourData:
    input_path = Path(path)
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("contour_json root must be an object")
    schema_version = str(raw.get("schema_version", "contour/v1"))
    if schema_version != "contour/v1":
        raise ValueError(f"unsupported contour schema_version: {schema_version}")
    units = str(units_override or raw.get("units", "nm"))
    factor = _unit_to_nm_factor(units)
    axes_raw = raw.get("coordinate_axes", ["x", "y", "z"])
    if not isinstance(axes_raw, list):
        raise ValueError("contour_json.coordinate_axes must be a list")
    coordinate_axes = [str(v).lower() for v in axes_raw]
    contours_raw = raw.get("contours")
    if not isinstance(contours_raw, list) or not contours_raw:
        raise ValueError("contour_json must include non-empty contours list")

    contours: list[ContourItem] = []
    for idx, item in enumerate(contours_raw):
        if not isinstance(item, dict):
            raise ValueError(f"contours[{idx}] must be an object")
        row = {str(k): v for k, v in item.items()}
        points_xy = _project_points(
            row.get("points"),
            coordinate_axes=coordinate_axes,
            view_axes=view_axes,
        )
        material_raw = row.get("material_id")
        material_id = None if material_raw is None else int(cast(int | str, material_raw))
        contours.append(
            ContourItem(
                contour_id=str(row.get("id", f"contour_{idx}")),
                label=str(row.get("label", "global")),
                material_id=material_id,
                closed=bool(row.get("closed", True)),
                points_xy_nm=points_xy * np.float32(factor),
                meta={"source_index": idx},
            )
        )
    return ContourData(
        units="nm",
        contours=contours,
        meta={
            "path": str(input_path),
            "source_units": units,
            "coordinate_axes": coordinate_axes,
            "view_axes": list(view_axes),
        },
    )


CONTOUR_LOADERS = {
    "contour_json": load_contour_json,
}


def is_contour_input_kind(kind: str) -> bool:
    return kind in CONTOUR_LOADERS


def contour_data_to_json(data: ContourData) -> dict[str, object]:
    return {
        "schema_version": "contour/v1",
        "units": data.units,
        "coordinate_axes": ["x", "y"],
        "contours": [
            {
                "id": contour.contour_id,
                "label": contour.label,
                "material_id": contour.material_id,
                "closed": contour.closed,
                "points": contour.points_xy_nm.astype(float).tolist(),
            }
            for contour in data.contours
        ],
        "meta": dict(data.meta),
    }
