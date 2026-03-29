from __future__ import annotations

import csv
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RawContourLoop:
    loop_id: str
    role: str
    points_xy: np.ndarray
    is_closed_hint: bool | None = None

    def __post_init__(self) -> None:
        if not self.loop_id:
            raise ValueError("loop_id must be non-empty")
        if self.points_xy.ndim != 2 or self.points_xy.shape[1] != 2:
            raise ValueError("points_xy must be shape (N,2)")
        if self.points_xy.shape[0] < 2:
            raise ValueError("points_xy must contain at least 2 points")


@dataclass(frozen=True)
class RawContourSet:
    coord_system: str
    units: str
    loops_raw: list[RawContourLoop]
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.coord_system:
            raise ValueError("coord_system must be non-empty")
        if not self.units:
            raise ValueError("units must be non-empty")
        if not self.loops_raw:
            raise ValueError("loops_raw must be non-empty")


@dataclass(frozen=True)
class SEMImageRaw:
    image: np.ndarray
    pixel_size_nm: float | None
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image.ndim != 2:
            raise ValueError("SEM image must be 2D grayscale")


def _to_points_xy(points_raw: Any, *, where: str) -> np.ndarray:
    if not isinstance(points_raw, list):
        raise ValueError(f"{where}.points must be a list")
    pts = np.asarray(points_raw, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"{where}.points must be shape (N,2)")
    if pts.shape[0] < 2:
        raise ValueError(f"{where}.points must contain at least 2 points")
    return pts


def read_contours_json(path: str | Path) -> RawContourSet:
    file_path = Path(path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("contour JSON root must be an object")

    loops_raw = raw.get("loops")
    if not isinstance(loops_raw, list) or len(loops_raw) == 0:
        raise ValueError("contour JSON must include non-empty 'loops' list")

    loops: list[RawContourLoop] = []
    for idx, loop_raw in enumerate(loops_raw):
        if not isinstance(loop_raw, dict):
            raise ValueError(f"loops[{idx}] must be an object")
        row = {str(k): v for k, v in loop_raw.items()}
        loop_id = str(row.get("id", f"loop_{idx}"))
        role = str(row.get("role", "outer"))
        points = _to_points_xy(row.get("points"), where=f"loops[{idx}]")
        is_closed_hint: bool | None
        if "is_closed" in row:
            is_closed_hint = bool(row["is_closed"])
        else:
            is_closed_hint = None
        loops.append(
            RawContourLoop(
                loop_id=loop_id,
                role=role,
                points_xy=points,
                is_closed_hint=is_closed_hint,
            )
        )

    return RawContourSet(
        coord_system=str(raw.get("coord_system", "pixel")),
        units=str(raw.get("units", "px")),
        loops_raw=loops,
        meta={"path": str(file_path)},
    )


def _coord_columns(fieldnames: list[str]) -> tuple[str, str]:
    candidates = [("x", "y"), ("u", "v"), ("px", "py")]
    lowered = {name.lower(): name for name in fieldnames}
    for x_name, y_name in candidates:
        if x_name in lowered and y_name in lowered:
            return lowered[x_name], lowered[y_name]
    raise ValueError("CSV must contain coordinate columns x/y or u/v or px/py")


def _parse_optional_bool(value: object | None) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def read_contours_csv(path: str | Path) -> RawContourSet:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV must include header row")
        x_col, y_col = _coord_columns(list(reader.fieldnames))

        grouped: OrderedDict[str, dict[str, object]] = OrderedDict()
        row_count = 0
        for row_index, row in enumerate(reader, start=2):
            row_count += 1
            try:
                x_val = float(row[x_col])
                y_val = float(row[y_col])
            except Exception as exc:
                raise ValueError(f"invalid coordinate at CSV line {row_index}") from exc

            loop_id = str(row.get("loop_id", "loop_0"))
            role = str(row.get("role", "outer"))
            is_closed_hint = _parse_optional_bool(row.get("is_closed"))

            if loop_id not in grouped:
                grouped[loop_id] = {
                    "role": role,
                    "is_closed_hint": is_closed_hint,
                    "points": [],
                }
            cast_points = grouped[loop_id]["points"]
            assert isinstance(cast_points, list)
            cast_points.append((x_val, y_val))

        if row_count == 0:
            raise ValueError("CSV contour file has no data rows")

    loops: list[RawContourLoop] = []
    for loop_id, data in grouped.items():
        points_xy = np.asarray(data["points"], dtype=np.float32)
        role = str(data["role"])
        is_closed_hint_obj = data["is_closed_hint"]
        is_closed_hint = _parse_optional_bool(is_closed_hint_obj)
        loops.append(
            RawContourLoop(
                loop_id=loop_id,
                role=role,
                points_xy=points_xy,
                is_closed_hint=is_closed_hint,
            )
        )

    return RawContourSet(
        coord_system="pixel",
        units="px",
        loops_raw=loops,
        meta={"path": str(file_path)},
    )


def read_sem_image(path: str | Path) -> SEMImageRaw:
    file_path = Path(path)
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "SEM image reader requires Pillow. Install: pip install 'wafergeo[sem]'"
        ) from exc

    with Image.open(file_path) as img:
        gray = img.convert("L")
        image = np.asarray(gray, dtype=np.uint8)
        info = dict(img.info)

    pixel_size_nm: float | None = None
    for key in ("pixel_size_nm", "PixelSizeNm", "pixel_nm"):
        if key in info:
            try:
                pixel_size_nm = float(info[key])
            except (TypeError, ValueError):
                pixel_size_nm = None
            break

    return SEMImageRaw(
        image=image,
        pixel_size_nm=pixel_size_nm,
        meta={"path": str(file_path), "format": str(file_path.suffix.lower())},
    )
