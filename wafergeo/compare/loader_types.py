from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from wafergeo.core.types import LabelVolume

AxisName = Literal["x", "y", "z"]


class LabelLoader(Protocol):
    def __call__(self, path: str | Path, *, void_id: int | None = None) -> LabelVolume: ...


@dataclass(frozen=True)
class ContourItem:
    contour_id: str
    label: str
    points_xy_nm: np.ndarray
    closed: bool = True
    material_id: int | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.contour_id:
            raise ValueError("contour id must be non-empty")
        if self.points_xy_nm.ndim != 2 or self.points_xy_nm.shape[1] != 2:
            raise ValueError("contour points must be shape (N,2) after projection")
        if self.points_xy_nm.shape[0] < 2:
            raise ValueError("contour must contain at least two points")


@dataclass(frozen=True)
class ContourData:
    units: str
    contours: list[ContourItem]
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.contours:
            raise ValueError("contours must be non-empty")
