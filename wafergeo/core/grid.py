from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

AxisOrder = Literal["ZYX", "YXC", "XYZ", "YX"]
SampleLocation = Literal["cell_center", "grid_point"]


@dataclass(frozen=True)
class GridSpec:
    """Canonical grid specification.

    Internal convention is ZYX for 3D volumes and YX for 2D observations.
    """

    dim: int
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    axis_order: AxisOrder
    sample_location: SampleLocation
    units: str

    def __post_init__(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {self.dim}")
        if len(self.spacing) != self.dim:
            raise ValueError(
                f"spacing length must match dim: len(spacing)={len(self.spacing)}, dim={self.dim}"
            )
        if len(self.origin) != self.dim:
            raise ValueError(
                f"origin length must match dim: len(origin)={len(self.origin)}, dim={self.dim}"
            )
        if not self.units:
            raise ValueError("units must be non-empty")

        for idx, value in enumerate(self.spacing):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"spacing[{idx}] must be finite and > 0, got {value}")
        for idx, value in enumerate(self.origin):
            if not isfinite(value):
                raise ValueError(f"origin[{idx}] must be finite, got {value}")

        if self.dim == 2 and self.axis_order not in ("YX", "YXC"):
            raise ValueError(f"axis_order must be YX or YXC for dim=2, got {self.axis_order}")
        if self.dim == 3 and self.axis_order not in ("ZYX", "XYZ"):
            raise ValueError(f"axis_order must be ZYX or XYZ for dim=3, got {self.axis_order}")
