from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wafergeo.core.grid import GridSpec
from wafergeo.core.types import MaterialSpec


@dataclass(frozen=True)
class LabelQA:
    material_counts: dict[int, int]
    material_volume_nm3: dict[int, float]
    void_fraction: float
    unknown_count: int
    unknown_rate: float
    unknown_values: list[int]
    converted_from_point: bool
    mask_conflict_count: int
    notes: list[str] = field(default_factory=list)


def compute_label_qa(
    material_id_zyx: np.ndarray,
    grid: GridSpec,
    material: MaterialSpec,
    *,
    unknown_count: int,
    unknown_values: list[int],
    converted_from_point: bool,
    mask_conflict_count: int,
    notes: list[str] | None = None,
) -> LabelQA:
    total_voxels = int(material_id_zyx.size)
    voxel_volume_nm3 = float(np.prod(np.asarray(grid.spacing, dtype=np.float64)))

    material_counts: dict[int, int] = {}
    material_volume_nm3: dict[int, float] = {}
    for material_id in material.ids:
        count = int(np.sum(material_id_zyx == material_id))
        material_counts[material_id] = count
        material_volume_nm3[material_id] = float(count) * voxel_volume_nm3

    void_count = material_counts.get(material.void_id, 0)
    void_fraction = float(void_count) / float(total_voxels) if total_voxels > 0 else 0.0
    unknown_rate = float(unknown_count) / float(total_voxels) if total_voxels > 0 else 0.0

    return LabelQA(
        material_counts=material_counts,
        material_volume_nm3=material_volume_nm3,
        void_fraction=void_fraction,
        unknown_count=int(unknown_count),
        unknown_rate=unknown_rate,
        unknown_values=sorted(int(v) for v in unknown_values),
        converted_from_point=bool(converted_from_point),
        mask_conflict_count=int(mask_conflict_count),
        notes=list(notes or []),
    )
