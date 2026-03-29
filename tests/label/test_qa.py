from __future__ import annotations

import numpy as np

from wafergeo.core.grid import GridSpec
from wafergeo.core.types import MaterialSpec
from wafergeo.label.qa import compute_label_qa


def _material() -> MaterialSpec:
    return MaterialSpec(
        ids=[0, 1, 2],
        names=["void", "resist", "oxide"],
        void_id=0,
        priority=[0, 10, 20],
        ignore_in_exposure=[True, False, False],
    )


def test_compute_label_qa_basic_fields() -> None:
    grid = GridSpec(
        dim=3,
        spacing=(2.0, 2.0, 2.0),
        origin=(0.0, 0.0, 0.0),
        axis_order="ZYX",
        sample_location="cell_center",
        units="nm",
    )
    material_id = np.array([[[0, 1], [2, 2]]], dtype=np.uint8)

    qa = compute_label_qa(
        material_id,
        grid,
        _material(),
        unknown_count=1,
        unknown_values=[9],
        converted_from_point=True,
        mask_conflict_count=2,
        notes=["test"],
    )

    assert qa.material_counts[2] == 2
    assert qa.material_volume_nm3[2] == 16.0
    assert qa.void_fraction == 0.25
    assert qa.unknown_rate == 0.25
    assert qa.converted_from_point is True
    assert qa.mask_conflict_count == 2
