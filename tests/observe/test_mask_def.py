from __future__ import annotations

import numpy as np

from tests.observe.helpers import build_material_spec
from wafergeo.observe.mask_def import mask2d_from_exposed_id, mask2d_from_slice_label
from wafergeo.observe.spec import MaskDefSpec


def test_mask_def_binary_solid_and_material_union() -> None:
    material = build_material_spec()
    exposed = np.array(
        [
            [0, 1, 2],
            [2, 0, 1],
        ],
        dtype=np.uint8,
    )

    binary = mask2d_from_exposed_id(exposed, MaskDefSpec(kind="binary_solid"), material)
    union = mask2d_from_exposed_id(
        exposed,
        MaskDefSpec(kind="exposed_union", include_materials=("resist",)),
        material,
    )

    assert binary.dtype == np.uint8
    assert int(binary.sum()) == 4
    assert int(union.sum()) == 2


def test_mask_def_slice_material_union() -> None:
    material = build_material_spec()
    label2d = np.array([[0, 1], [2, 1]], dtype=np.uint8)

    mask = mask2d_from_slice_label(
        label2d,
        MaskDefSpec(kind="material_union", include_ids=(2,)),
        material,
    )

    assert mask.tolist() == [[0, 0], [1, 0]]
