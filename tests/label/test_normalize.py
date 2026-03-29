from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from wafergeo.core.types import MaterialSpec
from wafergeo.io.vti_reader import RawVtiImage
from wafergeo.label.errors import InvalidArrayShapeError, UnknownMaterialIdError
from wafergeo.label.normalize import LabelNormalizeConfig, normalize_raw_to_label


def _material() -> MaterialSpec:
    return MaterialSpec(
        ids=[0, 1, 2],
        names=["void", "resist", "oxide"],
        void_id=0,
        priority=[0, 10, 20],
        ignore_in_exposure=[True, False, False],
    )


def _base_raw(
    arrays: dict[str, np.ndarray],
    locations: dict[str, str],
    dims_xyz: tuple[int, int, int] = (3, 2, 1),
) -> RawVtiImage:
    return RawVtiImage(
        spacing_xyz=(10.0, 20.0, 30.0),
        origin_xyz=(100.0, 200.0, 300.0),
        dims_xyz=dims_xyz,
        arrays=arrays,
        array_location=locations,
        vtk_meta={},
    )


def _normalize(raw: RawVtiImage, config: LabelNormalizeConfig) -> tuple[object, object]:
    return normalize_raw_to_label(
        raw,
        _material(),
        config,
        source_path="synthetic.vti",
        input_hash="input-hash",
        generator_version="0.1.0",
        git_commit="deadbeef",
        created_at=datetime.now(UTC).isoformat(),
    )


def test_normalize_cell_scalar_to_labelvolume() -> None:
    array_xyz = np.array([[[0], [1]], [[1], [2]], [[0], [2]]], dtype=np.int32)
    raw = _base_raw({"material_id": array_xyz}, {"material_id": "cell"}, dims_xyz=(3, 2, 1))

    label, qa = _normalize(raw, LabelNormalizeConfig())

    assert label.grid.axis_order == "ZYX"
    assert label.grid.sample_location == "cell_center"
    assert label.material_id.shape == (1, 2, 3)
    assert qa.converted_from_point is False


def test_label_normalize_default_policy_is_nearest() -> None:
    assert LabelNormalizeConfig().point_to_cell_policy == "nearest"


def test_point_scalar_majority_tie_break_by_priority() -> None:
    arr_xyz = np.empty((2, 2, 2), dtype=np.int32)
    arr_xyz[0, 0, 0] = 1
    arr_xyz[1, 0, 0] = 2
    arr_xyz[0, 1, 0] = 1
    arr_xyz[1, 1, 0] = 2
    arr_xyz[0, 0, 1] = 2
    arr_xyz[1, 0, 1] = 1
    arr_xyz[0, 1, 1] = 2
    arr_xyz[1, 1, 1] = 1

    raw = _base_raw({"material_id": arr_xyz}, {"material_id": "point"}, dims_xyz=(2, 2, 2))
    config = LabelNormalizeConfig(point_to_cell_policy="majority")

    label, qa = _normalize(raw, config)

    assert label.material_id.shape == (1, 1, 1)
    assert int(label.material_id[0, 0, 0]) == 2
    assert qa.converted_from_point is True


def test_point_to_cell_majority_nearest_tie_exists() -> None:
    arr_xyz = np.empty((2, 2, 2), dtype=np.int32)
    arr_xyz[0, 0, 0] = 1
    arr_xyz[1, 0, 0] = 2
    arr_xyz[0, 1, 0] = 1
    arr_xyz[1, 1, 0] = 2
    arr_xyz[0, 0, 1] = 2
    arr_xyz[1, 0, 1] = 1
    arr_xyz[0, 1, 1] = 2
    arr_xyz[1, 1, 1] = 1

    raw = _base_raw({"material_id": arr_xyz}, {"material_id": "point"}, dims_xyz=(2, 2, 2))
    label_nearest, _ = _normalize(
        raw,
        LabelNormalizeConfig(point_to_cell_policy="nearest"),
    )
    label_majority, _ = _normalize(
        raw,
        LabelNormalizeConfig(point_to_cell_policy="majority"),
    )
    label_new, _ = _normalize(
        raw,
        LabelNormalizeConfig(point_to_cell_policy="majority_nearest_tie"),
    )

    assert int(label_majority.material_id[0, 0, 0]) == 2
    assert int(label_nearest.material_id[0, 0, 0]) == 1
    assert int(label_new.material_id[0, 0, 0]) == 1


def test_unknown_label_policy_error_raises() -> None:
    array_xyz = np.array([[[9], [1]], [[1], [2]], [[0], [2]]], dtype=np.int32)
    raw = _base_raw({"material_id": array_xyz}, {"material_id": "cell"}, dims_xyz=(3, 2, 1))

    with pytest.raises(UnknownMaterialIdError):
        _normalize(raw, LabelNormalizeConfig(unknown_label_policy="error"))


def test_unknown_label_policy_map_to_void_records_qa() -> None:
    array_xyz = np.array([[[9], [1]], [[1], [2]], [[0], [2]]], dtype=np.int32)
    raw = _base_raw({"material_id": array_xyz}, {"material_id": "cell"}, dims_xyz=(3, 2, 1))
    config = LabelNormalizeConfig(unknown_label_policy="map_to_void", unknown_to_void_id=0)

    label, qa = _normalize(raw, config)

    assert qa.unknown_count == 1
    assert qa.unknown_values == [9]
    assert 9 not in np.unique(label.material_id)


def test_mask_merge_priority_works() -> None:
    mask_resist = np.zeros((3, 2, 1), dtype=np.uint8)
    mask_oxide = np.zeros((3, 2, 1), dtype=np.uint8)
    mask_resist[0, 0, 0] = 1
    mask_resist[1, 0, 0] = 1
    mask_oxide[1, 0, 0] = 1
    mask_oxide[2, 1, 0] = 1

    raw = _base_raw(
        {"mask_resist": mask_resist, "mask_oxide": mask_oxide},
        {"mask_resist": "cell", "mask_oxide": "cell"},
        dims_xyz=(3, 2, 1),
    )
    config = LabelNormalizeConfig(
        mask_merge_policy="priority",
        label_array_candidates=["material_id"],
    )

    label, qa = _normalize(raw, config)

    assert int(label.material_id[0, 0, 1]) == 2
    assert qa.mask_conflict_count == 1


def test_mask_merge_error_on_conflict_raises() -> None:
    mask_resist = np.zeros((3, 2, 1), dtype=np.uint8)
    mask_oxide = np.zeros((3, 2, 1), dtype=np.uint8)
    mask_resist[1, 0, 0] = 1
    mask_oxide[1, 0, 0] = 1

    raw = _base_raw(
        {"mask_resist": mask_resist, "mask_oxide": mask_oxide},
        {"mask_resist": "cell", "mask_oxide": "cell"},
        dims_xyz=(3, 2, 1),
    )
    config = LabelNormalizeConfig(
        mask_merge_policy="error_on_conflict",
        label_array_candidates=["material_id"],
    )

    with pytest.raises(InvalidArrayShapeError):
        _normalize(raw, config)


def test_z1_case_preserved() -> None:
    array_xyz = np.array([[[0], [1]], [[1], [2]], [[0], [2]], [[0], [1]]], dtype=np.int32)
    raw = _base_raw({"material_id": array_xyz}, {"material_id": "cell"}, dims_xyz=(4, 2, 1))

    label, _ = _normalize(raw, LabelNormalizeConfig())

    assert label.material_id.shape[0] == 1
    assert label.grid.dim == 3


def test_legacy_layout_compat() -> None:
    point_zyx = np.array(
        [
            [[0, 1], [1, 2]],
            [[2, 1], [0, 2]],
        ],
        dtype=np.int32,
    )
    point_xyz = point_zyx.transpose(2, 1, 0)
    raw = RawVtiImage(
        spacing_xyz=(10.0, 20.0, 30.0),
        origin_xyz=(100.0, 200.0, 300.0),
        dims_xyz=(2, 2, 2),
        arrays={"material_id": point_xyz.ravel(order="C")},
        array_location={"material_id": "point"},
        vtk_meta={"source": "synthetic"},
    )

    label_auto, _ = _normalize(raw, LabelNormalizeConfig(point_to_cell_policy="nearest"))
    label_legacy, _ = _normalize(
        raw,
        LabelNormalizeConfig(
            point_to_cell_policy="nearest",
            flat_array_layout="legacy_xyz_transpose",
        ),
    )
    np.testing.assert_array_equal(label_auto.material_id, label_legacy.material_id)
