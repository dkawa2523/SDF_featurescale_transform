from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from wafergeo.core.grid import GridSpec
from wafergeo.core.meta import Meta
from wafergeo.core.types import MaterialSpec, PointCloud, TSDFVolume


def _meta() -> Meta:
    return Meta(
        schema_version="schema/v1",
        profile_id="phase0",
        config_hash="abc123",
        generator_version="0.1.0",
        git_commit="deadbeef",
        input_hash="inhash",
        created_at=datetime.now(UTC).isoformat(),
        extra={"source": "test"},
    )


def _material() -> MaterialSpec:
    return MaterialSpec(
        ids=[0, 1, 2],
        names=["void", "resist", "oxide"],
        void_id=0,
        priority=[0, 10, 20],
        ignore_in_exposure=[True, False, False],
    )


def test_gridspec_valid_3d() -> None:
    grid = GridSpec(
        dim=3,
        spacing=(1.0, 2.0, 3.0),
        origin=(0.0, 0.0, 0.0),
        axis_order="ZYX",
        sample_location="cell_center",
        units="nm",
    )
    assert grid.dim == 3


def test_gridspec_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        GridSpec(
            dim=4,
            spacing=(1.0, 1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0, 0.0),
            axis_order="ZYX",  # type: ignore[arg-type]
            sample_location="cell_center",
            units="nm",
        )


def test_gridspec_spacing_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        GridSpec(
            dim=3,
            spacing=(1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
            axis_order="ZYX",
            sample_location="cell_center",
            units="nm",
        )


def test_material_spec_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        MaterialSpec(
            ids=[0, 1],
            names=["void"],
            void_id=0,
            priority=[0, 1],
            ignore_in_exposure=[True, False],
        )


def test_material_spec_allows_dataset_scale_material_counts() -> None:
    ids = list(range(9))
    material = MaterialSpec(
        ids=ids,
        names=[f"material_{idx}" for idx in ids],
        void_id=0,
        priority=ids,
        ignore_in_exposure=[idx == 0 for idx in ids],
    )

    assert material.ids == ids


def test_tsdf_volume_accepts_z1_equivalent_for_2d_case() -> None:
    grid = GridSpec(
        dim=2,
        spacing=(2.0, 2.0),
        origin=(0.0, 0.0),
        axis_order="YX",
        sample_location="cell_center",
        units="nm",
    )
    tsdf = np.zeros((3, 1, 4, 5), dtype=np.float32)
    volume = TSDFVolume(grid=grid, material=_material(), mu_nm=200.0, tsdf=tsdf, meta=_meta())
    assert volume.tsdf.shape[1] == 1


def test_tsdf_volume_present_mask_validation() -> None:
    grid = GridSpec(
        dim=3,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        axis_order="ZYX",
        sample_location="cell_center",
        units="nm",
    )
    tsdf = np.zeros((3, 1, 2, 2), dtype=np.float32)
    volume = TSDFVolume(
        grid=grid,
        material=_material(),
        mu_nm=10.0,
        tsdf=tsdf,
        present_mask=np.array([True, False, True]),
        meta=_meta(),
    )
    assert volume.present_mask is not None

    with pytest.raises(ValueError):
        TSDFVolume(
            grid=grid,
            material=_material(),
            mu_nm=10.0,
            tsdf=tsdf,
            present_mask=np.array([1, 2, 0], dtype=np.uint8),
            meta=_meta(),
        )


def test_pointcloud_validates_point_is_exposed() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    pair_code = np.array([0, 1], dtype=np.int32)
    point_is_exposed = np.array([True, False], dtype=bool)

    cloud = PointCloud(
        points=points,
        normals=normals,
        pair_code=pair_code,
        point_is_exposed=point_is_exposed,
        meta=_meta(),
    )
    assert cloud.point_is_exposed.shape == (2,)

    with pytest.raises(ValueError):
        PointCloud(
            points=points,
            normals=normals,
            pair_code=pair_code,
            point_is_exposed=np.array([1, 2], dtype=np.int32),
            meta=_meta(),
        )
