from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from wafergeo.core.grid import GridSpec
from wafergeo.core.meta import Meta
from wafergeo.core.types import MaterialSpec, TSDFVolume


def build_material_spec() -> MaterialSpec:
    return MaterialSpec(
        ids=[0, 1, 2],
        names=["void", "resist", "oxide"],
        void_id=0,
        priority=[0, 10, 20],
        ignore_in_exposure=[True, False, False],
    )


def build_small_tsdf_volume() -> TSDFVolume:
    material = build_material_spec()
    grid = GridSpec(
        dim=3,
        spacing=(10.0, 10.0, 10.0),
        origin=(0.0, 0.0, 0.0),
        axis_order="ZYX",
        sample_location="cell_center",
        units="nm",
    )

    # Labels by design:
    # z=0
    # [[1, 1, 0],
    #  [2, 2, 0]]
    tsdf = np.full((3, 1, 2, 3), 0.8, dtype=np.float32)
    tsdf[1, 0, 0, 0] = -0.2
    tsdf[1, 0, 0, 1] = -0.2
    tsdf[2, 0, 1, 0] = -0.2
    tsdf[2, 0, 1, 1] = -0.2
    tsdf[0, 0, :, 2] = -0.2

    meta = Meta(
        schema_version="sdf/v1",
        profile_id="sdf_build_v1",
        config_hash="cfg",
        generator_version="0.1.0",
        git_commit="deadbeef",
        input_hash="input",
        created_at=datetime.now(UTC).isoformat(),
        extra={"selected_material_ids": "0,1,2", "sdf_backend": "brute"},
    )

    return TSDFVolume(
        grid=grid,
        material=material,
        mu_nm=20.0,
        tsdf=tsdf,
        meta=meta,
    )
