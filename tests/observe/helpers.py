from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from wafergeo.core.grid import GridSpec
from wafergeo.core.meta import Meta
from wafergeo.core.types import LabelVolume, MaterialSpec
from wafergeo.mesh.build import build_mesh_from_tsdf
from wafergeo.mesh.config import MeshBuildConfig
from wafergeo.observe.spec import ContourSpec, MaskDefSpec, ObserverSpecV2, Tsdf2DSpec
from wafergeo.sdf.build import build_tsdf_volume
from wafergeo.sdf.config import SDFBuildConfig


def build_material_spec() -> MaterialSpec:
    return MaterialSpec(
        ids=[0, 1, 2],
        names=["void", "resist", "oxide"],
        void_id=0,
        priority=[0, 10, 20],
        ignore_in_exposure=[True, False, False],
    )


def build_label_volume_for_observe() -> LabelVolume:
    grid = GridSpec(
        dim=3,
        spacing=(10.0, 10.0, 10.0),
        origin=(0.0, 0.0, 0.0),
        axis_order="ZYX",
        sample_location="cell_center",
        units="nm",
    )
    meta = Meta(
        schema_version="label/v1",
        profile_id="ingest_label_v1",
        config_hash="cfg",
        generator_version="0.1.0",
        git_commit="deadbeef",
        input_hash="input",
        created_at=datetime.now(UTC).isoformat(),
        extra={"source": "synthetic"},
    )

    label = np.zeros((3, 5, 5), dtype=np.uint8)
    label[0, 1:4, 1:4] = 1
    label[1, 2:4, 2:4] = 1
    label[2, 0:2, 0:2] = 2

    return LabelVolume(
        grid=grid,
        material=build_material_spec(),
        material_id=label,
        meta=meta,
    )


def build_observer_spec(
    *,
    kind: str,
    backend: str,
    contour_resample_points: int = 0,
    allow_missing_backend: bool = True,
    mask_kind: str = "binary_solid",
    params: dict[str, object] | None = None,
) -> ObserverSpecV2:
    grid2d = GridSpec(
        dim=2,
        spacing=(10.0, 10.0),
        origin=(0.0, 0.0),
        axis_order="YX",
        sample_location="cell_center",
        units="nm",
    )

    return ObserverSpecV2(
        schema_version="observer/v2",
        name=f"test_{kind}",
        kind=kind,
        target_grid_2d=grid2d,
        roi={},
        mask_definition=MaskDefSpec(kind=mask_kind),
        tsdf2d=Tsdf2DSpec(mu_nm=30.0, engine=backend, band_only=True),
        contour=ContourSpec(
            source="tsdf",
            level=0.0,
            resample_points=contour_resample_points,
            backend="skimage",
            allow_missing_backend=allow_missing_backend,
        ),
        params=params or {},
        debug={"save_exposed_id": True, "save_height_map": True},
        qa={"max_open_contours": 1},
    )


def build_tsdf_and_mesh(label: LabelVolume, backend: str):
    tsdf, _ = build_tsdf_volume(
        label,
        SDFBuildConfig(mu_nm=20.0, backend=backend, boundary_features=False),
    )
    mesh, _, _ = build_mesh_from_tsdf(
        tsdf,
        MeshBuildConfig(
            backend="naive_interface",
            mode="interface_mesh",
            sample_points_n=32,
            sample_seed=0,
            channel_material_ids=[0, 1, 2],
        ),
    )
    return tsdf, mesh
