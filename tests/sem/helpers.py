from __future__ import annotations

from wafergeo.core.grid import GridSpec
from wafergeo.sem.spec import (
    ContourNormalizeSpec,
    SEMInputSpec,
    SEMOverlaySpec,
    SEMPrepareSpecV1,
    SEMQASpec,
    SEMTSDFSpec,
    SEMWeightSpec,
)


def build_sem_spec(
    *,
    coord_system: str = "nm",
    units: str = "nm",
    pixel_size_nm: float | None = None,
    pixel_y_policy: str = "flip_y",
    tsdf_mode: str = "auto",
    distance_backend: str = "scipy",
    overlay_enable: bool = False,
    target_shape_yx: tuple[int, int] | None = None,
) -> SEMPrepareSpecV1:
    return SEMPrepareSpecV1(
        schema_version="sem_prepare/v1",
        profile_id="sem_prepare_test",
        target_grid_2d=GridSpec(
            dim=2,
            spacing=(10.0, 10.0),
            origin=(0.0, 0.0),
            axis_order="YX",
            sample_location="cell_center",
            units="nm",
        ),
        target_shape_yx=target_shape_yx,
        input=SEMInputSpec(
            contour_format="auto",
            coord_system=coord_system,  # type: ignore[arg-type]
            units=units,  # type: ignore[arg-type]
            pixel_size_nm=pixel_size_nm,
            pixel_y_policy=pixel_y_policy,  # type: ignore[arg-type]
        ),
        normalize=ContourNormalizeSpec(
            close_tol_nm=5.0,
            enforce_orientation=True,
            resample_points_closed=32,
            resample_points_open=16,
        ),
        tsdf=SEMTSDFSpec(
            mode=tsdf_mode,  # type: ignore[arg-type]
            mu_nm=30.0,
            open_tube_radius_nm=15.0,
            distance_backend=distance_backend,
        ),
        weight=SEMWeightSpec(mode="uniform", default_weight=1.0),
        qa=SEMQASpec(min_mask_fraction=0.0, max_open_contours=8),
        overlay=SEMOverlaySpec(enable=overlay_enable, draw_contours=True),
    )
