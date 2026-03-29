from __future__ import annotations

from tests.observe.helpers import (
    build_label_volume_for_observe,
    build_observer_spec,
    build_tsdf_and_mesh,
)
from tests.sdf.helpers import register_bruteforce_engine
from wafergeo.observe.slice import SliceObserver
from wafergeo.observe.topdown import TopDownExposedObserver


def test_observers_run_with_mesh_input_via_voxelize() -> None:
    backend = register_bruteforce_engine("brute_obs_mesh")
    label = build_label_volume_for_observe()
    tsdf, mesh = build_tsdf_and_mesh(label, backend)
    shape_zyx = list(tsdf.tsdf.shape[1:])

    top_spec = build_observer_spec(
        kind="topdown_exposed",
        backend=backend,
        contour_resample_points=0,
        mask_kind="exposed_union",
        params={"mesh_shape_zyx": shape_zyx},
    )
    slice_spec = build_observer_spec(
        kind="slice",
        backend=backend,
        contour_resample_points=0,
        mask_kind="binary_solid",
        params={"axis": "z", "coord_nm": 0.0, "mesh_shape_zyx": shape_zyx},
    )

    top_obs = TopDownExposedObserver().observe(mesh, top_spec)
    slice_obs = SliceObserver().observe(mesh, slice_spec)

    assert top_obs.mask.shape == label.material_id.shape[1:]
    assert slice_obs.mask.shape == label.material_id.shape[1:]
    assert top_obs.meta.extra["source_kind"] == "mesh"
    assert slice_obs.meta.extra["source_kind"] == "mesh"
