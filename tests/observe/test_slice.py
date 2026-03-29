from __future__ import annotations

from tests.observe.helpers import build_label_volume_for_observe, build_observer_spec
from tests.sdf.helpers import register_bruteforce_engine
from wafergeo.observe.slice import SliceObserver


def test_slice_observer_runs_and_slab_expands_mask() -> None:
    backend = register_bruteforce_engine("brute_obs_slice")
    label = build_label_volume_for_observe()

    base_spec = build_observer_spec(
        kind="slice",
        backend=backend,
        contour_resample_points=0,
        mask_kind="binary_solid",
        params={"axis": "z", "coord_nm": 10.0, "slab_thickness_nm": 0.0},
    )
    slab_spec = build_observer_spec(
        kind="slice",
        backend=backend,
        contour_resample_points=0,
        mask_kind="binary_solid",
        params={"axis": "z", "coord_nm": 10.0, "slab_thickness_nm": 20.0},
    )

    observer = SliceObserver()
    obs_base = observer.observe(label, base_spec)
    obs_slab = observer.observe(label, slab_spec)

    assert obs_base.mask.shape == obs_slab.mask.shape
    assert int(obs_slab.mask.sum()) >= int(obs_base.mask.sum())
    assert obs_slab.meta.extra["slice_axis"] == "z"
    assert obs_slab.meta.extra["source_kind"] == "label"
