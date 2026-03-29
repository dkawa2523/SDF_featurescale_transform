from __future__ import annotations

from tests.observe.helpers import (
    build_label_volume_for_observe,
    build_observer_spec,
    build_tsdf_and_mesh,
)
from tests.sdf.helpers import register_bruteforce_engine
from wafergeo.observe.topdown import TopDownExposedObserver


def test_topdown_observer_runs_on_label_input() -> None:
    backend = register_bruteforce_engine("brute_obs_topdown_label")
    label = build_label_volume_for_observe()
    spec = build_observer_spec(
        kind="topdown_exposed",
        backend=backend,
        contour_resample_points=0,
        mask_kind="exposed_union",
    )

    obs = TopDownExposedObserver().observe(label, spec)

    assert obs.mask.shape == label.material_id.shape[1:]
    assert obs.tsdf.shape == obs.mask.shape
    assert obs.meta.extra["observer_kind"] == "topdown_exposed"
    assert obs.meta.extra["source_kind"] == "label"
    assert "exposed_id" in obs.debug_maps


def test_topdown_observer_runs_on_tsdf_input() -> None:
    backend = register_bruteforce_engine("brute_obs_topdown_tsdf")
    label = build_label_volume_for_observe()
    tsdf, _ = build_tsdf_and_mesh(label, backend)
    spec = build_observer_spec(
        kind="topdown_exposed",
        backend=backend,
        contour_resample_points=0,
        mask_kind="exposed_union",
    )

    obs = TopDownExposedObserver().observe(tsdf, spec)

    assert obs.mask.shape == label.material_id.shape[1:]
    assert float(obs.tsdf.min()) >= -1.0
    assert float(obs.tsdf.max()) <= 1.0
    assert obs.meta.extra["source_kind"] == "tsdf"
