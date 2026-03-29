from __future__ import annotations

from tests.observe.helpers import build_label_volume_for_observe, build_observer_spec
from tests.sdf.helpers import register_bruteforce_engine
from wafergeo.observe.factory import create_observer, list_observers


def test_observer_contract_and_factory_registry() -> None:
    backend = register_bruteforce_engine("brute_obs_contract")
    label = build_label_volume_for_observe()
    spec = build_observer_spec(
        kind="topdown_exposed",
        backend=backend,
        contour_resample_points=0,
        mask_kind="exposed_union",
    )

    observer = create_observer("topdown_exposed")
    obs = observer.observe(label, spec)

    assert "topdown_exposed" in list_observers()
    assert obs.mask.ndim == 2
    assert obs.tsdf.shape == obs.mask.shape
    assert isinstance(obs.loops, list)
