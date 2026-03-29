from __future__ import annotations

import numpy as np

from tests.sdf.helpers import register_bruteforce_engine
from wafergeo.sdf.engines.registry import (
    get_sdf_engine,
    list_sdf_engines,
    register_default_sdf_engines,
    register_sdf_engine,
)
from wafergeo.sdf.engines.spec import EngineCapabilities, MethodCard


def test_registry_selects_registered_engine() -> None:
    engine_name = register_bruteforce_engine("brute_engine_registry")
    engine = get_sdf_engine(engine_name)

    assert engine.name == engine_name
    distance = engine.distance(np.array([[True, False]]), (1.0, 1.0))
    assert distance.shape == (1, 2)


def test_engine_exposes_metadata() -> None:
    class _MetaEngine:
        def __init__(self) -> None:
            self.name = "meta_engine"
            self.version = "2.0.0"
            self.capabilities = EngineCapabilities(
                supported_inputs=("binary_mask",),
                supports_2d=True,
                supports_3d=False,
                exact=False,
                deterministic=True,
                supports_anisotropic_spacing=False,
                supports_roi_margin=False,
                gpu_accelerated=False,
            )
            self.method_card = MethodCard(
                summary="metadata test engine",
                dependencies=("none",),
                limitations=("test only",),
                references=(),
                recommended_use_cases=("unit test",),
                install_hint="",
            )

        def distance(self, mask: np.ndarray, sampling_zyx: tuple[float, ...]) -> np.ndarray:
            _ = sampling_zyx
            return np.asarray(mask, dtype=np.float32)

    register_sdf_engine(_MetaEngine())
    engine = get_sdf_engine("meta_engine")

    assert engine.version == "2.0.0"
    assert engine.capabilities.supports_2d is True
    assert engine.capabilities.supports_3d is False
    assert engine.method_card.summary == "metadata test engine"
    assert "meta_engine" in list_sdf_engines()


def test_scipy_engine_exposes_metadata_fields() -> None:
    register_default_sdf_engines()
    engine = get_sdf_engine("scipy")

    assert engine.name == "scipy"
    assert isinstance(engine.version, str) and engine.version
    assert engine.capabilities.supports_3d is True
    assert "wafergeo[scipy]" in engine.method_card.install_hint
