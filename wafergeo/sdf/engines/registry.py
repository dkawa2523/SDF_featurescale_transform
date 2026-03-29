from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from wafergeo.core.registry import sdf_backend_registry
from wafergeo.sdf.engines.spec import EngineCapabilities, MethodCard, SDFEngineProtocol


@dataclass
class LegacyCallableEngineAdapter:
    """Adapter for legacy callable backends stored in the registry."""

    name: str
    distance_fn: Any
    version: str = "legacy"
    capabilities: EngineCapabilities = EngineCapabilities(
        supported_inputs=("binary_mask",),
        supports_2d=True,
        supports_3d=True,
        exact=False,
        deterministic=True,
        supports_anisotropic_spacing=True,
        supports_roi_margin=True,
        gpu_accelerated=False,
    )
    method_card: MethodCard = MethodCard(
        summary="Legacy callable backend adapter.",
        dependencies=(),
        limitations=("Missing explicit engine metadata.",),
        references=(),
        recommended_use_cases=("Backward compatibility.",),
        install_hint="",
    )

    def distance(self, mask: np.ndarray, sampling_zyx: tuple[float, ...]) -> np.ndarray:
        return np.asarray(self.distance_fn(mask, sampling_zyx), dtype=np.float32)


def _is_engine_like(value: object) -> bool:
    return (
        hasattr(value, "name")
        and hasattr(value, "version")
        and hasattr(value, "capabilities")
        and hasattr(value, "method_card")
        and hasattr(value, "distance")
        and callable(value.distance)  # type: ignore[attr-defined]
    )


def _coerce_to_engine(key: str, value: object) -> SDFEngineProtocol:
    if _is_engine_like(value):
        return cast(SDFEngineProtocol, value)

    if callable(value):
        return LegacyCallableEngineAdapter(name=key, distance_fn=value)

    if hasattr(value, "distance") and callable(value.distance):  # type: ignore[attr-defined]
        return LegacyCallableEngineAdapter(name=key, distance_fn=value.distance)  # type: ignore[attr-defined]

    raise TypeError(f"sdf_backend '{key}' must be callable or implement SDFEngineProtocol")


def register_sdf_engine(engine: SDFEngineProtocol, *, aliases: tuple[str, ...] = ()) -> None:
    if not engine.name:
        raise ValueError("engine.name must be non-empty")

    sdf_backend_registry.register(engine.name, engine, override=True)
    for alias in aliases:
        if not alias:
            raise ValueError("engine alias must be non-empty")
        sdf_backend_registry.register(alias, engine, override=True)


def list_sdf_engines() -> tuple[str, ...]:
    register_default_sdf_engines()
    return sdf_backend_registry.list()


def get_sdf_engine(name: str) -> SDFEngineProtocol:
    register_default_sdf_engines()
    raw = sdf_backend_registry.get(name)
    return _coerce_to_engine(name, raw)


def register_default_sdf_engines() -> None:
    from wafergeo.sdf.backends.cupy_backend import CupyJFADTEngine
    from wafergeo.sdf.backends.itk_backend import ItkMaurerEDTEngine
    from wafergeo.sdf.backends.scipy_backend import ScipyEDTEngine

    if "scipy" not in sdf_backend_registry.list():
        register_sdf_engine(ScipyEDTEngine())
    if "itk_maurer" not in sdf_backend_registry.list():
        register_sdf_engine(ItkMaurerEDTEngine(), aliases=("itk",))
    if "cupy_jfa" not in sdf_backend_registry.list():
        register_sdf_engine(CupyJFADTEngine(), aliases=("cupy",))
