from __future__ import annotations

import numpy as np

from wafergeo.sdf.engines.spec import EngineCapabilities, MethodCard
from wafergeo.sdf.errors import (
    EDTComputationError,
    OptionalDependencyUnavailableError,
)


def scipy_edt_distance(mask: np.ndarray, sampling_zyx: tuple[float, ...]) -> np.ndarray:
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover - tested via integration path
        raise OptionalDependencyUnavailableError(
            engine_name="scipy",
            dependency="scipy",
            install_hint="pip install 'wafergeo[scipy]'",
        ) from exc

    mask_bool = np.asarray(mask, dtype=bool)
    try:
        distance = distance_transform_edt(mask_bool, sampling=sampling_zyx)
    except Exception as exc:  # pragma: no cover - backend error mapping
        raise EDTComputationError(f"scipy distance_transform_edt failed: {exc}") from exc
    return distance.astype(np.float32, copy=False)


class ScipyEDTEngine:
    name = "scipy"
    version = "1.0.0"
    capabilities = EngineCapabilities(
        supported_inputs=("binary_mask",),
        supports_2d=True,
        supports_3d=True,
        exact=True,
        deterministic=True,
        supports_anisotropic_spacing=True,
        supports_roi_margin=True,
        gpu_accelerated=False,
    )
    method_card = MethodCard(
        summary="Exact EDT via scipy.ndimage.distance_transform_edt.",
        dependencies=("scipy>=1.11",),
        limitations=("CPU implementation.",),
        references=("scipy.ndimage.distance_transform_edt",),
        recommended_use_cases=("Default exact EDT backend.",),
        install_hint="pip install 'wafergeo[scipy]'",
    )

    def distance(self, mask: np.ndarray, sampling_zyx: tuple[float, ...]) -> np.ndarray:
        return scipy_edt_distance(mask, sampling_zyx)


class ScipyEDTBackend(ScipyEDTEngine):
    """Backward-compatible alias for previous backend class name."""
