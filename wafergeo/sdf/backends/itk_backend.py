from __future__ import annotations

import numpy as np

from wafergeo.sdf.engines.spec import EngineCapabilities, MethodCard
from wafergeo.sdf.errors import OptionalDependencyUnavailableError


def itk_maurer_distance(mask: np.ndarray, sampling_zyx: tuple[float, ...]) -> np.ndarray:
    _ = mask
    _ = sampling_zyx
    raise OptionalDependencyUnavailableError(
        engine_name="itk_maurer",
        dependency="itk",
        install_hint="pip install 'wafergeo[itk]'",
    )


class ItkMaurerEDTEngine:
    name = "itk_maurer"
    version = "0.1.0"
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
        summary="ITK SignedMaurer EDT backend stub.",
        dependencies=("itk",),
        limitations=("Phase 2.1 stub backend.",),
        references=("itk::SignedMaurerDistanceMapImageFilter",),
        recommended_use_cases=("Future exact-EDT cross-check.",),
        install_hint="pip install 'wafergeo[itk]'",
    )

    def distance(self, mask: np.ndarray, sampling_zyx: tuple[float, ...]) -> np.ndarray:
        return itk_maurer_distance(mask, sampling_zyx)
