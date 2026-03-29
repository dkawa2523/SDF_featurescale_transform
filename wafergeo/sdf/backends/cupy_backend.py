from __future__ import annotations

import numpy as np

from wafergeo.sdf.engines.spec import EngineCapabilities, MethodCard
from wafergeo.sdf.errors import OptionalDependencyUnavailableError


def cupy_jfa_distance(mask: np.ndarray, sampling_zyx: tuple[float, ...]) -> np.ndarray:
    _ = mask
    _ = sampling_zyx
    raise OptionalDependencyUnavailableError(
        engine_name="cupy_jfa",
        dependency="cupy",
        install_hint="pip install 'wafergeo[cuda]'",
    )


class CupyJFADTEngine:
    name = "cupy_jfa"
    version = "0.1.0"
    capabilities = EngineCapabilities(
        supported_inputs=("binary_mask",),
        supports_2d=True,
        supports_3d=True,
        exact=False,
        deterministic=True,
        supports_anisotropic_spacing=True,
        supports_roi_margin=True,
        gpu_accelerated=True,
    )
    method_card = MethodCard(
        summary="CuPy JFA EDT backend stub.",
        dependencies=("cupy",),
        limitations=("Phase 2.1 stub backend.",),
        references=("Jump Flooding Algorithm",),
        recommended_use_cases=("Future high-throughput approximate EDT.",),
        install_hint="pip install 'wafergeo[cuda]'",
    )

    def distance(self, mask: np.ndarray, sampling_zyx: tuple[float, ...]) -> np.ndarray:
        return cupy_jfa_distance(mask, sampling_zyx)
