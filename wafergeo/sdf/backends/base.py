from __future__ import annotations

from typing import Protocol

import numpy as np

from wafergeo.sdf.engines.spec import EngineCapabilities, MethodCard


class EDTBackendProtocol(Protocol):
    name: str
    version: str
    capabilities: EngineCapabilities
    method_card: MethodCard

    def distance(self, mask: np.ndarray, sampling_zyx: tuple[float, ...]) -> np.ndarray:
        ...
