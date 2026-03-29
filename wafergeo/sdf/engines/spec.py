from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class EngineCapabilities:
    supported_inputs: tuple[str, ...] = ("binary_mask",)
    supports_2d: bool = True
    supports_3d: bool = True
    exact: bool = True
    deterministic: bool = True
    supports_anisotropic_spacing: bool = True
    supports_roi_margin: bool = True
    gpu_accelerated: bool = False


@dataclass(frozen=True)
class MethodCard:
    summary: str
    dependencies: tuple[str, ...]
    limitations: tuple[str, ...]
    references: tuple[str, ...]
    recommended_use_cases: tuple[str, ...]
    install_hint: str


class SDFEngineProtocol(Protocol):
    name: str
    version: str
    capabilities: EngineCapabilities
    method_card: MethodCard

    def distance(self, mask: np.ndarray, sampling_zyx: tuple[float, ...]) -> np.ndarray:
        ...


def method_card_to_dict(method_card: MethodCard) -> dict[str, object]:
    return asdict(method_card)


def capabilities_to_dict(capabilities: EngineCapabilities) -> dict[str, object]:
    return asdict(capabilities)
