from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from wafergeo.core.types import TSDFVolume
from wafergeo.mesh.config import MeshBuildConfig


@dataclass(frozen=True)
class MeshCapabilities:
    input_types: tuple[str, ...]
    supports_anisotropic_spacing: bool
    deterministic: bool


@dataclass(frozen=True)
class MeshMethodCard:
    summary: str
    dependencies: tuple[str, ...]
    limitations: tuple[str, ...]
    install_hint: str


@dataclass(frozen=True)
class RawMesh:
    vertices: np.ndarray
    faces: np.ndarray


class MeshExtractorProtocol(Protocol):
    name: str
    version: str
    capabilities: MeshCapabilities
    method_card: MeshMethodCard

    def extract_from_tsdf(
        self,
        tsdf: TSDFVolume,
        cfg: MeshBuildConfig,
        material_ids: list[int],
    ) -> RawMesh:
        ...
