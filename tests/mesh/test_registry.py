from __future__ import annotations

import numpy as np

from tests.mesh.helpers import build_small_tsdf_volume
from wafergeo.mesh.config import MeshBuildConfig
from wafergeo.mesh.extractors.base import MeshCapabilities, MeshMethodCard, RawMesh
from wafergeo.mesh.extractors.registry import (
    get_mesh_extractor,
    list_mesh_extractors,
    register_mesh_extractor,
)


def test_mesh_extractor_registry_selects_registered_extractor() -> None:
    class _DummyExtractor:
        name = "dummy_mesh"
        version = "1.0.0"
        capabilities = MeshCapabilities(
            input_types=("tsdf",),
            supports_anisotropic_spacing=True,
            deterministic=True,
        )
        method_card = MeshMethodCard(
            summary="dummy",
            dependencies=(),
            limitations=(),
            install_hint="",
        )

        def extract_from_tsdf(self, tsdf, cfg: MeshBuildConfig, material_ids: list[int]) -> RawMesh:
            _ = tsdf, cfg, material_ids
            return RawMesh(
                vertices=np.array(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                ),
                faces=np.array([[0, 1, 2]], dtype=np.int32),
            )

    register_mesh_extractor(_DummyExtractor())
    ext = get_mesh_extractor("dummy_mesh")

    assert ext.name == "dummy_mesh"
    assert "dummy_mesh" in list_mesh_extractors()


def test_naive_extractor_registered_by_default() -> None:
    ext = get_mesh_extractor("naive_interface")
    tsdf = build_small_tsdf_volume()
    raw = ext.extract_from_tsdf(tsdf, MeshBuildConfig(), [0, 1, 2])
    assert raw.vertices.shape[1] == 3
