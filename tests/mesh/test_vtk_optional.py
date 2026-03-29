from __future__ import annotations

import importlib.util

import pytest

from tests.mesh.helpers import build_small_tsdf_volume
from wafergeo.mesh.config import MeshBuildConfig
from wafergeo.mesh.errors import MeshOptionalDependencyError
from wafergeo.mesh.extractors.registry import get_mesh_extractor


def test_vtk_backend_missing_dependency_error() -> None:
    if importlib.util.find_spec("vtk") is not None:
        pytest.skip("vtk is installed; missing-dependency path is not applicable")

    tsdf = build_small_tsdf_volume()
    ext = get_mesh_extractor("vtk")

    with pytest.raises(MeshOptionalDependencyError) as exc_info:
        ext.extract_from_tsdf(tsdf, MeshBuildConfig(backend="vtk"), [0, 1, 2])

    message = str(exc_info.value)
    assert "wafergeo[vtk]" in message
