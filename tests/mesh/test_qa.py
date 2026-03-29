from __future__ import annotations

from tests.mesh.helpers import build_small_tsdf_volume
from wafergeo.mesh.build import build_mesh_from_tsdf
from wafergeo.mesh.config import MeshBuildConfig


def test_mesh_qa_fields() -> None:
    tsdf = build_small_tsdf_volume()
    mesh, _, qa = build_mesh_from_tsdf(
        tsdf,
        MeshBuildConfig(
            backend="naive_interface",
            mode="interface_mesh",
            sample_points_n=32,
            sample_seed=11,
        ),
    )

    assert qa.num_vertices == mesh.vertices.shape[0]
    assert qa.num_faces == mesh.faces.shape[0]
    assert 0.0 <= qa.degenerate_face_rate <= 1.0
    assert qa.interface_area_total > 0.0
    assert 0.0 <= qa.exposed_face_ratio <= 1.0
    assert isinstance(qa.interface_area_by_pair, dict)
