from __future__ import annotations

import numpy as np

from tests.mesh.helpers import build_small_tsdf_volume
from wafergeo.mesh.attrib import annotate_faces_from_tsdf
from wafergeo.mesh.build import build_mesh_from_tsdf, resolve_channel_material_ids
from wafergeo.mesh.config import MeshBuildConfig
from wafergeo.mesh.extractors.naive_interface import NaiveInterfaceExtractor
from wafergeo.mesh.sampling import sample_pointcloud


def test_sampling_is_deterministic_with_seed() -> None:
    tsdf = build_small_tsdf_volume()
    cfg = MeshBuildConfig(backend="naive_interface", mode="interface_mesh", sample_points_n=32)
    material_ids = resolve_channel_material_ids(tsdf, cfg)

    raw = NaiveInterfaceExtractor().extract_from_tsdf(tsdf, cfg, material_ids)
    attrs = annotate_faces_from_tsdf(raw.vertices, raw.faces, tsdf, material_ids)

    pc1 = sample_pointcloud(
        vertices=raw.vertices,
        faces=raw.faces,
        face_normals=attrs.face_normals,
        face_pair_code=attrs.face_pair_code,
        face_is_exposed=attrs.face_is_exposed,
        n_points=32,
        seed=123,
        meta=tsdf.meta,
    )
    pc2 = sample_pointcloud(
        vertices=raw.vertices,
        faces=raw.faces,
        face_normals=attrs.face_normals,
        face_pair_code=attrs.face_pair_code,
        face_is_exposed=attrs.face_is_exposed,
        n_points=32,
        seed=123,
        meta=tsdf.meta,
    )
    pc3 = sample_pointcloud(
        vertices=raw.vertices,
        faces=raw.faces,
        face_normals=attrs.face_normals,
        face_pair_code=attrs.face_pair_code,
        face_is_exposed=attrs.face_is_exposed,
        n_points=32,
        seed=999,
        meta=tsdf.meta,
    )

    np.testing.assert_allclose(pc1.points, pc2.points)
    np.testing.assert_array_equal(pc1.pair_code, pc2.pair_code)
    np.testing.assert_array_equal(pc1.point_is_exposed, pc2.point_is_exposed)
    assert not np.allclose(pc1.points, pc3.points)


def test_pointcloud_has_pair_and_exposed() -> None:
    tsdf = build_small_tsdf_volume()
    _, pc, _ = build_mesh_from_tsdf(
        tsdf,
        MeshBuildConfig(
            backend="naive_interface",
            mode="interface_mesh",
            sample_points_n=64,
            sample_seed=7,
        ),
    )

    assert pc.pair_code.shape == (64,)
    assert pc.point_is_exposed.shape == (64,)
    assert pc.point_is_exposed.dtype == bool
