from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from tests.mesh.helpers import build_small_tsdf_volume
from wafergeo.core.grid import GridSpec
from wafergeo.core.meta import Meta
from wafergeo.core.types import LabelVolume, MaterialSpec
from wafergeo.mesh.build import build_mesh_from_tsdf
from wafergeo.mesh.config import MeshBuildConfig
from wafergeo.sdf.build import build_tsdf_volume
from wafergeo.sdf.config import SDFBuildConfig


def _build_vtk_ready_tsdf():
    label_arr = np.full((16, 16, 16), 0, dtype=np.uint8)
    label_arr[4:12, 4:12, 4:12] = 1
    material = MaterialSpec(
        ids=[0, 1],
        names=["void", "solid"],
        void_id=0,
        priority=[0, 1],
        ignore_in_exposure=[True, False],
    )
    label = LabelVolume(
        grid=GridSpec(
            dim=3,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.5, 0.5, 0.5),
            axis_order="ZYX",
            sample_location="cell_center",
            units="nm",
        ),
        material=material,
        material_id=label_arr,
        meta=Meta(
            schema_version="label/v1",
            profile_id="mesh_test",
            config_hash="cfg",
            generator_version="0.1.0",
            git_commit="deadbeef",
            input_hash="input",
            created_at=datetime.now(UTC).isoformat(),
            extra={},
        ),
    )
    tsdf, _ = build_tsdf_volume(
        label,
        SDFBuildConfig(
            mu_nm=4.0,
            backend="scipy",
            include_void_channel=True,
            boundary_features=True,
            compute_present_mask=True,
        ),
    )
    return tsdf


def test_build_mesh_from_small_tsdf() -> None:
    tsdf = build_small_tsdf_volume()
    mesh, pc, qa = build_mesh_from_tsdf(
        tsdf,
        MeshBuildConfig(
            backend="naive_interface",
            mode="interface_mesh",
            sample_points_n=64,
            sample_seed=42,
        ),
    )

    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0
    assert mesh.face_mat_in.shape == (mesh.faces.shape[0],)
    assert mesh.face_mat_out.shape == (mesh.faces.shape[0],)
    assert mesh.face_is_exposed.shape == (mesh.faces.shape[0],)

    valid_ids = set(tsdf.material.ids)
    assert set(np.unique(mesh.face_mat_in).tolist()).issubset(valid_ids)
    assert set(np.unique(mesh.face_mat_out).tolist()).issubset(valid_ids)
    assert bool(np.any(mesh.face_is_exposed))

    assert pc.points.shape == (64, 3)
    assert qa.num_faces == mesh.faces.shape[0]


def test_interface_vs_material_shell_modes() -> None:
    tsdf = build_small_tsdf_volume()

    mesh_interface, _, _ = build_mesh_from_tsdf(
        tsdf,
        MeshBuildConfig(
            backend="naive_interface",
            mode="interface_mesh",
            sample_points_n=32,
            sample_seed=1,
        ),
    )
    mesh_shell, _, _ = build_mesh_from_tsdf(
        tsdf,
        MeshBuildConfig(
            backend="naive_interface",
            mode="material_shell",
            sample_points_n=32,
            sample_seed=1,
        ),
    )

    assert mesh_interface.faces.shape[0] <= mesh_shell.faces.shape[0]


def test_mesh_attrib_reduces_same_side_faces_vtk() -> None:
    pytest.importorskip("vtk")
    tsdf = _build_vtk_ready_tsdf()
    mesh, _, _ = build_mesh_from_tsdf(
        tsdf,
        MeshBuildConfig(
            backend="vtk",
            mode="material_shell",
            sample_points_n=32,
            sample_seed=1,
        ),
    )
    same_side_ratio = float(np.mean(mesh.face_mat_in == mesh.face_mat_out))
    assert same_side_ratio < 0.05


def test_vtk_postprocess_disabled_keeps_geometry_path() -> None:
    pytest.importorskip("vtk")
    tsdf = _build_vtk_ready_tsdf()
    _, _, qa = build_mesh_from_tsdf(
        tsdf,
        MeshBuildConfig(
            backend="vtk",
            mode="material_shell",
            vtk_viz_postprocess_enabled=False,
            sample_points_n=16,
            sample_seed=0,
        ),
    )
    assert qa.postprocess_status == "OK"
    assert qa.post_bbox_shift_nm is not None
    assert qa.post_area_rel_error is not None


def test_vtk_postprocess_enabled_returns_metrics() -> None:
    pytest.importorskip("vtk")
    tsdf = _build_vtk_ready_tsdf()
    _, _, qa = build_mesh_from_tsdf(
        tsdf,
        MeshBuildConfig(
            backend="vtk",
            mode="material_shell",
            vtk_viz_postprocess_enabled=True,
            sample_points_n=16,
            sample_seed=0,
        ),
    )
    assert qa.postprocess_status in {"OK", "WARN"}
    assert qa.post_bbox_shift_nm is not None
    assert qa.post_area_rel_error is not None


def test_vtk_postprocess_warn_on_threshold_exceed() -> None:
    pytest.importorskip("vtk")
    tsdf = _build_vtk_ready_tsdf()
    _, _, qa = build_mesh_from_tsdf(
        tsdf,
        MeshBuildConfig(
            backend="vtk",
            mode="material_shell",
            vtk_viz_postprocess_enabled=True,
            qa_max_bbox_shift_nm=-1.0,
            qa_max_area_rel_error=0.0,
            qa_postprocess_on_exceed="warn",
            sample_points_n=16,
            sample_seed=0,
        ),
    )
    assert qa.postprocess_status == "WARN"
    assert any("vtk postprocess exceeds thresholds" in note for note in qa.notes)


def test_vtk_postprocess_fail_on_threshold_exceed() -> None:
    pytest.importorskip("vtk")
    tsdf = _build_vtk_ready_tsdf()
    with pytest.raises(ValueError, match="vtk postprocess exceeds thresholds"):
        build_mesh_from_tsdf(
            tsdf,
            MeshBuildConfig(
                backend="vtk",
                mode="material_shell",
                vtk_viz_postprocess_enabled=True,
                qa_max_bbox_shift_nm=-1.0,
                qa_max_area_rel_error=0.0,
                qa_postprocess_on_exceed="fail",
                sample_points_n=16,
                sample_seed=0,
            ),
        )
