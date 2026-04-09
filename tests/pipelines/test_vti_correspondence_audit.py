from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np

from wafergeo.core.hashing import hash_config
from wafergeo.io.vti_reader import RawVtiImage, VtiReadResult
from wafergeo.pipelines import vti_correspondence_audit as audit


def _raw_cell_labels() -> RawVtiImage:
    cell_zyx = np.array(
        [
            [[2, 2, 3, 3], [2, 4, 4, 3], [5, 5, 4, 2]],
            [[2, 6, 6, 3], [2, 4, 6, 3], [5, 5, 4, 2]],
        ],
        dtype=np.int32,
    )
    cell_xyz = cell_zyx.transpose(2, 1, 0)
    return RawVtiImage(
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        dims_xyz=(5, 4, 3),
        arrays={"MaterialIds": cell_xyz.ravel(order="C")},
        array_location={"MaterialIds": "cell"},
    )


def _patch_read_result(
    monkeypatch,
    *,
    raw: RawVtiImage | None = None,
    backend_used: str = "vtk",
    messages: tuple[str, ...] = (),
) -> None:
    result = VtiReadResult(
        raw=_raw_cell_labels() if raw is None else raw,
        backend_used=backend_used,
        messages=messages,
    )
    monkeypatch.setattr(
        audit.vti_reader,
        "read_vti_with_xml_fallback",
        lambda _p: result,
    )


def _patch_plots(monkeypatch) -> None:
    def _write(*args, **kwargs) -> None:
        output = kwargs.get("output_path")
        if output is None and args:
            output = args[-1]
        assert output is not None
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    monkeypatch.setattr(audit, "_plot_shells_translucent", _write)
    monkeypatch.setattr(audit, "_plot_overlay_shells", _write)
    monkeypatch.setattr(audit, "_plot_slice_compare", _write)
    monkeypatch.setattr(audit, "_plot_boundary_overlay", _write)


def _patch_converted_identity(monkeypatch) -> None:
    def _fake_build_converted_label_and_tsdf(**kwargs):
        label = kwargs["label_cell_zyx"]
        selected_ids = kwargs["selected_ids"]
        tsdf = np.zeros((len(selected_ids),) + label.shape, dtype=np.float32)
        return label.copy(), tsdf

    monkeypatch.setattr(
        audit,
        "_build_converted_label_and_tsdf",
        _fake_build_converted_label_and_tsdf,
    )


def test_audit_standard_profile_is_fixed(tmp_path: Path, monkeypatch) -> None:
    vti_path = tmp_path / "dummy.vti"
    vti_path.write_bytes(b"vti")
    monkeypatch.setattr(audit, "read_vti", lambda _p: _raw_cell_labels())
    monkeypatch.setattr(audit, "_boundary_chamfer", lambda _a, _b: 0.0)
    _patch_converted_identity(monkeypatch)
    _patch_plots(monkeypatch)

    manifest = audit.run_vti_correspondence_audit(vti_path, tmp_path / "out")

    assert manifest["point_to_cell_policy"] == "nearest"
    assert manifest["material_policy"] == "full"
    assert manifest["mesh_mode"] == "material_shell"
    assert manifest["mesh_backend"] == "vtk"
    assert manifest["mesh_backend_used"] == "vtk"
    assert manifest["read_backend_used"] == "vtk"
    assert manifest["flat_layout_used"] in {"vtk_x_fastest", "legacy_xyz_transpose"}
    assert manifest["schema_version"] == "vti_audit/v2"
    assert manifest["profile_id"] == "vti_standard_full_v1"
    assert manifest["profile_hash"] == hash_config(audit.get_standard_vti_profile())
    assert "postprocess" in manifest
    assert manifest["postprocess"]["enabled"] is True  # type: ignore[index]


def test_standard_profile_is_public_and_shared() -> None:
    profile = audit.get_standard_vti_profile()
    assert profile["point_to_cell_policy"] == "nearest"
    assert profile["material_policy"] == "full"
    assert profile["mesh_mode"] == "material_shell"
    assert profile["mesh_backend"] == "vtk"


def test_standard_profile_has_no_dead_keys() -> None:
    profile = audit.get_standard_vti_profile()
    assert "flat_layout" not in profile


def test_no_decimation_in_standard_outputs(tmp_path: Path, monkeypatch) -> None:
    vti_path = tmp_path / "dummy.vti"
    vti_path.write_bytes(b"vti")
    monkeypatch.setattr(audit, "read_vti", lambda _p: _raw_cell_labels())
    monkeypatch.setattr(audit, "_boundary_chamfer", lambda _a, _b: 0.0)
    _patch_converted_identity(monkeypatch)
    _patch_plots(monkeypatch)
    monkeypatch.setattr(
        audit,
        "apply_vtk_visual_postprocess",
        lambda v, f, cfg: (
            v,
            f,
            {
                "bbox_shift_nm": 0.0,
                "area_rel_error": 0.0,
                "pre_faces": float(f.shape[0]),
                "post_faces": float(f.shape[0]),
            },
        ),
    )

    faces = np.arange(0, 300, dtype=np.int32).reshape(100, 3)
    verts = np.zeros((300, 3), dtype=np.float32)
    monkeypatch.setattr(audit, "_extract_label_shells", lambda **_k: {2: (verts, faces)})

    manifest = audit.run_vti_correspondence_audit(vti_path, tmp_path / "out")
    assert manifest["mesh_faces_total"]["raw_shells"] == 100  # type: ignore[index]
    assert manifest["mesh_faces_plotted"]["raw_shells"] == 100  # type: ignore[index]
    assert manifest["mesh_faces_total"]["converted_shells"] == 100  # type: ignore[index]
    assert manifest["mesh_faces_plotted"]["converted_shells"] == 100  # type: ignore[index]


def test_full_materials_preserved_in_standard(tmp_path: Path, monkeypatch) -> None:
    vti_path = tmp_path / "dummy.vti"
    vti_path.write_bytes(b"vti")
    monkeypatch.setattr(audit, "read_vti", lambda _p: _raw_cell_labels())
    monkeypatch.setattr(audit, "_boundary_chamfer", lambda _a, _b: 0.0)
    _patch_converted_identity(monkeypatch)
    _patch_plots(monkeypatch)

    manifest = audit.run_vti_correspondence_audit(vti_path, tmp_path / "out")
    material_ids_raw = cast(list[int], manifest["material_ids_raw"])
    material_ids_converted = cast(list[int], manifest["material_ids_converted"])
    assert set(material_ids_raw) == {2, 3, 4, 5, 6}
    assert set(material_ids_converted) == {2, 3, 4, 5, 6}


def test_cli_rejects_removed_options(tmp_path: Path) -> None:
    script = Path("scripts/run_vti_correspondence_audit.py")
    dummy = tmp_path / "dummy.vti"
    dummy.write_bytes(b"vti")
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--vti",
            str(dummy),
            "--out",
            str(out),
            "--mesh-mode",
            "material_shell",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "unrecognized arguments: --mesh-mode material_shell" in proc.stderr


def test_audit_outputs_pre_post_mesh_figures(tmp_path: Path, monkeypatch) -> None:
    vti_path = tmp_path / "dummy.vti"
    vti_path.write_bytes(b"vti")
    monkeypatch.setattr(audit, "read_vti", lambda _p: _raw_cell_labels())
    monkeypatch.setattr(audit, "_boundary_chamfer", lambda _a, _b: 0.0)
    _patch_converted_identity(monkeypatch)
    _patch_plots(monkeypatch)
    monkeypatch.setattr(
        audit,
        "apply_vtk_visual_postprocess",
        lambda v, f, cfg: (
            v,
            f,
            {
                "bbox_shift_nm": 0.1,
                "area_rel_error": 0.01,
                "pre_faces": float(f.shape[0]),
                "post_faces": float(f.shape[0]),
            },
        ),
    )

    out = tmp_path / "out"
    manifest = audit.run_vti_correspondence_audit(vti_path, out)
    fig_names = set(manifest["outputs"]["figures"])  # type: ignore[index]
    assert "3d_converted_shell_translucent_pre.png" in fig_names
    assert "3d_converted_shell_translucent_post.png" in fig_names
    assert "3d_converted_shell_overlay_pre_post.png" in fig_names


def test_audit_manifest_contains_postprocess_metrics(tmp_path: Path, monkeypatch) -> None:
    vti_path = tmp_path / "dummy.vti"
    vti_path.write_bytes(b"vti")
    monkeypatch.setattr(audit, "read_vti", lambda _p: _raw_cell_labels())
    monkeypatch.setattr(audit, "_boundary_chamfer", lambda _a, _b: 0.0)
    _patch_converted_identity(monkeypatch)
    _patch_plots(monkeypatch)
    monkeypatch.setattr(
        audit,
        "apply_vtk_visual_postprocess",
        lambda v, f, cfg: (
            v,
            f,
            {
                "bbox_shift_nm": 0.2,
                "area_rel_error": 0.02,
                "pre_faces": float(f.shape[0]),
                "post_faces": float(f.shape[0]),
            },
        ),
    )

    manifest = audit.run_vti_correspondence_audit(vti_path, tmp_path / "out")
    post = cast(dict[str, object], manifest["postprocess"])
    assert cast(str, post["status"]) in {"OK", "WARN"}
    metrics = cast(dict[str, float], post["metrics"])
    assert metrics["bbox_shift_nm"] >= 0.0
    assert metrics["area_rel_error"] >= 0.0


def test_audit_manifest_records_read_fallback_message(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vti_path = tmp_path / "dummy.vti"
    vti_path.write_bytes(b"vti")
    _patch_read_result(
        monkeypatch,
        backend_used="xml_fallback",
        messages=("vtk read fallback to xml: ImportError: no vtk runtime",),
    )
    monkeypatch.setattr(audit, "_boundary_chamfer", lambda _a, _b: 0.0)
    _patch_converted_identity(monkeypatch)
    _patch_plots(monkeypatch)

    manifest = audit.run_vti_correspondence_audit(vti_path, tmp_path / "out")

    assert manifest["read_backend_used"] == "xml_fallback"
    assert any("fallback to xml" in message for message in manifest["messages"])
