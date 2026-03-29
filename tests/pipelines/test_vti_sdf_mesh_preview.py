from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from wafergeo.core.hashing import hash_config
from wafergeo.io.vti_reader import RawVtiImage
from wafergeo.pipelines import vti_sdf_mesh_preview as preview
from wafergeo.pipelines.vti_correspondence_audit import StandardVTIBundle, get_standard_vti_profile


def _synthetic_bundle() -> StandardVTIBundle:
    labels = np.array(
        [
            [[2, 2, 2], [2, 4, 4], [3, 3, 4]],
            [[2, 2, 2], [3, 4, 4], [3, 3, 4]],
        ],
        dtype=np.int64,
    )
    tsdf = np.zeros((2, 2, 3, 3), dtype=np.float32)
    raw = RawVtiImage(
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        dims_xyz=(3, 3, 2),
        arrays={"MaterialIds": labels.ravel(order="C")},
        array_location={"MaterialIds": "point"},
        vtk_meta={"reader": "synthetic"},
    )
    return StandardVTIBundle(
        profile=get_standard_vti_profile(),
        raw=raw,
        input_hash="h",
        read_backend_used="vtk",
        source_array="MaterialIds",
        source_location="point",
        converted_from_point=True,
        flat_layout_used="legacy_xyz_transpose",
        point_to_cell_match=0.97,
        selected_ids=[2, 4],
        normalized_label=labels,
        converted_label=labels.copy(),
        tsdf_stack=tsdf,
        spacing_zyx=(1.0, 1.0, 1.0),
        origin_zyx=(0.5, 0.5, 0.5),
        raw_shells={},
        converted_shells={},
    )


def _patch_audit_success(monkeypatch, out_dir: Path) -> None:
    def _fake_audit(*, vti_path, output_dir, outside_material_id, _bundle=None):
        figures = Path(output_dir) / "figures"
        tables = Path(output_dir) / "tables"
        figures.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)
        for name in (
            "3d_raw_shell_translucent.png",
            "3d_converted_shell_translucent.png",
            "3d_overlay_shell_translucent.png",
            "slice_x_mid_raw_vs_conv.png",
            "slice_y_mid_raw_vs_conv.png",
            "slice_z_mid_raw_vs_conv.png",
            "slice_x_mid_boundary_overlay.png",
            "slice_y_mid_boundary_overlay.png",
            "slice_z_mid_boundary_overlay.png",
        ):
            (figures / name).write_bytes(b"png")
        (tables / "material_volume_compare.csv").write_text("h\n", encoding="utf-8")
        (tables / "slice_metrics.csv").write_text("h\n", encoding="utf-8")
        return {
            "status": "OK",
            "input_path": str(vti_path),
            "point_to_cell_policy": "nearest",
            "material_policy": "full",
            "mesh_mode": "material_shell",
            "mesh_backend": "vtk",
            "mesh_backend_used": "vtk",
            "read_backend_used": "vtk",
            "flat_layout_used": "legacy_xyz_transpose",
            "mesh_faces_total": {"raw_shells": 10, "converted_shells": 12},
            "mesh_faces_plotted": {"raw_shells": 10, "converted_shells": 12},
            "postprocess": {
                "enabled": True,
                "params": {"vtk_subdivision_levels": 1},
                "metrics": {"bbox_shift_nm": 0.1, "area_rel_error": 0.01},
                "status": "OK",
            },
            "outputs": {
                "figures": sorted(str(p.name) for p in figures.glob("*.png")),
                "tables": sorted(str(p.name) for p in tables.glob("*.csv")),
            },
        }

    monkeypatch.setattr(preview, "run_vti_correspondence_audit", _fake_audit)


def test_preview_standard_profile_is_fixed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vti_path = tmp_path / "vox_t08.vti"
    vti_path.write_bytes(b"dummy")
    monkeypatch.setattr(
        preview,
        "compute_standard_vti_bundle",
        lambda *_a, **_k: _synthetic_bundle(),
    )
    _patch_audit_success(monkeypatch, tmp_path / "preview")

    manifest = preview.run_single_vti_preview(vti_path, tmp_path / "preview")
    assert manifest["point_to_cell_policy"] == "nearest"
    assert manifest["material_policy"] == "full"
    assert manifest["mesh_mode"] == "material_shell"
    assert manifest["mesh_backend_used"] == "vtk"
    assert manifest["read_backend_used"] == "vtk"
    assert manifest["flat_layout_used"] in {"vtk_x_fastest", "legacy_xyz_transpose"}


def test_full_materials_preserved_in_standard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vti_path = tmp_path / "vox_t08.vti"
    vti_path.write_bytes(b"dummy")
    monkeypatch.setattr(
        preview,
        "compute_standard_vti_bundle",
        lambda *_a, **_k: _synthetic_bundle(),
    )
    _patch_audit_success(monkeypatch, tmp_path / "preview")

    manifest = preview.run_single_vti_preview(vti_path, tmp_path / "preview")
    selected = cast(list[int], manifest["selected_material_ids"])
    assert set(selected) == {2, 4}


def test_preview_outputs_include_standard_sdf_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vti_path = tmp_path / "vox_t08.vti"
    vti_path.write_bytes(b"dummy")
    monkeypatch.setattr(
        preview,
        "compute_standard_vti_bundle",
        lambda *_a, **_k: _synthetic_bundle(),
    )
    _patch_audit_success(monkeypatch, tmp_path / "preview")

    out_dir = tmp_path / "preview"
    manifest = preview.run_single_vti_preview(vti_path, out_dir)
    figures = set(manifest["outputs"]["figures"])  # type: ignore[index]
    assert "sdf_minabs_xyz_mid.png" in figures
    assert "sdf_channels_zmid_full.png" in figures
    assert "sdf_channels_xmid_full.png" in figures
    assert "sdf_channels_ymid_full.png" in figures
    assert (out_dir / "sdf" / "tsdf_full_stack.npy").exists()
    assert (out_dir / "sdf" / "sdf_summary_full.json").exists()


def test_preview_reuses_audit_compute_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vti_path = tmp_path / "vox_t08.vti"
    vti_path.write_bytes(b"dummy")
    calls: dict[str, int] = {"bundle": 0}

    def _fake_bundle(*_a, **_k):
        calls["bundle"] += 1
        return _synthetic_bundle()

    monkeypatch.setattr(preview, "compute_standard_vti_bundle", _fake_bundle)
    _patch_audit_success(monkeypatch, tmp_path / "preview")

    preview.run_single_vti_preview(vti_path, tmp_path / "preview")
    assert calls["bundle"] == 1


def test_preview_audit_tsdf_shape_consistent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vti_path = tmp_path / "vox_t08.vti"
    vti_path.write_bytes(b"dummy")
    bundle = _synthetic_bundle()

    def _fake_bundle(*_a, **_k):
        return bundle

    def _fake_audit(*, vti_path, output_dir, outside_material_id, _bundle=None):
        assert _bundle is bundle
        figures = Path(output_dir) / "figures"
        tables = Path(output_dir) / "tables"
        figures.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)
        return {
            "status": "OK",
            "mesh_faces_total": {"raw_shells": 1, "converted_shells": 1},
            "mesh_faces_plotted": {"raw_shells": 1, "converted_shells": 1},
            "outputs": {"figures": [], "tables": []},
        }

    monkeypatch.setattr(preview, "compute_standard_vti_bundle", _fake_bundle)
    monkeypatch.setattr(preview, "run_vti_correspondence_audit", _fake_audit)
    out = tmp_path / "preview"
    manifest = preview.run_single_vti_preview(vti_path, out)
    assert manifest["sdf"]["shape"] == [2, 2, 3, 3]  # type: ignore[index]


def test_manifest_contains_schema_and_profile_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vti_path = tmp_path / "vox_t08.vti"
    vti_path.write_bytes(b"dummy")
    monkeypatch.setattr(
        preview,
        "compute_standard_vti_bundle",
        lambda *_a, **_k: _synthetic_bundle(),
    )
    _patch_audit_success(monkeypatch, tmp_path / "preview")
    manifest = preview.run_single_vti_preview(vti_path, tmp_path / "preview")
    assert manifest["schema_version"] == "vti_preview/v2"
    assert manifest["profile_id"] == "vti_standard_full_v1"
    assert manifest["profile_hash"] == hash_config(preview.get_standard_vti_profile())
    assert manifest["audit_manifest_path"] == "audit_manifest.json"


def test_preview_manifest_contains_postprocess_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vti_path = tmp_path / "vox_t08.vti"
    vti_path.write_bytes(b"dummy")
    monkeypatch.setattr(
        preview,
        "compute_standard_vti_bundle",
        lambda *_a, **_k: _synthetic_bundle(),
    )
    _patch_audit_success(monkeypatch, tmp_path / "preview")

    manifest = preview.run_single_vti_preview(vti_path, tmp_path / "preview")
    assert "postprocess" in manifest
    assert manifest["postprocess"]["enabled"] is True  # type: ignore[index]
    assert manifest["postprocess"]["status"] == "OK"  # type: ignore[index]


def test_cli_rejects_removed_options(tmp_path: Path) -> None:
    script = Path("scripts/run_vti_preview.py")
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
