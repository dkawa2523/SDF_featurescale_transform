from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.compare.helpers import write_internal_boundary_npz, write_npz
from wafergeo.compare import run_transform_from_config


def test_transform_outputs_feature_files(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim.npz")
    out = tmp_path / "transform"
    cfg = tmp_path / "transform.yaml"
    cfg.write_text(
        f"""
task: transform
input:
  simulation:
    kind: npz_label
    path: {sim}
features:
  use: [sdf, mesh, contour, slice]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_from_config(cfg)

    assert summary["status"] == "OK"
    assert (out / "summary.json").exists()
    assert (out / "features" / "simulation_contours.json").exists()
    assert (out / "features" / "simulation_sdf.npz").exists()
    assert not (out / "features" / "sdf.npz").exists()
    assert (out / "features" / "mesh.npz").exists()
    assert (out / "features" / "mesh_summary.json").exists()
    assert (out / "feature_summary.json").exists()
    assert (out / "label_summary.json").exists()
    assert (out / "preview.png").exists()
    assert (out / "_run" / "run_info.json").exists()
    label_summary = json.loads((out / "label_summary.json").read_text(encoding="utf-8"))
    assert label_summary["label_volume"]["shape_zyx"] == [2, 8, 8]
    assert label_summary["view"]["non_void_pixels"] > 0


def test_transform_writes_sdf3d_only_when_requested(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim.npz")
    out = tmp_path / "transform_sdf3d"
    cfg = tmp_path / "transform_sdf3d.yaml"
    cfg.write_text(
        f"""
task: transform
input:
  simulation:
    kind: npz_label
    path: {sim}
features:
  use: [sdf3d]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_from_config(cfg)

    assert summary["features"] == {"sdf3d": "sdf.npz"}
    assert (out / "features" / "sdf.npz").exists()


def test_transform_writes_sdf_raw_only_when_requested(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim_sdf_raw.npz")
    out = tmp_path / "transform_sdf_raw"
    cfg = tmp_path / "transform_sdf_raw.yaml"
    cfg.write_text(
        f"""
task: transform
input:
  simulation:
    kind: npz_label
    path: {sim}
features:
  use: [sdf_raw]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_from_config(cfg)

    assert summary["features"] == {"sdf_raw": "sdf_raw.npz"}
    assert summary["feature_summary"] == "feature_summary.json"
    data = np.load(out / "features" / "sdf_raw.npz")
    assert set(data.files) == {
        "sdf_nm",
        "mask",
        "spacing_zyx_nm",
        "origin_zyx_nm",
        "material_ids",
        "void_id",
    }
    assert data["sdf_nm"].shape == (2, 8, 8)
    assert data["mask"].shape == data["sdf_nm"].shape
    assert np.min(data["sdf_nm"][data["mask"].astype(bool)]) < 0.0
    assert np.max(data["sdf_nm"][~data["mask"].astype(bool)]) > 0.0
    feature_summary = json.loads((out / "feature_summary.json").read_text(encoding="utf-8"))
    assert feature_summary["features"][0]["name"] == "sdf_raw"
    assert feature_summary["features"][0]["semantics"] == "signed_distance"
    assert feature_summary["features"][0]["source_region"] == "non_void_union"


def test_transform_writes_tsdf_views_only_when_requested(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim_tsdf_views.npz")
    out = tmp_path / "transform_tsdf_views"
    cfg = tmp_path / "transform_tsdf_views.yaml"
    cfg.write_text(
        f"""
task: transform
input:
  simulation:
    kind: npz_label
    path: {sim}
features:
  use: [tsdf_views]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_from_config(cfg)

    assert summary["features"] == {"tsdf_views": "tsdf_views.npz"}
    data = np.load(out / "features" / "tsdf_views.npz")
    assert set(data.files) == {
        "sdf_nm",
        "tsdf_10nm",
        "tsdf_30nm",
        "tsdf_100nm",
        "log_abs_sdf",
        "mask",
        "clip_nm",
        "spacing_zyx_nm",
        "origin_zyx_nm",
        "material_ids",
        "void_id",
    }
    assert data["sdf_nm"].shape == data["mask"].shape
    assert np.max(data["tsdf_10nm"]) <= 1.0
    assert np.min(data["tsdf_10nm"]) >= -1.0
    np.testing.assert_allclose(data["clip_nm"], np.asarray([10.0, 30.0, 100.0]))
    np.testing.assert_allclose(data["log_abs_sdf"], np.log1p(np.abs(data["sdf_nm"])))
    feature_summary = json.loads((out / "feature_summary.json").read_text(encoding="utf-8"))
    assert feature_summary["features"][0]["name"] == "tsdf_views"
    assert feature_summary["features"][0]["source_feature"] == "sdf_raw"


def test_transform_writes_udf_only_when_requested(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim_udf.npz")
    out = tmp_path / "transform_udf"
    cfg = tmp_path / "transform_udf.yaml"
    cfg.write_text(
        f"""
task: transform
input:
  simulation:
    kind: npz_label
    path: {sim}
features:
  use: [udf]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_from_config(cfg)

    assert summary["features"] == {"udf": "udf.npz"}
    data = np.load(out / "features" / "udf.npz")
    assert set(data.files) == {
        "udf_nm",
        "mask",
        "spacing_zyx_nm",
        "origin_zyx_nm",
        "material_ids",
        "void_id",
    }
    assert data["udf_nm"].shape == (2, 8, 8)
    assert np.min(data["udf_nm"]) >= 0.0
    feature_summary = json.loads((out / "feature_summary.json").read_text(encoding="utf-8"))
    assert feature_summary["features"][0]["name"] == "udf"
    assert feature_summary["features"][0]["semantics"] == "unsigned_distance"


def test_transform_writes_material_sdf_only_when_requested(tmp_path: Path) -> None:
    sim = write_internal_boundary_npz(tmp_path / "sim_material_sdf.npz")
    out = tmp_path / "transform_material_sdf"
    cfg = tmp_path / "transform_material_sdf.yaml"
    cfg.write_text(
        f"""
task: transform
input:
  simulation:
    kind: npz_label
    path: {sim}
features:
  use: [material_sdf]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_from_config(cfg)

    assert summary["features"] == {"material_sdf": "material_sdf.npz"}
    data = np.load(out / "features" / "material_sdf.npz")
    assert set(data.files) == {
        "sdf_nm",
        "material_ids",
        "voxel_counts",
        "spacing_zyx_nm",
        "origin_zyx_nm",
        "void_id",
    }
    assert data["sdf_nm"].shape == (2, 2, 8, 8)
    np.testing.assert_array_equal(data["material_ids"], np.asarray([1, 2], dtype=np.int32))
    assert np.all(data["voxel_counts"] > 0)
    assert np.min(data["sdf_nm"][0]) < 0.0
    assert np.max(data["sdf_nm"][0]) > 0.0
    feature_summary = json.loads((out / "feature_summary.json").read_text(encoding="utf-8"))
    assert feature_summary["features"][0]["name"] == "material_sdf"
    assert feature_summary["features"][0]["semantics"] == "per_material_signed_distance"


def test_transform_writes_material_profile_only_when_requested(tmp_path: Path) -> None:
    sim = write_internal_boundary_npz(tmp_path / "sim_material_profile.npz")
    out = tmp_path / "transform_material_profile"
    cfg = tmp_path / "transform_material_profile.yaml"
    cfg.write_text(
        f"""
task: transform
input:
  simulation:
    kind: npz_label
    path: {sim}
features:
  use: [material_profile]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_from_config(cfg)

    assert summary["features"] == {
        "material_profile": "material_profile.csv",
        "material_profile_z_profile": "material_profile_z_profile.csv",
        "material_profile_summary": "material_profile_summary.json",
    }
    profile = (out / "features" / "material_profile.csv").read_text(encoding="utf-8")
    z_profile = (out / "features" / "material_profile_z_profile.csv").read_text(
        encoding="utf-8"
    )
    material_summary = json.loads(
        (out / "features" / "material_profile_summary.json").read_text(encoding="utf-8")
    )
    feature_summary = json.loads((out / "feature_summary.json").read_text(encoding="utf-8"))

    assert "material_id,material_name,is_void,voxel_count,voxel_fraction" in profile
    assert "0,material_0,true,56,0.4375" in profile
    assert "1,material_1,false,36,0.28125" in profile
    assert "2,material_2,false,36,0.28125" in profile
    assert "z_index,z_nm,material_id,material_name,is_void,voxel_count,slice_fraction" in z_profile
    assert "0,0.0,1,material_1,false,18,0.28125" in z_profile
    assert material_summary["schema_version"] == "material_profile/v1"
    assert material_summary["material_ids"] == [0, 1, 2]
    assert material_summary["material_count"] == 3
    assert material_summary["total_voxels"] == 128
    assert material_summary["materials"][0]["is_void"] is True
    assert feature_summary["features"][0]["name"] == "material_profile"
    assert feature_summary["features"][0]["semantics"] == "per_material_profile"
    assert feature_summary["features"][0]["outputs"]["z_profile"] == (
        "material_profile_z_profile.csv"
    )


def test_transform_process_mode_records_reference_without_extra_features(tmp_path: Path) -> None:
    reference = write_npz(tmp_path / "initial.npz")
    simulation = write_npz(tmp_path / "final.npz")
    out = tmp_path / "transform_process"
    cfg = tmp_path / "transform_process.yaml"
    cfg.write_text(
        f"""
task: transform
input:
  reference:
    kind: npz_label
    path: {reference}
  simulation:
    kind: npz_label
    path: {simulation}
process:
  enabled: true
features:
  use: [sdf_raw]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_from_config(cfg)

    assert summary["process"] == {"enabled": True}
    assert summary["reference"] == {
        "kind": "npz_label",
        "path": str(reference),
        "void_id": None,
    }
    assert summary["features"] == {"sdf_raw": "sdf_raw.npz"}
    label_summary = json.loads((out / "label_summary.json").read_text(encoding="utf-8"))
    run_info = json.loads((out / "_run" / "run_info.json").read_text(encoding="utf-8"))
    assert label_summary["label_volume"]["shape_zyx"] == [2, 8, 8]
    assert label_summary["reference_label_volume"]["shape_zyx"] == [2, 8, 8]
    assert run_info["inputs"]["simulation"] == str(simulation)
    assert run_info["inputs"]["reference"] == str(reference)


def test_transform_writes_process_delta_profile_when_requested(tmp_path: Path) -> None:
    initial_labels = np.zeros((4, 4, 2), dtype=np.uint8)
    initial_labels[0:2, :, :] = 1
    final_labels = initial_labels.copy()
    final_labels[0, 0, 0] = 0
    final_labels[3, 3, 0] = 2
    final_labels[1, 1, 0] = 2
    reference = tmp_path / "initial_delta.npz"
    simulation = tmp_path / "final_delta.npz"
    np.savez(
        reference,
        labels=initial_labels,
        spacing=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        material_ids=np.array([0, 1], dtype=np.int32),
    )
    np.savez(
        simulation,
        labels=final_labels,
        spacing=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        material_ids=np.array([0, 1, 2], dtype=np.int32),
    )
    out = tmp_path / "transform_process_delta"
    cfg = tmp_path / "transform_process_delta.yaml"
    cfg.write_text(
        f"""
task: transform
input:
  reference:
    kind: npz_label
    path: {reference}
  simulation:
    kind: npz_label
    path: {simulation}
process:
  enabled: true
features:
  use: [process_delta_profile]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_from_config(cfg)

    assert summary["features"] == {
        "process_delta_profile": "process_delta_profile.csv",
        "process_delta_z_profile": "process_delta_z_profile.csv",
        "process_delta_summary": "process_delta_summary.json",
    }
    profile = (out / "features" / "process_delta_profile.csv").read_text(encoding="utf-8")
    process_summary = json.loads(
        (out / "features" / "process_delta_summary.json").read_text(encoding="utf-8")
    )
    feature_summary = json.loads((out / "feature_summary.json").read_text(encoding="utf-8"))

    assert "transition_key,change_type,initial_material_id" in profile
    assert "0_to_2,deposited,0" in profile
    assert "1_to_0,etched,1" in profile
    assert "1_to_2,material_changed,1" in profile
    assert process_summary["schema_version"] == "process_delta_profile/v1"
    assert process_summary["changed_voxels"] == 3
    assert process_summary["transition_count"] == 3
    assert feature_summary["features"][0]["name"] == "process_delta_profile"
    assert feature_summary["features"][0]["semantics"] == "process_delta_profile"
    assert feature_summary["features"][0]["changed_voxels"] == 3


def test_transform_writes_process_delta_sdf_when_requested(tmp_path: Path) -> None:
    initial_labels = np.zeros((4, 4, 2), dtype=np.uint8)
    initial_labels[0:2, :, :] = 1
    final_labels = initial_labels.copy()
    final_labels[0, 0, 0] = 0
    final_labels[3, 3, 0] = 2
    final_labels[1, 1, 0] = 2
    reference = tmp_path / "initial_delta_sdf.npz"
    simulation = tmp_path / "final_delta_sdf.npz"
    np.savez(
        reference,
        labels=initial_labels,
        spacing=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        material_ids=np.array([0, 1], dtype=np.int32),
    )
    np.savez(
        simulation,
        labels=final_labels,
        spacing=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        material_ids=np.array([0, 1, 2], dtype=np.int32),
    )
    out = tmp_path / "transform_process_delta_sdf"
    cfg = tmp_path / "transform_process_delta_sdf.yaml"
    cfg.write_text(
        f"""
task: transform
input:
  reference:
    kind: npz_label
    path: {reference}
  simulation:
    kind: npz_label
    path: {simulation}
process:
  enabled: true
features:
  use: [process_delta_sdf]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_from_config(cfg)

    assert summary["features"] == {
        "process_delta_sdf": "process_delta_sdf.npz",
        "process_delta_sdf_legend": "process_delta_sdf_legend.json",
        "process_delta_sdf_preview": "process_delta_sdf_preview.png",
        "process_delta_sdf_summary": "process_delta_sdf_summary.json",
    }
    assert (out / "features" / "process_delta_sdf_preview.png").exists()
    assert (out / "features" / "process_delta_sdf_legend.json").exists()
    data = np.load(out / "features" / "process_delta_sdf.npz")
    assert set(data.files) == {
        "changed_mask",
        "changed_sdf_nm",
        "deposited_mask",
        "deposited_sdf_nm",
        "etched_mask",
        "etched_sdf_nm",
        "final_void_id",
        "initial_void_id",
        "material_changed_mask",
        "material_changed_sdf_nm",
        "origin_zyx_nm",
        "spacing_zyx_nm",
    }
    assert data["changed_sdf_nm"].shape == (2, 4, 4)
    assert int(data["changed_mask"].sum()) == 3
    assert int(data["etched_mask"].sum()) == 1
    assert int(data["deposited_mask"].sum()) == 1
    assert int(data["material_changed_mask"].sum()) == 1
    assert np.min(data["changed_sdf_nm"][data["changed_mask"].astype(bool)]) < 0.0
    process_summary = json.loads(
        (out / "features" / "process_delta_sdf_summary.json").read_text(encoding="utf-8")
    )
    feature_summary = json.loads((out / "feature_summary.json").read_text(encoding="utf-8"))
    assert process_summary["schema_version"] == "process_delta_sdf/v1"
    assert process_summary["changed_voxels"] == 3
    assert process_summary["masks"]["etched"]["voxel_count"] == 1
    legend = json.loads(
        (out / "features" / "process_delta_sdf_legend.json").read_text(encoding="utf-8")
    )
    assert legend["preview"]["view"] == "xz"
    assert legend["category_meaning"]["etched"] == "initial non-void material became final void"
    assert feature_summary["features"][0]["name"] == "process_delta_sdf"
    assert feature_summary["features"][0]["semantics"] == "process_delta_signed_distance"
    assert feature_summary["features"][0]["outputs"]["preview"] == "process_delta_sdf_preview.png"


def test_transform_accepts_view_spec(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim_view.npz")
    out = tmp_path / "transform_view"
    cfg = tmp_path / "transform_view.yaml"
    cfg.write_text(
        f"""
task: transform
input:
  simulation:
    kind: npz_label
    path: {sim}
view:
  axes: [x, z]
  depth_axis: y
features:
  use: [slice]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_from_config(cfg)

    label_summary = json.loads((out / "label_summary.json").read_text(encoding="utf-8"))
    assert summary["view"]["axes"] == ("x", "z")
    assert label_summary["view"]["axes"] == ["x", "z"]
    assert (out / "preview.png").exists()
