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
  use: [sdf_raw, tsdf_views, udf, material_sdf]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_from_config(cfg)

    assert summary["status"] == "OK"
    assert (out / "summary.json").exists()
    assert summary["features"] == {
        "sdf_raw": "sdf_raw.npz",
        "tsdf_views": "tsdf_views.npz",
        "udf": "udf.npz",
        "material_sdf": "material_sdf.npz",
        "material_interface_relation": "material_interface_relation.npz",
    }
    assert (out / "features" / "sdf_raw.npz").exists()
    assert (out / "features" / "tsdf_views.npz").exists()
    assert (out / "features" / "udf.npz").exists()
    assert (out / "features" / "material_sdf.npz").exists()
    assert (out / "features" / "material_interface_relation.npz").exists()
    assert not (out / "features" / "simulation_contours.json").exists()
    assert not (out / "features" / "simulation_sdf.npz").exists()
    assert (out / "feature_summary.json").exists()
    assert (out / "label_summary.json").exists()
    assert (out / "input_shape.png").exists()
    assert (out / "_run" / "run_info.json").exists()
    label_summary = json.loads((out / "label_summary.json").read_text(encoding="utf-8"))
    assert label_summary["label_volume"]["shape_zyx"] == [2, 8, 8]
    assert label_summary["view"]["non_void_pixels"] > 0

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

    assert summary["features"] == {
        "material_sdf": "material_sdf.npz",
        "material_interface_relation": "material_interface_relation.npz",
    }
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
    summary_by_name = {item["name"]: item for item in feature_summary["features"]}
    assert summary_by_name["material_sdf"]["semantics"] == "per_material_signed_distance"
    assert summary_by_name["material_interface_relation"]["semantics"] == (
        "material_interface_relation"
    )
    relation = np.load(out / "features" / "material_interface_relation.npz")
    assert set(relation.files) >= {
        "interface_distance_nm",
        "nearest_material_id",
        "second_material_id",
        "pair_code",
        "distance_gap_nm",
        "interface_band_10nm",
        "pair_codebook",
    }
    assert relation["interface_distance_nm"].shape == (2, 8, 8)


def test_transform_writes_material_tsdf_and_udf(tmp_path: Path) -> None:
    sim = write_internal_boundary_npz(tmp_path / "sim_material_multi.npz")
    out = tmp_path / "transform_material_multi"
    cfg = tmp_path / "transform_material_multi.yaml"
    cfg.write_text(
        f"""
task: transform
input:
  simulation:
    kind: npz_label
    path: {sim}
features:
  use: [material_tsdf_views, material_udf]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_from_config(cfg)

    assert summary["features"] == {
        "material_tsdf_views": "material_tsdf_views.npz",
        "material_udf": "material_udf.npz",
    }
    tsdf = np.load(out / "features" / "material_tsdf_views.npz")
    udf = np.load(out / "features" / "material_udf.npz")
    assert tsdf["sdf_nm"].shape == (2, 2, 8, 8)
    assert tsdf["tsdf_30nm"].shape == tsdf["sdf_nm"].shape
    assert udf["udf_nm"].shape == tsdf["sdf_nm"].shape
    assert np.min(udf["udf_nm"]) >= 0.0

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
        "process_transition_relation": "process_transition_relation.npz",
    }
    assert not (out / "features" / "process_delta_sdf_change_map.png").exists()
    assert not (out / "features" / "process_delta_sdf_legend.json").exists()
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
        "delta_names",
    }
    assert data["changed_sdf_nm"].shape == (2, 4, 4)
    assert int(data["changed_mask"].sum()) == 3
    assert int(data["etched_mask"].sum()) == 1
    assert int(data["deposited_mask"].sum()) == 1
    assert int(data["material_changed_mask"].sum()) == 1
    assert np.min(data["changed_sdf_nm"][data["changed_mask"].astype(bool)]) < 0.0
    feature_summary = json.loads((out / "feature_summary.json").read_text(encoding="utf-8"))
    assert feature_summary["features"][0]["name"] == "process_delta_sdf"
    assert feature_summary["features"][0]["semantics"] == "process_delta_signed_distance"
    transition = np.load(out / "features" / "process_transition_relation.npz")
    assert transition["transition_code"].shape == (2, 4, 4)
    assert transition["transition_codebook"].shape[1] == 3


def test_transform_writes_process_delta_tsdf_and_udf(tmp_path: Path) -> None:
    initial_labels = np.zeros((4, 4, 2), dtype=np.uint8)
    initial_labels[0:2, :, :] = 1
    final_labels = initial_labels.copy()
    final_labels[0, 0, 0] = 0
    final_labels[3, 3, 0] = 2
    reference = tmp_path / "initial_delta_multi.npz"
    simulation = tmp_path / "final_delta_multi.npz"
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
    out = tmp_path / "transform_process_delta_multi"
    cfg = tmp_path / "transform_process_delta_multi.yaml"
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
  use: [process_delta_tsdf_views, process_delta_udf]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_from_config(cfg)

    assert summary["features"] == {
        "process_delta_tsdf_views": "process_delta_tsdf_views.npz",
        "process_delta_udf": "process_delta_udf.npz",
    }
    tsdf = np.load(out / "features" / "process_delta_tsdf_views.npz")
    udf = np.load(out / "features" / "process_delta_udf.npz")
    assert tsdf["sdf_nm"].shape == (4, 2, 4, 4)
    assert tsdf["tsdf_10nm"].shape == tsdf["sdf_nm"].shape
    assert udf["udf_nm"].shape == tsdf["sdf_nm"].shape


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
  use: [sdf_raw]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_from_config(cfg)

    label_summary = json.loads((out / "label_summary.json").read_text(encoding="utf-8"))
    assert summary["view"]["axes"] == ("x", "z")
    assert label_summary["view"]["axes"] == ["x", "z"]
    assert (out / "input_shape.png").exists()


