from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.compare.helpers import write_npz
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
