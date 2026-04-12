from __future__ import annotations

import json
from pathlib import Path

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
