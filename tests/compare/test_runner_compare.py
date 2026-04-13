from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tests.compare.helpers import (
    write_cd_material_feature_npz,
    write_cd_opening_npz,
    write_compare_config,
    write_contour,
    write_corner_npz,
    write_hidden_material_npz,
    write_internal_boundary_npz,
    write_label_target_compare_config,
    write_npz,
    write_open_contour,
    write_swapped_material_npz,
    write_topology_npz,
)
from wafergeo.compare import run_compare_from_config
from wafergeo.compare.schema import load_compare_spec_yaml


def _read_metric_details(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    details = payload.get("details", [])
    if not isinstance(details, list):
        return []
    return [row for row in details if isinstance(row, dict)]


def test_compare_outputs_scores_and_difference(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim.npz")
    target = write_contour(tmp_path / "target.json")
    out = tmp_path / "compare"
    cfg = write_compare_config(
        tmp_path / "compare.yaml",
        sim_path=sim,
        target_path=target,
        out_dir=out,
    )

    summary = run_compare_from_config(cfg)

    assert summary["status"] == "OK"
    assert (out / "score.json").exists()
    assert (out / "objective.json").exists()
    assert (out / "metrics.csv").exists()
    assert (out / "metric_details.json").exists()
    assert (out / "difference.png").exists()
    assert (out / "difference_legend.json").exists()
    assert (out / "difference_summary.json").exists()
    assert (out / "simulation_label_summary.json").exists()
    assert (out / "target_label_summary.json").exists()
    assert (out / "cd_profile_summary.json").exists()
    assert not (out / "material_confusion.csv").exists()
    assert (out / "features" / "simulation_contours.json").exists()
    assert (out / "_run" / "used_config.yaml").exists()

    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    objective = json.loads((out / "objective.json").read_text(encoding="utf-8"))
    metric_names = {row["name"] for row in score["metrics"]}
    assert {"cd", "chamfer", "sdf", "iou"} <= metric_names
    assert "normalized_total_score" in score
    assert objective["schema_version"] == "objective/v1"
    assert objective["status"] == "PARTIAL"
    assert objective["direction"] == "minimize"
    assert objective["objective_name"] == "normalized_total_score"
    assert objective["objective"] == score["normalized_total_score"]
    assert set(objective["metrics"]) >= {"cd", "chamfer", "sdf", "iou"}
    assert objective["skipped_metrics"]
    assert all("normalized_loss" in row for row in score["metrics"])
    assert {row["metric"] for row in score["metric_details"]} == {"sdf", "iou"}
    details_payload = json.loads((out / "metric_details.json").read_text(encoding="utf-8"))
    assert details_payload["_summary"]["metrics_with_details"] == ["sdf", "iou"]
    assert {row["metric"] for row in details_payload["details"]} == {"sdf", "iou"}
    diff_summary = json.loads((out / "difference_summary.json").read_text(encoding="utf-8"))
    assert int(diff_summary["height"]) > 0
    assert int(diff_summary["width"]) > 0
    assert diff_summary["mode"] in {"boundary", "label"}
    sim_summary = json.loads((out / "simulation_label_summary.json").read_text(encoding="utf-8"))
    assert sim_summary["label_volume"]["material_ids"] == [0, 1]
    assert sim_summary["view"]["axes"] == ["x", "y"]


def test_compare_topology_self_comparison_is_zero(tmp_path: Path) -> None:
    sim = write_topology_npz(tmp_path / "topology_self.npz")
    out = tmp_path / "topology_self"
    cfg = tmp_path / "topology_self.yaml"
    cfg.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {sim}
  target:
    kind: npz_label
    path: {sim}
view:
  axes: [x, z]
  depth_axis: y
features:
  use: [sdf]
metrics:
  use: [topology]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    run_compare_from_config(cfg)

    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    metrics = {row["name"]: row for row in score["metrics"]}
    assert metrics["topology"]["status"] == "OK"
    assert float(metrics["topology"]["loss"]) == pytest.approx(0.0)


def test_compare_topology_detects_material_split(tmp_path: Path) -> None:
    sim = write_topology_npz(tmp_path / "topology_split.npz", split_material=True)
    target = write_topology_npz(tmp_path / "topology_connected.npz")
    out = tmp_path / "topology_split"
    cfg = tmp_path / "topology_split.yaml"
    cfg.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {sim}
  target:
    kind: npz_label
    path: {target}
view:
  axes: [x, z]
  depth_axis: y
features:
  use: [sdf]
metrics:
  use: [topology]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    run_compare_from_config(cfg)

    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    metrics = {row["name"]: row for row in score["metrics"]}
    details = score["metric_details"][0]
    assert metrics["topology"]["status"] == "OK"
    assert float(metrics["topology"]["loss"]) > 0.0
    assert details["metric"] == "topology"
    assert details["mode"] == "projected_2d_component_count"


def test_compare_open_contour_uses_unsigned_distance_and_skips_iou(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim_open.npz")
    target = write_open_contour(tmp_path / "target_open.json")
    out = tmp_path / "open_contour"
    cfg = tmp_path / "open_contour.yaml"
    cfg.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {sim}
  target:
    kind: contour_json
    path: {target}
    units: nm
view:
  axes: [x, y]
  depth_axis: z
features:
  use: [sdf, contour]
metrics:
  use: [chamfer, sdf, sdf_band, iou]
output:
  dir: {out}
  difference_image: true
""",
        encoding="utf-8",
    )

    run_compare_from_config(cfg)

    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    metrics = {row["name"]: row for row in score["metrics"]}
    assert metrics["chamfer"]["status"] == "OK"
    assert metrics["sdf"]["status"] == "OK"
    assert metrics["sdf_band"]["status"] == "OK"
    assert metrics["iou"]["status"] == "SKIPPED"
    assert "iou" in score["skipped_metrics"]
    details = _read_metric_details(out / "metric_details.json")
    details_by_metric = {row["metric"]: row for row in details}
    assert details_by_metric["sdf"]["distance_semantics"] == "unsigned"
    assert details_by_metric["sdf_band"]["distance_semantics"] == "unsigned"
    target_sdf = np.load(out / "features" / "target_sdf.npz")
    assert float(np.min(target_sdf["sdf_nm"])) >= 0.0


def test_compare_accepts_label_target_and_self_score_is_best(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim.npz")
    shifted = write_npz(tmp_path / "shifted.npz", shift_x=1)
    self_cfg = write_label_target_compare_config(
        tmp_path / "self.yaml",
        sim_path=sim,
        target_path=sim,
        out_dir=tmp_path / "self",
    )
    shifted_cfg = write_label_target_compare_config(
        tmp_path / "shifted.yaml",
        sim_path=shifted,
        target_path=sim,
        out_dir=tmp_path / "shifted",
    )

    run_compare_from_config(self_cfg)
    run_compare_from_config(shifted_cfg)

    self_score = json.loads((tmp_path / "self" / "score.json").read_text(encoding="utf-8"))
    shifted_score = json.loads((tmp_path / "shifted" / "score.json").read_text(encoding="utf-8"))
    assert float(self_score["total_score"]) < float(shifted_score["total_score"])
    metrics = {row["name"]: row for row in self_score["metrics"]}
    shifted_metrics = {row["name"]: row for row in shifted_score["metrics"]}
    assert float(metrics["iou"]["value"]) == pytest.approx(1.0)
    assert float(metrics["sdf"]["loss"]) == pytest.approx(0.0)
    assert float(shifted_metrics["iou"]["value"]) < 1.0
    assert float(shifted_metrics["sdf"]["loss"]) > 0.0
    assert (tmp_path / "self" / "material_confusion.csv").exists()
    assert (tmp_path / "self" / "material_confusion_summary.json").exists()
    shifted_details = _read_metric_details(tmp_path / "shifted" / "metric_details.json")
    details_by_metric = {row["metric"]: row for row in shifted_details}
    assert details_by_metric["sdf"]["mask_sdf_loss_nm"] > 0.0
    assert details_by_metric["iou"]["label_iou"] < 1.0


def test_compare_rejects_mismatched_label_target_grid(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim_grid.npz")
    target = write_npz(tmp_path / "target_grid.npz", spacing=(2.0, 1.0, 1.0))
    cfg = write_label_target_compare_config(
        tmp_path / "grid_mismatch.yaml",
        sim_path=sim,
        target_path=target,
        out_dir=tmp_path / "grid_mismatch",
    )

    with pytest.raises(ValueError, match="projected view spacing differ"):
        run_compare_from_config(cfg)


def test_compare_label_target_scores_internal_material_boundary(tmp_path: Path) -> None:
    sim = write_internal_boundary_npz(tmp_path / "sim.npz", split_x=5)
    target = write_internal_boundary_npz(tmp_path / "target.npz", split_x=4)
    cfg = write_label_target_compare_config(
        tmp_path / "internal_boundary.yaml",
        sim_path=sim,
        target_path=target,
        out_dir=tmp_path / "internal_boundary",
        axes="[x, z]",
        depth_axis="y",
    )

    run_compare_from_config(cfg)

    score = json.loads(
        (tmp_path / "internal_boundary" / "score.json").read_text(encoding="utf-8")
    )
    metrics = {row["name"]: row for row in score["metrics"]}
    assert metrics["cd"]["status"] == "OK"
    assert float(metrics["cd"]["loss"]) > 0.0
    cd_summary = json.loads(
        (tmp_path / "internal_boundary" / "cd_profile_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert cd_summary["status"] == "OK"
    assert cd_summary["profile_mode"] == "auto_material_boundary"
    assert cd_summary["transition_loss_mean_nm"] > 0.0
    assert float(metrics["chamfer"]["loss"]) > 0.0
    assert float(metrics["sdf"]["loss"]) > 0.0
    assert float(metrics["iou"]["value"]) < 1.0


def test_compare_sdf_band_focuses_on_boundary_neighborhood(tmp_path: Path) -> None:
    sim = write_internal_boundary_npz(tmp_path / "sim_band.npz", split_x=5)
    target = write_internal_boundary_npz(tmp_path / "target_band.npz", split_x=4)
    out = tmp_path / "sdf_band"
    cfg = tmp_path / "sdf_band.yaml"
    cfg.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {sim}
  target:
    kind: npz_label
    path: {target}
view:
  axes: [x, z]
  depth_axis: y
features:
  use: [sdf]
metrics:
  use: [sdf_band]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    run_compare_from_config(cfg)

    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    metrics = {row["name"]: row for row in score["metrics"]}
    details = _read_metric_details(out / "metric_details.json")
    assert metrics["sdf_band"]["status"] == "OK"
    assert float(metrics["sdf_band"]["loss"]) > 0.0
    assert details[0]["metric"] == "sdf_band"
    assert details[0]["mode"] == "boundary_sdf_band"
    assert details[0]["band_pixel_count"] > 0


def test_compare_label_target_scores_material_id_mismatch(tmp_path: Path) -> None:
    sim = write_swapped_material_npz(tmp_path / "sim.npz")
    target = write_swapped_material_npz(tmp_path / "target.npz", swap=True)
    cfg = write_label_target_compare_config(
        tmp_path / "material_id_mismatch.yaml",
        sim_path=sim,
        target_path=target,
        out_dir=tmp_path / "material_id_mismatch",
    )

    run_compare_from_config(cfg)

    score = json.loads(
        (tmp_path / "material_id_mismatch" / "score.json").read_text(encoding="utf-8")
    )
    metrics = {row["name"]: row for row in score["metrics"]}
    assert float(metrics["chamfer"]["loss"]) == pytest.approx(0.0)
    assert float(metrics["sdf"]["loss"]) > 0.0
    assert float(metrics["iou"]["value"]) < 1.0
    details = _read_metric_details(tmp_path / "material_id_mismatch" / "metric_details.json")
    details_by_metric = {row["metric"]: row for row in details}
    assert details_by_metric["sdf"]["selected_loss_source"] in {"label_sdf", "boundary_sdf"}
    assert details_by_metric["iou"]["label_iou"] < 1.0
    confusion_rows = list(
        csv.DictReader(
            (tmp_path / "material_id_mismatch" / "material_confusion.csv").open(
                "r",
                encoding="utf-8",
            )
        )
    )
    off_diagonal = [
        row
        for row in confusion_rows
        if row["simulation_material_id"] != row["target_material_id"]
    ]
    assert off_diagonal
    confusion_summary = json.loads(
        (tmp_path / "material_id_mismatch" / "material_confusion_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert confusion_summary["mismatching_pixels"] > 0
    assert confusion_summary["major_confusion_pair"] is not None


def test_compare_sdf_material_reports_auto_detected_material_losses(tmp_path: Path) -> None:
    sim = write_swapped_material_npz(tmp_path / "sim_material_sdf.npz")
    target = write_swapped_material_npz(tmp_path / "target_material_sdf.npz", swap=True)
    out = tmp_path / "sdf_material"
    cfg = tmp_path / "sdf_material.yaml"
    cfg.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {sim}
  target:
    kind: npz_label
    path: {target}
features:
  use: [sdf]
metrics:
  use: [sdf_material]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    run_compare_from_config(cfg)

    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    metrics = {row["name"]: row for row in score["metrics"]}
    details = _read_metric_details(out / "metric_details.json")
    assert metrics["sdf_material"]["status"] == "OK"
    assert float(metrics["sdf_material"]["loss"]) > 0.0
    assert details[0]["metric"] == "sdf_material"
    assert details[0]["selected_material_ids"] == [1, 2]
    assert {row["material_id"] for row in details[0]["per_material"]} == {1, 2}
    assert all(row["sdf_loss_nm"] > 0.0 for row in details[0]["per_material"])
    per_material_rows = list(
        csv.DictReader((out / "per_material_sdf.csv").open("r", encoding="utf-8"))
    )
    assert {int(row["material_id"]) for row in per_material_rows} == {1, 2}


def test_compare_sdf_material_uses_projected_material_masks(tmp_path: Path) -> None:
    sim = write_hidden_material_npz(tmp_path / "sim_hidden_material.npz", shift_x=0)
    target = write_hidden_material_npz(tmp_path / "target_hidden_material.npz", shift_x=1)
    out = tmp_path / "sdf_material_hidden"
    cfg = tmp_path / "sdf_material_hidden.yaml"
    cfg.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {sim}
  target:
    kind: npz_label
    path: {target}
view:
  axes: [x, z]
  depth_axis: y
features:
  use: [sdf]
metrics:
  use: [sdf_material]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    run_compare_from_config(cfg)

    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    metrics = {row["name"]: row for row in score["metrics"]}
    details = _read_metric_details(out / "metric_details.json")
    assert metrics["sdf_material"]["status"] == "OK"
    assert float(metrics["sdf_material"]["loss"]) > 0.0
    assert details[0]["mask_source"] == "projected_material_masks"
    material_2 = [
        row for row in details[0]["per_material"] if int(row["material_id"]) == 2
    ][0]
    assert material_2["sdf_loss_nm"] > 0.0


def test_compare_cd_measures_cross_section_width_profile(tmp_path: Path) -> None:
    sim = write_cd_opening_npz(tmp_path / "sim_cd.npz", half_width=2)
    target = write_cd_opening_npz(tmp_path / "target_cd.npz", half_width=1)
    out = tmp_path / "cd_width"
    cfg = write_label_target_compare_config(
        tmp_path / "cd_width.yaml",
        sim_path=sim,
        target_path=target,
        out_dir=out,
        axes="[x, z]",
        depth_axis="y",
    )

    run_compare_from_config(cfg)

    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    metrics = {row["name"]: row for row in score["metrics"]}
    assert metrics["cd"]["status"] == "OK"
    assert float(metrics["cd"]["loss"]) == pytest.approx(1.0)
    assert (out / "cd_profile.csv").exists()
    assert (out / "cd_profile.png").exists()
    assert (out / "cd_profile_summary.json").exists()
    profile = list(csv.DictReader((out / "cd_profile.csv").open("r", encoding="utf-8")))
    assert profile
    assert {float(row["abs_diff_nm"]) for row in profile} == {2.0}
    assert {float(row["edge_loss_nm"]) for row in profile} == {1.0}
    cd_summary = json.loads((out / "cd_profile_summary.json").read_text(encoding="utf-8"))
    assert cd_summary["status"] == "OK"
    assert cd_summary["shared_height_count"] == len(profile)
    assert cd_summary["edge_loss_mean_nm"] == pytest.approx(1.0)


def test_compare_cd_measures_internal_material_width_profile(tmp_path: Path) -> None:
    sim = write_cd_material_feature_npz(tmp_path / "sim_material_cd.npz", half_width=2)
    target = write_cd_material_feature_npz(tmp_path / "target_material_cd.npz", half_width=1)
    out = tmp_path / "material_cd_width"
    cfg = write_label_target_compare_config(
        tmp_path / "material_cd_width.yaml",
        sim_path=sim,
        target_path=target,
        out_dir=out,
        axes="[x, z]",
        depth_axis="y",
    )

    run_compare_from_config(cfg)

    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    metrics = {row["name"]: row for row in score["metrics"]}
    assert metrics["cd"]["status"] == "OK"
    assert float(metrics["cd"]["loss"]) == pytest.approx(1.0)
    assert float(metrics["sdf"]["loss"]) > 0.0
    assert float(metrics["iou"]["value"]) < 1.0


def test_compare_profile_metric_reports_width_and_center_profile(tmp_path: Path) -> None:
    sim = write_cd_material_feature_npz(
        tmp_path / "sim_profile.npz",
        half_width=2,
        center_offset=1,
    )
    target = write_cd_material_feature_npz(tmp_path / "target_profile.npz", half_width=1)
    out = tmp_path / "profile"
    cfg = tmp_path / "profile.yaml"
    cfg.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {sim}
  target:
    kind: npz_label
    path: {target}
view:
  axes: [x, z]
  depth_axis: y
features:
  use: [contour]
metrics:
  use: [profile]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    run_compare_from_config(cfg)

    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    metrics = {row["name"]: row for row in score["metrics"]}
    assert metrics["profile"]["status"] == "OK"
    assert float(metrics["profile"]["loss"]) == pytest.approx(2.0)
    assert (out / "profile.csv").exists()
    assert (out / "profile_summary.json").exists()
    profile = list(csv.DictReader((out / "profile.csv").open("r", encoding="utf-8")))
    summary = json.loads((out / "profile_summary.json").read_text(encoding="utf-8"))
    assert profile
    assert {float(row["width_abs_diff_nm"]) for row in profile} == {2.0}
    assert {float(row["center_abs_diff_nm"]) for row in profile} == {1.0}
    assert summary["status"] == "OK"
    assert summary["selected_loss_source"] == "width_abs_diff_mean_nm"


def test_compare_profile_metric_self_comparison_is_zero(tmp_path: Path) -> None:
    sim = write_cd_material_feature_npz(tmp_path / "sim_profile_self.npz", half_width=1)
    out = tmp_path / "profile_self"
    cfg = tmp_path / "profile_self.yaml"
    cfg.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {sim}
  target:
    kind: npz_label
    path: {sim}
view:
  axes: [x, z]
  depth_axis: y
features:
  use: [contour]
metrics:
  use: [profile]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    run_compare_from_config(cfg)

    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    metrics = {row["name"]: row for row in score["metrics"]}
    assert metrics["profile"]["status"] == "OK"
    assert float(metrics["profile"]["loss"]) == pytest.approx(0.0)
    summary = json.loads((out / "profile_summary.json").read_text(encoding="utf-8"))
    assert summary["profile_loss_mean_nm"] == pytest.approx(0.0)


def test_compare_corner_metric_reports_bottom_corner_shift(tmp_path: Path) -> None:
    sim = write_corner_npz(tmp_path / "sim_corner.npz", bottom_shift_x=1)
    target = write_corner_npz(tmp_path / "target_corner.npz", bottom_shift_x=0)
    out = tmp_path / "corner"
    cfg = tmp_path / "corner.yaml"
    cfg.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {sim}
  target:
    kind: npz_label
    path: {target}
view:
  axes: [x, z]
  depth_axis: y
features:
  use: [contour]
metrics:
  use: [corner]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    run_compare_from_config(cfg)

    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    metrics = {row["name"]: row for row in score["metrics"]}
    assert metrics["corner"]["status"] == "OK"
    assert float(metrics["corner"]["loss"]) == pytest.approx(1.0)
    summary = json.loads((out / "corner_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "OK"
    assert summary["left_error_nm"] == pytest.approx(1.0)
    assert summary["right_error_nm"] == pytest.approx(1.0)


def test_compare_corner_metric_self_comparison_is_zero(tmp_path: Path) -> None:
    sim = write_corner_npz(tmp_path / "sim_corner_self.npz")
    out = tmp_path / "corner_self"
    cfg = tmp_path / "corner_self.yaml"
    cfg.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {sim}
  target:
    kind: npz_label
    path: {sim}
view:
  axes: [x, z]
  depth_axis: y
features:
  use: [contour]
metrics:
  use: [corner]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    run_compare_from_config(cfg)

    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    metrics = {row["name"]: row for row in score["metrics"]}
    assert metrics["corner"]["status"] == "OK"
    assert float(metrics["corner"]["loss"]) == pytest.approx(0.0)


def test_compare_cd_can_focus_on_material_ids(tmp_path: Path) -> None:
    sim = write_cd_material_feature_npz(tmp_path / "sim_material_focus.npz", half_width=2)
    target = write_cd_material_feature_npz(tmp_path / "target_material_focus.npz", half_width=1)
    out = tmp_path / "material_cd_focus"
    cfg = tmp_path / "material_cd_focus.yaml"
    cfg.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {sim}
  target:
    kind: npz_label
    path: {target}
view:
  kind: topview
  axes: [x, z]
  depth_axis: y
features:
  use: [contour]
metrics:
  use: [cd]
  cd:
    material_ids: [2]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    run_compare_from_config(cfg)

    spec = load_compare_spec_yaml(cfg)
    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    metrics = {row["name"]: row for row in score["metrics"]}
    assert spec.metrics.cd_material_ids == (2,)
    assert metrics["cd"]["status"] == "OK"
    assert float(metrics["cd"]["loss"]) == pytest.approx(1.0)
    profile = list(csv.DictReader((out / "cd_profile.csv").open("r", encoding="utf-8")))
    assert profile
    assert {float(row["edge_loss_nm"]) for row in profile} == {1.0}


def test_compare_cd_gauge_limits_height_range_and_center(tmp_path: Path) -> None:
    sim = write_cd_material_feature_npz(tmp_path / "sim_material_gauge.npz", half_width=2)
    target = write_cd_material_feature_npz(tmp_path / "target_material_gauge.npz", half_width=1)
    out = tmp_path / "material_cd_gauge"
    cfg = tmp_path / "material_cd_gauge.yaml"
    cfg.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {sim}
  target:
    kind: npz_label
    path: {target}
view:
  kind: topview
  axes: [x, z]
  depth_axis: y
features:
  use: [contour]
metrics:
  use: [cd]
  cd:
    material_ids: [2]
    gauge:
      axis: x
      height_axis: z
      center: 4.0
      height_range: [2.0, 6.0]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    run_compare_from_config(cfg)

    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    metrics = {row["name"]: row for row in score["metrics"]}
    profile = list(csv.DictReader((out / "cd_profile.csv").open("r", encoding="utf-8")))
    cd_summary = json.loads((out / "cd_profile_summary.json").read_text(encoding="utf-8"))
    assert metrics["cd"]["status"] == "OK"
    assert len(profile) == 3
    assert {float(row["z_nm"]) for row in profile} == {2.0, 4.0, 6.0}
    assert cd_summary["gauge"] == {
        "axis": "x",
        "center_nm": 4.0,
        "height_axis": "z",
        "height_range_nm": [2.0, 6.0],
        "source": "yaml",
    }


def test_compare_cd_penalizes_edge_placement_shift(tmp_path: Path) -> None:
    sim = write_cd_material_feature_npz(
        tmp_path / "sim_material_cd_shift.npz", half_width=1, center_offset=1
    )
    target = write_cd_material_feature_npz(
        tmp_path / "target_material_cd_shift.npz", half_width=1
    )
    out = tmp_path / "material_cd_shift"
    cfg = write_label_target_compare_config(
        tmp_path / "material_cd_shift.yaml",
        sim_path=sim,
        target_path=target,
        out_dir=out,
        axes="[x, z]",
        depth_axis="y",
    )

    run_compare_from_config(cfg)

    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    metrics = {row["name"]: row for row in score["metrics"]}
    assert metrics["cd"]["status"] == "OK"
    assert float(metrics["cd"]["loss"]) == pytest.approx(0.5)
    profile = list(csv.DictReader((out / "cd_profile.csv").open("r", encoding="utf-8")))
    assert profile
    assert {float(row["abs_diff_nm"]) for row in profile} == {1.0}
    assert {float(row["edge_loss_nm"]) for row in profile} == {0.5}
    cd_summary = json.loads((out / "cd_profile_summary.json").read_text(encoding="utf-8"))
    assert cd_summary["transition_loss_mean_nm"] == pytest.approx(0.5)


def test_compare_score_worsens_when_shape_is_shifted(tmp_path: Path) -> None:
    sim_a = write_npz(tmp_path / "sim_a.npz", shift_x=0)
    sim_b = write_npz(tmp_path / "sim_b.npz", shift_x=1)
    target = write_contour(tmp_path / "target.json")
    cfg_a = write_compare_config(
        tmp_path / "compare_a.yaml",
        sim_path=sim_a,
        target_path=target,
        out_dir=tmp_path / "compare_a",
    )
    cfg_b = write_compare_config(
        tmp_path / "compare_b.yaml",
        sim_path=sim_b,
        target_path=target,
        out_dir=tmp_path / "compare_b",
    )

    run_compare_from_config(cfg_a)
    run_compare_from_config(cfg_b)

    score_a = json.loads((tmp_path / "compare_a" / "score.json").read_text(encoding="utf-8"))
    score_b = json.loads((tmp_path / "compare_b" / "score.json").read_text(encoding="utf-8"))
    assert float(score_b["total_score"]) > float(score_a["total_score"])
