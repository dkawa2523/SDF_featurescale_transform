from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.compare.helpers import write_npz
from wafergeo.compare import run_compare_eval_from_config


def test_compare_eval_runs_metric_sets_and_writes_summary_tables(tmp_path: Path) -> None:
    sim_good = write_npz(tmp_path / "sim_good.npz", shift_x=0)
    sim_shifted = write_npz(tmp_path / "sim_shifted.npz", shift_x=1)
    target = write_npz(tmp_path / "target.npz", shift_x=0)
    index = tmp_path / "pairs.csv"
    with index.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "simulation_kind",
                "simulation_path",
                "target_kind",
                "target_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "good",
                "simulation_kind": "npz_label",
                "simulation_path": sim_good.name,
                "target_kind": "npz_label",
                "target_path": target.name,
            }
        )
        writer.writerow(
            {
                "case_id": "shifted",
                "simulation_kind": "npz_label",
                "simulation_path": sim_shifted.name,
                "target_kind": "npz_label",
                "target_path": target.name,
            }
        )
    out = tmp_path / "compare_eval"
    cfg = tmp_path / "compare_eval.yaml"
    cfg.write_text(
        f"""
task: compare-eval
input:
  index: {index}
view:
  axes: [x, z]
  depth_axis: y
eval:
  metric_sets:
    height_cd:
      features:
        use: [contour]
      metrics:
        use: [cd]
    shape_distance:
      features:
        use: [sdf, contour]
      metrics:
        use: [sdf, iou]
    material_distance:
      features:
        use: [sdf, contour]
      metrics:
        use: [sdf, iou, sdf_material]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_compare_eval_from_config(cfg)

    assert summary["task"] == "compare-eval"
    assert summary["case_count"] == 2
    assert summary["metric_set_count"] == 3
    assert summary["target_cache"]["entries"] == 1
    assert summary["target_cache"]["hits"] == 5
    assert (out / "metric_set_summary.csv").exists()
    assert (out / "case_scores.csv").exists()
    assert (out / "metric_summary.csv").exists()
    assert (out / "ranking_consistency.csv").exists()
    assert (out / "axis_agreement.csv").exists()
    assert (out / "summary.json").exists()
    assert not (out / "metric_sets").exists()

    metric_set_rows = list(
        csv.DictReader((out / "metric_set_summary.csv").open("r", encoding="utf-8"))
    )
    assert {row["metric_set"] for row in metric_set_rows} == {
        "height_cd",
        "shape_distance",
        "material_distance",
    }
    assert all("mean_comparison_loss" in row for row in metric_set_rows)
    assert all(row["status"] == "OK" for row in metric_set_rows)
    assert all(row["direction"] == "minimize" for row in metric_set_rows)
    assert all("case_separation" in row for row in metric_set_rows)
    assert all("best_case_id" in row for row in metric_set_rows)
    assert all("best_comparison_loss" in row for row in metric_set_rows)
    assert all("ranking_shift_mean" in row for row in metric_set_rows)
    assert metric_set_rows[0]["metric_set"] == "height_cd"
    assert metric_set_rows[0]["best_case_id"] == "good"
    assert metric_set_rows[0]["changed_rank_count"] == "0"
    assert metric_set_rows[0]["ok_case_count"] == "2"
    assert metric_set_rows[0]["partial_case_count"] == "0"
    assert all(row["skipped_metric_count"] == "0" for row in metric_set_rows)

    case_rows = list(csv.DictReader((out / "case_scores.csv").open("r", encoding="utf-8")))
    assert "output_dir" not in case_rows[0]
    assert case_rows[0]["status"] == "OK"
    assert case_rows[0]["direction"] == "minimize"
    assert case_rows[0]["skipped_metrics"] == ""
    assert "comparison_loss" in case_rows[0]
    assert "raw_loss" in case_rows[0]
    cd_rows = [row for row in case_rows if row["metric_set"] == "height_cd"]
    cd_by_case = {row["case_id"]: row for row in cd_rows}
    assert float(cd_by_case["good"]["comparison_loss"]) <= float(
        cd_by_case["shifted"]["comparison_loss"]
    )
    assert "sdf_material_loss" in case_rows[0]

    metric_rows = list(csv.DictReader((out / "metric_summary.csv").open("r", encoding="utf-8")))
    assert {row["metric"] for row in metric_rows} >= {"cd", "sdf", "iou"}
    assert {row["metric_family"] for row in metric_rows} >= {
        "shape_distance",
        "shape_overlap",
    }

    ranking_rows = list(
        csv.DictReader((out / "ranking_consistency.csv").open("r", encoding="utf-8"))
    )
    assert {row["metric_set"] for row in ranking_rows} == {
        "height_cd",
        "shape_distance",
        "material_distance",
    }
    assert "ranking_shift" in ranking_rows[0]

    agreement_rows = list(csv.DictReader((out / "axis_agreement.csv").open("r", encoding="utf-8")))
    assert agreement_rows
    assert {"axis_a", "axis_b", "rank_agreement"} <= set(agreement_rows[0])

    summary_payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["metric_sets"] == [
        "height_cd",
        "shape_distance",
        "material_distance",
    ]
    assert summary_payload["baseline_metric_set"] == "height_cd"


def test_compare_eval_writes_diagnostic_figures(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    sim_good = write_npz(tmp_path / "sim_good.npz", shift_x=0)
    sim_shifted = write_npz(tmp_path / "sim_shifted.npz", shift_x=1)
    target = write_npz(tmp_path / "target.npz", shift_x=0)
    index = tmp_path / "pairs.csv"
    with index.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "simulation_kind",
                "simulation_path",
                "target_kind",
                "target_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "good",
                "simulation_kind": "npz_label",
                "simulation_path": sim_good.name,
                "target_kind": "npz_label",
                "target_path": target.name,
            }
        )
        writer.writerow(
            {
                "case_id": "shifted",
                "simulation_kind": "npz_label",
                "simulation_path": sim_shifted.name,
                "target_kind": "npz_label",
                "target_path": target.name,
            }
        )
    out = tmp_path / "compare_eval_figures"
    cfg = tmp_path / "compare_eval_figures.yaml"
    cfg.write_text(
        f"""
task: compare-eval
input:
  index: {index}
view:
  axes: [x, z]
  depth_axis: y
eval:
  metric_sets:
    height_cd:
      features:
        use: [contour]
      metrics:
        use: [cd]
    shape_distance:
      features:
        use: [sdf, contour]
      metrics:
        use: [sdf, iou]
    material_distance:
      features:
        use: [sdf, contour]
      metrics:
        use: [sdf, iou, sdf_material]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_compare_eval_from_config(cfg)

    assert summary["figures"]["status"] == "OK"
    assert (out / "figures" / "comparison_loss_heatmap.png").exists()
    assert (out / "figures" / "ranking_shift_heatmap.png").exists()
    assert (out / "figures" / "metric_loss_breakdown.png").exists()
    assert (out / "figures" / "cd_vs_sdf_scatter.png").exists()
    assert (out / "figures" / "evaluation_axis_summary.png").exists()
    assert not (out / "figures" / "metric_breakdown_heatmap.png").exists()
    assert not (out / "figures" / "metric_set_health.png").exists()
    assert not (out / "figures" / "objective_heatmap.png").exists()
    assert not (out / "figures" / "metric_evaluation_score.png").exists()
    assert (out / "figures" / "README.md").exists()
    index = json.loads((out / "figures" / "index.json").read_text(encoding="utf-8"))
    assert "how_to_read" in index
    assert list((out / "figures" / "representative_differences").glob("*.png"))
    rows = list(
        csv.DictReader(
            (out / "figures" / "evaluation_axis_summary.csv").open(
                "r",
                encoding="utf-8",
            )
        )
    )
    assert rows
    for row in rows:
        assert 0.0 <= float(row["case_coverage"]) <= 1.0
        assert 0.0 <= float(row["metric_coverage"]) <= 1.0
        assert 0.0 <= float(row["case_separation"]) <= 1.0
        assert float(row["ranking_shift_mean"]) >= 0.0


def test_compare_eval_removes_stale_metric_set_and_figure_outputs(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim.npz")
    target = write_npz(tmp_path / "target.npz")
    index = tmp_path / "pairs.csv"
    with index.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "simulation_kind",
                "simulation_path",
                "target_kind",
                "target_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case_a",
                "simulation_kind": "npz_label",
                "simulation_path": sim.name,
                "target_kind": "npz_label",
                "target_path": target.name,
            }
        )
    out = tmp_path / "compare_eval_clean"
    cfg = tmp_path / "compare_eval_clean.yaml"
    cfg.write_text(
        f"""
task: compare-eval
input:
  index: {index}
eval:
  metric_sets:
    shape_distance:
      features:
        use: [sdf, contour]
      metrics:
        use: [sdf, iou]
output:
  dir: {out}
""",
        encoding="utf-8",
    )
    run_compare_eval_from_config(cfg)
    stale_metric_set = out / "metric_sets" / "old" / "cases" / "old" / "old.png"
    stale_figure = out / "figures" / "old.png"
    stale_metric_set.parent.mkdir(parents=True)
    stale_figure.parent.mkdir(parents=True, exist_ok=True)
    stale_metric_set.write_text("old", encoding="utf-8")
    stale_figure.write_text("old", encoding="utf-8")

    run_compare_eval_from_config(cfg)

    assert not stale_metric_set.exists()
    assert not stale_figure.exists()
    assert not (out / "metric_sets").exists()


def test_compare_eval_rejects_metric_set_output_collisions(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim.npz")
    target = write_npz(tmp_path / "target.npz")
    index = tmp_path / "pairs.csv"
    with index.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "simulation_kind",
                "simulation_path",
                "target_kind",
                "target_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "self",
                "simulation_kind": "npz_label",
                "simulation_path": sim.name,
                "target_kind": "npz_label",
                "target_path": target.name,
            }
        )
    cfg = tmp_path / "compare_eval_collision.yaml"
    cfg.write_text(
        f"""
task: compare-eval
input:
  index: {index}
eval:
  metric_sets:
    a_b:
      metrics:
        use: [sdf]
    a b:
      metrics:
        use: [sdf]
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metric_set names collide"):
        run_compare_eval_from_config(cfg)
