from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.compare.helpers import write_npz
from wafergeo.compare import run_compare_eval_from_config


def test_compare_eval_runs_candidates_and_writes_summary_tables(tmp_path: Path) -> None:
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
  candidates:
    primary:
      features:
        use: [sdf, contour]
      metrics:
        use: [cd, sdf, iou]
    material_diagnostic:
      features:
        use: [sdf, contour]
      metrics:
        use: [cd, sdf, iou, sdf_material]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_compare_eval_from_config(cfg)

    assert summary["task"] == "compare-eval"
    assert summary["case_count"] == 2
    assert summary["candidate_count"] == 2
    assert summary["target_cache"]["entries"] == 1
    assert summary["target_cache"]["hits"] == 3
    assert (out / "candidate_summary.csv").exists()
    assert (out / "case_scores.csv").exists()
    assert (out / "metric_summary.csv").exists()
    assert (out / "ranking_consistency.csv").exists()
    assert (out / "summary.json").exists()
    assert not (out / "candidates").exists()

    candidate_rows = list(
        csv.DictReader((out / "candidate_summary.csv").open("r", encoding="utf-8"))
    )
    assert {row["candidate"] for row in candidate_rows} == {"primary", "material_diagnostic"}
    assert all("mean_normalized_total_score" in row for row in candidate_rows)
    assert all(row["status"] == "OK" for row in candidate_rows)
    assert all(row["objective_name"] == "normalized_total_score" for row in candidate_rows)
    assert all(row["direction"] == "minimize" for row in candidate_rows)
    assert all("mean_objective" in row for row in candidate_rows)
    assert all("best_case_id" in row for row in candidate_rows)
    assert all("best_objective" in row for row in candidate_rows)
    assert all("max_abs_rank_delta" in row for row in candidate_rows)
    assert candidate_rows[0]["candidate"] == "primary"
    assert candidate_rows[0]["best_case_id"] == "good"
    assert candidate_rows[0]["changed_rank_count"] == "0"
    assert candidate_rows[0]["ok_case_count"] == "2"
    assert candidate_rows[0]["partial_case_count"] == "0"
    assert all(row["skipped_metric_count"] == "0" for row in candidate_rows)

    case_rows = list(csv.DictReader((out / "case_scores.csv").open("r", encoding="utf-8")))
    assert "output_dir" not in case_rows[0]
    assert case_rows[0]["status"] == "OK"
    assert case_rows[0]["objective_name"] == "normalized_total_score"
    assert case_rows[0]["direction"] == "minimize"
    assert case_rows[0]["skipped_metrics"] == ""
    primary_rows = [row for row in case_rows if row["candidate"] == "primary"]
    primary_by_case = {row["case_id"]: row for row in primary_rows}
    assert float(primary_by_case["good"]["normalized_total_score"]) <= float(
        primary_by_case["shifted"]["normalized_total_score"]
    )
    assert "sdf_material_loss" in case_rows[0]

    metric_rows = list(csv.DictReader((out / "metric_summary.csv").open("r", encoding="utf-8")))
    assert {row["metric"] for row in metric_rows} >= {"cd", "sdf", "iou"}

    ranking_rows = list(
        csv.DictReader((out / "ranking_consistency.csv").open("r", encoding="utf-8"))
    )
    assert {row["candidate"] for row in ranking_rows} == {"primary", "material_diagnostic"}

    summary_payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["candidates"] == ["primary", "material_diagnostic"]
    assert summary_payload["baseline_candidate"] == "primary"


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
  candidates:
    primary:
      features:
        use: [sdf, contour]
      metrics:
        use: [cd, sdf, iou]
    material_diagnostic:
      features:
        use: [sdf, contour]
      metrics:
        use: [cd, sdf, iou, sdf_material]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_compare_eval_from_config(cfg)

    assert summary["figures"]["status"] == "OK"
    assert (out / "figures" / "objective_heatmap.png").exists()
    assert (out / "figures" / "rank_delta_heatmap.png").exists()
    assert (out / "figures" / "metric_contribution_heatmap.png").exists()
    assert (out / "figures" / "metric_evaluation_score.png").exists()
    assert (out / "figures" / "README.md").exists()
    manifest = json.loads((out / "figures" / "figure_manifest.json").read_text(encoding="utf-8"))
    assert "how_to_read" in manifest
    assert list((out / "figures" / "representative_differences").glob("*.png"))
    rows = list(
        csv.DictReader(
            (out / "figures" / "metric_evaluation_scores.csv").open(
                "r",
                encoding="utf-8",
            )
        )
    )
    assert rows
    for row in rows:
        assert 0.0 <= float(row["case_coverage"]) <= 1.0
        assert 0.0 <= float(row["metric_coverage"]) <= 1.0
        assert 0.0 <= float(row["metric_evaluation_score"]) <= 1.0


def test_compare_eval_removes_stale_candidate_and_figure_outputs(tmp_path: Path) -> None:
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
  candidates:
    primary:
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
    stale_candidate = out / "candidates" / "old" / "cases" / "old" / "preview.png"
    stale_figure = out / "figures" / "old.png"
    stale_candidate.parent.mkdir(parents=True)
    stale_figure.parent.mkdir(parents=True, exist_ok=True)
    stale_candidate.write_text("old", encoding="utf-8")
    stale_figure.write_text("old", encoding="utf-8")

    run_compare_eval_from_config(cfg)

    assert not stale_candidate.exists()
    assert not stale_figure.exists()
    assert not (out / "candidates").exists()


def test_compare_eval_rejects_candidate_output_collisions(tmp_path: Path) -> None:
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
  candidates:
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

    with pytest.raises(ValueError, match="candidate names collide"):
        run_compare_eval_from_config(cfg)
