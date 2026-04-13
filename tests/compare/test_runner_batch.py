from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tests.compare.helpers import write_contour, write_npz
from wafergeo.compare import run_batch_compare_from_config


def test_batch_compare_outputs_ranking(tmp_path: Path) -> None:
    sim_a = write_npz(tmp_path / "sim_a.npz", shift_x=0)
    sim_b = write_npz(tmp_path / "sim_b.npz", shift_x=1)
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
                "target_units",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "good",
                "simulation_kind": "npz_label",
                "simulation_path": sim_a.name,
                "target_kind": "npz_label",
                "target_path": target.name,
                "target_units": "",
            }
        )
        writer.writerow(
            {
                "case_id": "shifted",
                "simulation_kind": "npz_label",
                "simulation_path": sim_b.name,
                "target_kind": "npz_label",
                "target_path": target.name,
                "target_units": "",
            }
        )
    out = tmp_path / "batch"
    cfg = tmp_path / "batch.yaml"
    cfg.write_text(
        f"""
task: batch-compare
input:
  index: {index}
features:
  use: [sdf, contour]
metrics:
  use: [cd, chamfer, sdf, iou]
output:
  dir: {out}
  ranking: true
  difference_images: true
""",
        encoding="utf-8",
    )

    summary = run_batch_compare_from_config(cfg)

    assert summary["best_case_id"] == "good"
    assert summary["target_cache"]["entries"] == 1
    assert summary["target_cache"]["hits"] == 1
    assert summary["target_cache"]["shared_targets"]
    assert (out / "ranking.csv").exists()
    assert (out / "ranking_top.png").exists()
    assert (out / "objectives.csv").exists()
    assert (out / "metrics.csv").exists()
    assert (out / "metric_summary.csv").exists()
    assert (out / "material_confusion.csv").exists()
    assert (out / "difference_summary.csv").exists()
    assert (out / "differences" / "good.png").exists()
    assert (out / "shared_targets" / "target_0001" / "features" / "target_sdf.npz").exists()
    assert not (out / "cases" / "good" / "features" / "target_sdf.npz").exists()
    assert (out / "cases" / "good" / "difference_summary.json").exists()
    ranking = list(csv.DictReader((out / "ranking.csv").open("r", encoding="utf-8")))
    assert ranking[0]["case_id"] == "good"
    assert "normalized_total_score" in ranking[0]
    assert "total_score" in ranking[0]
    objectives = list(csv.DictReader((out / "objectives.csv").open("r", encoding="utf-8")))
    assert {row["case_id"] for row in objectives} == {"good", "shifted"}
    assert objectives[0]["objective_name"] == "normalized_total_score"
    assert objectives[0]["direction"] == "minimize"
    metrics = list(csv.DictReader((out / "metrics.csv").open("r", encoding="utf-8")))
    assert "normalized_loss" in metrics[0]
    assert "loss_scale" in metrics[0]
    metric_summary = list(csv.DictReader((out / "metric_summary.csv").open("r", encoding="utf-8")))
    assert {row["name"] for row in metric_summary} == {"cd", "chamfer", "sdf", "iou"}
    assert summary["metric_scales"] == {"cd": 10.0, "chamfer": 10.0, "sdf": 10.0, "iou": 1.0}
    difference_rows = list(
        csv.DictReader((out / "difference_summary.csv").open("r", encoding="utf-8"))
    )
    assert {row["case_id"] for row in difference_rows} == {"good", "shifted"}
    assert all(row["changed_pixels"] for row in difference_rows)
    confusion_rows = list(
        csv.DictReader((out / "material_confusion.csv").open("r", encoding="utf-8"))
    )
    assert {row["case_id"] for row in confusion_rows} == {"good", "shifted"}


def test_batch_compare_accepts_utf8_bom_index(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim.npz")
    target = write_npz(tmp_path / "target.npz")
    index = tmp_path / "pairs_bom.csv"
    index.write_text(
        "\ufeffcase_id,simulation_kind,simulation_path,target_kind,target_path\n"
        f"self,npz_label,{sim.name},npz_label,{target.name}\n",
        encoding="utf-8",
    )
    out = tmp_path / "batch_bom"
    cfg = tmp_path / "batch_bom.yaml"
    cfg.write_text(
        f"""
task: batch-compare
input:
  index: {index}
features:
  use: [sdf, contour]
metrics:
  use: [cd, chamfer, sdf, iou]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_batch_compare_from_config(cfg)

    assert summary["case_count"] == 1
    assert (out / "metrics.csv").exists()


def test_batch_compare_sanitizes_case_id_and_rejects_duplicates(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim.npz")
    target = write_contour(tmp_path / "target.json")
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
                "case_id": "../case 1",
                "simulation_kind": "npz_label",
                "simulation_path": sim.name,
                "target_kind": "contour_json",
                "target_path": target.name,
            }
        )
    out = tmp_path / "batch"
    cfg = tmp_path / "batch.yaml"
    cfg.write_text(
        f"""
task: batch-compare
input:
  index: {index}
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    run_batch_compare_from_config(cfg)

    assert (out / "cases" / "case_1").exists()
    assert not (tmp_path / "case 1").exists()

    with index.open("a", encoding="utf-8", newline="") as f:
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
        writer.writerow(
            {
                "case_id": "../case 1",
                "simulation_kind": "npz_label",
                "simulation_path": sim.name,
                "target_kind": "contour_json",
                "target_path": target.name,
            }
        )

    with pytest.raises(ValueError, match="duplicate case_id"):
        run_batch_compare_from_config(cfg)


def test_batch_compare_rejects_missing_required_columns(tmp_path: Path) -> None:
    index = tmp_path / "bad_pairs.csv"
    index.write_text("case_id,simulation_kind\ncase,npz_label\n", encoding="utf-8")
    cfg = tmp_path / "batch.yaml"
    cfg.write_text(
        f"""
task: batch-compare
input:
  index: {index}
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        run_batch_compare_from_config(cfg)
