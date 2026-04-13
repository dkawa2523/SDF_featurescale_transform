from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tests.compare.helpers import write_internal_boundary_npz, write_npz
from wafergeo.compare import run_transform_eval_from_config
from wafergeo.compare.schema import load_transform_eval_spec_yaml


def test_transform_eval_outputs_feature_tables(tmp_path: Path) -> None:
    sim_a = write_npz(tmp_path / "sim_a.npz")
    sim_b = write_internal_boundary_npz(tmp_path / "sim_b.npz")
    index = tmp_path / "cases.csv"
    index.write_text(
        f"""case_id,input_kind,input_path
case_a,npz_label,{sim_a.name}
case_b,npz_label,{sim_b.name}
""",
        encoding="utf-8",
    )
    out = tmp_path / "transform_eval"
    cfg = tmp_path / "transform_eval.yaml"
    cfg.write_text(
        f"""
task: transform-eval
input:
  index: {index.name}
eval:
  features:
    - target_shape: full_shape
      method: sdf
    - target_shape: full_shape
      method: multi_scale_tsdf
    - target_shape: material_shape
      method: sdf
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_eval_from_config(cfg)

    assert summary["task"] == "transform-eval"
    assert summary["case_count"] == 2
    assert summary["eval_feature_count"] == 3
    assert (out / "summary.json").exists()
    assert (out / "eval_feature_summary.csv").exists()
    assert (out / "eval_feature_signal.csv").exists()
    assert (out / "eval_feature_signal.json").exists()
    assert (out / "case_summary.csv").exists()
    assert (out / "feature_stats.csv").exists()
    assert (out / "case_variation_summary.csv").exists()
    assert (out / "material_coverage.csv").exists()
    assert (out / "_run" / "run_info.json").exists()
    assert (
        out
        / "eval_features"
        / "full_shape"
        / "sdf"
        / "cases"
        / "case_a"
        / "features"
        / "sdf_raw.npz"
    ).exists()
    assert (
        out
        / "eval_features"
        / "full_shape"
        / "multi_scale_tsdf"
        / "cases"
        / "case_a"
        / "features"
        / "tsdf_views.npz"
    ).exists()
    assert (
        out
        / "eval_features"
        / "material_shape"
        / "sdf"
        / "cases"
        / "case_b"
        / "features"
        / "material_sdf.npz"
    ).exists()

    feature_summary = (out / "eval_feature_summary.csv").read_text(encoding="utf-8")
    feature_signal = (out / "eval_feature_signal.csv").read_text(encoding="utf-8")
    feature_stats = (out / "feature_stats.csv").read_text(encoding="utf-8")
    case_variation = (out / "case_variation_summary.csv").read_text(encoding="utf-8")
    material_coverage = (out / "material_coverage.csv").read_text(encoding="utf-8")

    assert "execution_label,target_shape,method,code_name" in feature_summary
    assert "full_shape_sdf,full_shape,sdf,sdf_raw" in feature_summary
    assert (
        "full_shape_multi_scale_tsdf,full_shape,multi_scale_tsdf,tsdf_views"
        in feature_summary
    )
    assert "material_shape_sdf,material_shape,sdf,material_sdf" in feature_summary
    assert "execution_label,target_shape,method,code_name,case_count,signal_status" in (
        feature_signal
    )
    assert "full_shape_sdf,full_shape,sdf,sdf_raw,2,varies" in feature_signal
    assert "full_shape,multi_scale_tsdf,tsdf_views,case_a,tsdf_10nm" in feature_stats
    assert "array_hash" in feature_stats
    assert "material_shape,sdf,material_sdf,case_b,sdf_nm" in feature_stats
    assert "full_shape,sdf,sdf_raw,sdf_nm,2,2,true" in case_variation
    assert "material_shape,sdf,material_sdf,sdf_nm,2,2,true" in case_variation
    assert "material_shape,sdf,material_sdf,case_b,1" in material_coverage


def test_transform_eval_writes_diagnostic_figures(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    ref_a = write_npz(tmp_path / "ref_a.npz")
    ref_b = write_internal_boundary_npz(tmp_path / "ref_b.npz", split_x=5)
    sim_a = write_npz(tmp_path / "sim_a.npz", shift_x=1)
    sim_b = write_internal_boundary_npz(tmp_path / "sim_b.npz")
    index = tmp_path / "cases.csv"
    index.write_text(
        f"""case_id,input_kind,input_path,reference_kind,reference_path
case_a,npz_label,{sim_a.name},npz_label,{ref_a.name}
case_b,npz_label,{sim_b.name},npz_label,{ref_b.name}
""",
        encoding="utf-8",
    )
    out = tmp_path / "transform_eval_figures"
    cfg = tmp_path / "transform_eval_figures.yaml"
    cfg.write_text(
        f"""
task: transform-eval
input:
  index: {index.name}
process:
  enabled: true
eval:
  features:
    - target_shape: full_shape
      method: sdf
    - target_shape: full_shape
      method: udf
    - target_shape: material_shape
      method: sdf
    - target_shape: material_shape
      method: multi_scale_tsdf
    - target_shape: material_shape
      method: udf
    - target_shape: process_delta_shape
      method: sdf
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_eval_from_config(cfg)

    assert summary["figures"]["status"] == "OK"
    index = json.loads((out / "figures" / "index.json").read_text(encoding="utf-8"))
    assert index["status"] == "OK"
    assert (out / "figures" / "input_shape_sections.png").exists()
    assert (out / "figures" / "README.md").exists()
    assert "how_to_read" in index
    assert "input_shape_sections.png" in index["how_to_read"]
    assert "by_target_shape" in index["how_to_read"]
    by_target_shape = out / "figures" / "by_target_shape"
    field_figures = list(by_target_shape.glob("*/*/field.png"))
    score_figures = list(by_target_shape.glob("*/*/scores.png"))
    distance_figures = list(by_target_shape.glob("*/*/case_distance.png"))
    assert field_figures
    assert score_figures
    assert distance_figures
    assert (by_target_shape / "process_delta_shape" / "sdf" / "field.png").exists()
    assert (
        by_target_shape
        / "material_shape"
        / "relations"
        / "interface_relation"
        / "field.png"
    ).exists()
    assert (
        by_target_shape
        / "process_delta_shape"
        / "relations"
        / "transition_relation"
        / "field.png"
    ).exists()
    assert (by_target_shape / "material_shape" / "multi_scale_tsdf" / "field.png").exists()
    assert (by_target_shape / "material_shape" / "udf" / "field.png").exists()
    rows = list(
        csv.DictReader(
            (out / "figures" / "feature_scores.csv").open(
                "r",
                encoding="utf-8",
            )
        )
    )
    assert rows
    assert "code_name" in rows[0]
    assert "target_shape" in rows[0]
    assert "method" in rows[0]
    assert "relation" in rows[0]
    assert "role" in rows[0]
    assert "shape_match" in rows[0]
    assert "boundary_match" in rows[0]
    assert "interface_match" in rows[0]
    assert "case_sensitivity" in rows[0]
    assert "data_cost" in rows[0]
    udf_rows = [row for row in rows if row["code_name"] == "udf"]
    assert udf_rows
    assert udf_rows[0]["method"] == "udf"
    assert udf_rows[0]["target_shape"] == "full_shape"
    assert udf_rows[0]["boundary_match"]
    assert not udf_rows[0]["shape_match"]
    process_rows = [row for row in rows if row["code_name"] == "process_delta_sdf"]
    assert process_rows
    assert process_rows[0]["method"] == "sdf"
    assert process_rows[0]["target_shape"] == "process_delta_shape"
    relation_rows = [row for row in rows if row["code_name"] == "material_interface_relation"]
    assert relation_rows
    assert relation_rows[0]["method"] == ""
    assert relation_rows[0]["relation"] == "interface_relation"
    assert relation_rows[0]["role"] == "derived_relation"
    assert relation_rows[0]["interface_match"]
    distance_rows = list(
        csv.DictReader(
            (out / "figures" / "case_distance.csv").open("r", encoding="utf-8")
        )
    )
    assert distance_rows
    assert "method" in distance_rows[0]
    assert "target_shape" in distance_rows[0]
    assert "normalized_rmse" in distance_rows[0]


def test_transform_eval_removes_stale_feature_and_figure_outputs(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim.npz")
    index = tmp_path / "cases.csv"
    index.write_text(
        f"""case_id,input_kind,input_path
case_a,npz_label,{sim.name}
""",
        encoding="utf-8",
    )
    out = tmp_path / "transform_eval_clean"
    cfg = tmp_path / "transform_eval_clean.yaml"
    cfg.write_text(
        f"""
task: transform-eval
input:
  index: {index.name}
eval:
  features:
    - target_shape: full_shape
      method: sdf
output:
  dir: {out}
""",
        encoding="utf-8",
    )
    run_transform_eval_from_config(cfg)
    stale_feature = out / "eval_features" / "old" / "cases" / "old" / "old.png"
    stale_figure = out / "figures" / "old.png"
    stale_profile = out / "feature_profile_values.csv"
    stale_feature.parent.mkdir(parents=True)
    stale_figure.parent.mkdir(parents=True, exist_ok=True)
    stale_feature.write_text("old", encoding="utf-8")
    stale_figure.write_text("old", encoding="utf-8")
    stale_profile.write_text("old", encoding="utf-8")

    run_transform_eval_from_config(cfg)

    assert not stale_feature.exists()
    assert not stale_figure.exists()
    assert not stale_profile.exists()
    assert (
        out
        / "eval_features"
        / "full_shape"
        / "sdf"
        / "cases"
        / "case_a"
        / "input_shape.png"
    ).exists()


def test_transform_eval_rejects_duplicate_features(tmp_path: Path) -> None:
    sim = write_npz(tmp_path / "sim.npz")
    index = tmp_path / "cases.csv"
    index.write_text(
        f"""case_id,input_kind,input_path
self,npz_label,{sim.name}
""",
        encoding="utf-8",
    )
    cfg = tmp_path / "transform_eval_collision.yaml"
    cfg.write_text(
        f"""
task: transform-eval
input:
  index: {index.name}
eval:
  features:
    - target_shape: full_shape
      method: sdf
    - target_shape: full_shape
      method: sdf
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate transform-eval feature"):
        run_transform_eval_from_config(cfg)


def test_transform_eval_requires_features(tmp_path: Path) -> None:
    cfg = tmp_path / "transform_eval_bad.yaml"
    cfg.write_text(
        f"""
task: transform-eval
input:
  index: {tmp_path / "cases.csv"}
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required key: eval"):
        load_transform_eval_spec_yaml(cfg)


def test_transform_eval_process_delta_sdf_requires_process(tmp_path: Path) -> None:
    initial = np.zeros((4, 4, 2), dtype=np.uint8)
    final = initial.copy()
    final[0, 0, 0] = 1
    initial_path = tmp_path / "initial.npz"
    final_path = tmp_path / "final.npz"
    for path, labels in ((initial_path, initial), (final_path, final)):
        np.savez(
            path,
            labels=labels,
            spacing=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            material_ids=np.array([0, 1], dtype=np.int32),
        )
    index = tmp_path / "process_cases.csv"
    index.write_text(
        f"""case_id,input_kind,input_path,reference_kind,reference_path
case_a,npz_label,{final_path.name},npz_label,{initial_path.name}
""",
        encoding="utf-8",
    )
    cfg = tmp_path / "transform_eval_process.yaml"
    cfg.write_text(
        f"""
task: transform-eval
input:
  index: {index.name}
eval:
  features:
    - target_shape: process_delta_shape
      method: sdf
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="process features require process.enabled"):
        run_transform_eval_from_config(cfg)
