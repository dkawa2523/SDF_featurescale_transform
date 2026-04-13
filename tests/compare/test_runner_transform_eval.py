from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tests.compare.helpers import write_internal_boundary_npz, write_npz
from wafergeo.compare import run_transform_eval_from_config
from wafergeo.compare.schema import load_transform_eval_spec_yaml


def test_transform_eval_outputs_candidate_tables(tmp_path: Path) -> None:
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
  candidates:
    raw:
      features:
        use: [sdf_raw]
    tsdf:
      features:
        use: [tsdf_views]
    material:
      features:
        use: [material_sdf]
    profile:
      features:
        use: [material_profile]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_eval_from_config(cfg)

    assert summary["task"] == "transform-eval"
    assert summary["case_count"] == 2
    assert summary["candidate_count"] == 4
    assert (out / "summary.json").exists()
    assert (out / "candidate_summary.csv").exists()
    assert (out / "candidate_eval_summary.csv").exists()
    assert (out / "candidate_eval_summary.json").exists()
    assert (out / "case_summary.csv").exists()
    assert (out / "feature_stats.csv").exists()
    assert (out / "case_variation_summary.csv").exists()
    assert (out / "material_coverage.csv").exists()
    assert (out / "feature_scalar_summary.csv").exists()
    assert (out / "scalar_variation_summary.csv").exists()
    assert (out / "feature_profile_values.csv").exists()
    assert (out / "profile_variation_summary.csv").exists()
    assert (out / "_run" / "run_info.json").exists()
    assert (out / "candidates" / "raw" / "cases" / "case_a" / "features" / "sdf_raw.npz").exists()
    assert (
        out
        / "candidates"
        / "profile"
        / "cases"
        / "case_b"
        / "features"
        / "material_profile.csv"
    ).exists()
    assert (
        out
        / "candidates"
        / "material"
        / "cases"
        / "case_b"
        / "features"
        / "material_sdf.npz"
    ).exists()
    candidate_summary = (out / "candidate_summary.csv").read_text(encoding="utf-8")
    candidate_eval = (out / "candidate_eval_summary.csv").read_text(encoding="utf-8")
    feature_stats = (out / "feature_stats.csv").read_text(encoding="utf-8")
    case_variation = (out / "case_variation_summary.csv").read_text(encoding="utf-8")
    material_coverage = (out / "material_coverage.csv").read_text(encoding="utf-8")
    scalar_summary = (out / "feature_scalar_summary.csv").read_text(encoding="utf-8")
    scalar_variation = (out / "scalar_variation_summary.csv").read_text(encoding="utf-8")
    profile_values = (out / "feature_profile_values.csv").read_text(encoding="utf-8")
    profile_variation = (out / "profile_variation_summary.csv").read_text(
        encoding="utf-8"
    )
    assert "raw,sdf_raw" in candidate_summary
    assert "profile,material_profile" in candidate_summary
    assert "variable_array_count" in candidate_summary
    assert "variable_scalar_count" in candidate_summary
    assert "variable_profile_count" in candidate_summary
    assert "candidate,features,case_count,signal_status,varying_output_count" in candidate_eval
    assert "raw,sdf_raw,2,varies" in candidate_eval
    assert "profile,material_profile,2,varies" in candidate_eval
    assert "tsdf,case_a,tsdf_views,tsdf_10nm" in feature_stats
    assert "array_hash" in feature_stats
    assert "material,case_b,material_sdf,sdf_nm" in feature_stats
    assert "raw,sdf_raw,sdf_nm,2,2,true" in case_variation
    assert "material,material_sdf,sdf_nm,2,2,true" in case_variation
    assert "material,case_b,material_sdf,1" in material_coverage
    assert (
        "profile,case_b,material_profile,material_profile_summary.json,material_count,3.0"
        in scalar_summary
    )
    assert "profile,material_profile,material_count,2,2,true" in scalar_variation
    assert "profile,case_b,material_profile,material_profile.csv,material_id,1,voxel_count" in (
        profile_values
    )
    assert "profile,material_profile,material_id,1,voxel_count,2,2,true" in (
        profile_variation
    )


def test_transform_eval_writes_diagnostic_figures(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
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
    out = tmp_path / "transform_eval_figures"
    cfg = tmp_path / "transform_eval_figures.yaml"
    cfg.write_text(
        f"""
task: transform-eval
input:
  index: {index.name}
eval:
  candidates:
    raw:
      features:
        use: [sdf_raw]
    material:
      features:
        use: [material_sdf]
    profile:
      features:
        use: [material_profile]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_eval_from_config(cfg)

    assert summary["figures"]["status"] == "OK"
    manifest = json.loads((out / "figures" / "figure_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "OK"
    assert (out / "figures" / "input_shape_sections.png").exists()
    assert (out / "figures" / "feature_method_overview.png").exists()
    assert (out / "figures" / "feature_representation_score.png").exists()
    assert (out / "figures" / "candidate_signal_heatmap.png").exists()
    assert (out / "figures" / "candidate_signal_cost.png").exists()
    assert (out / "figures" / "README.md").exists()
    assert "how_to_read" in manifest
    assert "input_shape_sections.png" in manifest["how_to_read"]
    assert list((out / "figures" / "representative_feature_slices").glob("*.png"))
    rows = list(
        csv.DictReader(
            (out / "figures" / "feature_representation_scores.csv").open(
                "r",
                encoding="utf-8",
            )
        )
    )
    assert rows
    assert "representation_score" in rows[0]
    assert "variation_capture" in rows[0]
    assert "compactness" in rows[0]


def test_transform_eval_removes_stale_candidate_and_figure_outputs(tmp_path: Path) -> None:
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
  candidates:
    raw:
      features:
        use: [sdf_raw]
output:
  dir: {out}
""",
        encoding="utf-8",
    )
    run_transform_eval_from_config(cfg)
    stale_candidate = out / "candidates" / "old_candidate" / "cases" / "old" / "preview.png"
    stale_figure = out / "figures" / "old.png"
    stale_candidate.parent.mkdir(parents=True)
    stale_figure.parent.mkdir(parents=True, exist_ok=True)
    stale_candidate.write_text("old", encoding="utf-8")
    stale_figure.write_text("old", encoding="utf-8")

    run_transform_eval_from_config(cfg)

    assert not stale_candidate.exists()
    assert not stale_figure.exists()
    assert (out / "candidates" / "raw" / "cases" / "case_a" / "preview.png").exists()


def test_transform_eval_skips_diagnostic_figures_without_matplotlib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wafergeo.compare.eval_figures as eval_figures

    monkeypatch.setattr(eval_figures, "_load_matplotlib_pyplot", lambda: None)
    sim = write_npz(tmp_path / "sim.npz")
    index = tmp_path / "cases.csv"
    index.write_text(
        f"""case_id,input_kind,input_path
case_a,npz_label,{sim.name}
""",
        encoding="utf-8",
    )
    out = tmp_path / "transform_eval_no_mpl"
    cfg = tmp_path / "transform_eval_no_mpl.yaml"
    cfg.write_text(
        f"""
task: transform-eval
input:
  index: {index.name}
eval:
  candidates:
    raw:
      features:
        use: [sdf_raw]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_eval_from_config(cfg)

    assert summary["status"] == "OK"
    assert summary["figures"]["status"] == "SKIPPED"
    manifest = json.loads((out / "figures" / "figure_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "SKIPPED"
    assert manifest["reason"] == "matplotlib is not installed"


def test_transform_eval_rejects_candidate_output_collisions(tmp_path: Path) -> None:
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
  candidates:
    a_b:
      features:
        use: [sdf_raw]
    a b:
      features:
        use: [sdf_raw]
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="candidate names collide"):
        run_transform_eval_from_config(cfg)


def test_transform_eval_requires_candidates(tmp_path: Path) -> None:
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


def test_transform_eval_collects_process_delta_scalars(tmp_path: Path) -> None:
    initial_a = np.zeros((4, 4, 2), dtype=np.uint8)
    initial_a[0:2, :, :] = 1
    final_a = initial_a.copy()
    final_a[0, 0, 0] = 0
    initial_b = initial_a.copy()
    final_b = initial_b.copy()
    final_b[0, 0, 0] = 0
    final_b[1, 1, 0] = 2

    paths = {
        "initial_a": tmp_path / "initial_a.npz",
        "final_a": tmp_path / "final_a.npz",
        "initial_b": tmp_path / "initial_b.npz",
        "final_b": tmp_path / "final_b.npz",
    }
    for key, labels in {
        "initial_a": initial_a,
        "final_a": final_a,
        "initial_b": initial_b,
        "final_b": final_b,
    }.items():
        np.savez(
            paths[key],
            labels=labels,
            spacing=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            material_ids=np.array([0, 1, 2], dtype=np.int32),
        )

    index = tmp_path / "process_cases.csv"
    index.write_text(
        f"""case_id,input_kind,input_path,reference_kind,reference_path
case_a,npz_label,{paths["final_a"].name},npz_label,{paths["initial_a"].name}
case_b,npz_label,{paths["final_b"].name},npz_label,{paths["initial_b"].name}
""",
        encoding="utf-8",
    )
    out = tmp_path / "transform_eval_process"
    cfg = tmp_path / "transform_eval_process.yaml"
    cfg.write_text(
        f"""
task: transform-eval
input:
  index: {index.name}
process:
  enabled: true
eval:
  candidates:
    process:
      features:
        use: [process_delta_profile]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_transform_eval_from_config(cfg)

    assert summary["candidate_count"] == 1
    scalar_summary = (out / "feature_scalar_summary.csv").read_text(encoding="utf-8")
    scalar_variation = (out / "scalar_variation_summary.csv").read_text(encoding="utf-8")
    profile_values = (out / "feature_profile_values.csv").read_text(encoding="utf-8")
    assert (
        "process,case_a,process_delta_profile,process_delta_summary.json,changed_voxels,1.0"
        in scalar_summary
    )
    assert (
        "process,case_b,process_delta_profile,process_delta_summary.json,changed_voxels,2.0"
        in scalar_summary
    )
    assert "process,process_delta_profile,changed_voxels,2,2,true,1.0,2.0,1.0" in scalar_variation
    assert (
        "process,case_b,process_delta_profile,process_delta_profile.csv,transition_key,1_to_2"
        in profile_values
    )
