from __future__ import annotations

from pathlib import Path

import pytest

from tests.compare.helpers import write_internal_boundary_npz, write_npz
from wafergeo.compare import run_batch_transform_from_config


def test_batch_transform_outputs_dataset_files(tmp_path: Path) -> None:
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
    out = tmp_path / "batch_transform"
    cfg = tmp_path / "batch_transform.yaml"
    cfg.write_text(
        f"""
task: batch-transform
input:
  index: {index.name}
features:
  use: [sdf_raw, udf, material_sdf]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_batch_transform_from_config(cfg)

    assert summary["task"] == "batch-transform"
    assert summary["case_count"] == 2
    assert (out / "summary.json").exists()
    assert (out / "dataset_index.csv").exists()
    assert (out / "features_summary.csv").exists()
    assert (out / "_run" / "run_info.json").exists()
    assert (out / "cases" / "case_a" / "features" / "sdf_raw.npz").exists()
    assert (out / "cases" / "case_a" / "features" / "udf.npz").exists()
    assert (out / "cases" / "case_b" / "features" / "material_sdf.npz").exists()
    features_summary = (out / "features_summary.csv").read_text(encoding="utf-8")
    assert "case_a,sdf_raw" in features_summary
    assert "case_b,material_sdf" in features_summary


def test_batch_transform_rejects_bad_index(tmp_path: Path) -> None:
    index = tmp_path / "bad_cases.csv"
    index.write_text("case_id,input_kind\ncase_a,npz_label\n", encoding="utf-8")
    cfg = tmp_path / "batch_transform_bad.yaml"
    cfg.write_text(
        f"""
task: batch-transform
input:
  index: {index.name}
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        run_batch_transform_from_config(cfg)


def test_batch_transform_process_mode_uses_reference_columns(tmp_path: Path) -> None:
    reference = write_npz(tmp_path / "initial.npz")
    simulation = write_npz(tmp_path / "final.npz")
    index = tmp_path / "cases_with_reference.csv"
    index.write_text(
        f"""case_id,input_kind,input_path,reference_kind,reference_path
case_a,npz_label,{simulation.name},npz_label,{reference.name}
""",
        encoding="utf-8",
    )
    out = tmp_path / "batch_process"
    cfg = tmp_path / "batch_process.yaml"
    cfg.write_text(
        f"""
task: batch-transform
input:
  index: {index.name}
process:
  enabled: true
features:
  use: [sdf_raw]
output:
  dir: {out}
""",
        encoding="utf-8",
    )

    summary = run_batch_transform_from_config(cfg)

    assert summary["case_count"] == 1
    dataset_index = (out / "dataset_index.csv").read_text(encoding="utf-8")
    label_summary = (out / "cases" / "case_a" / "label_summary.json").read_text(
        encoding="utf-8"
    )
    assert "reference_kind,reference_path" in dataset_index
    assert str(reference) in dataset_index
    assert "reference_label_volume" in label_summary
