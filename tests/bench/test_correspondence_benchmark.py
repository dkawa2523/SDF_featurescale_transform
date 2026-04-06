from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from wafergeo.bench.correspondence import runner
from wafergeo.bench.correspondence.generator import load_benchmark_scenario
from wafergeo.bench.correspondence.metrics import diagnose_root_cause
from wafergeo.bench.correspondence.spec import BenchmarkSpecV1


def _base_spec(**kwargs) -> BenchmarkSpecV1:
    data = {
        "schema_version": "correspondence_bench/v1",
        "case_id": "bench_test",
        "scenarios": ("cube",),
        "point_to_cell_policies": ("nearest",),
        "mesh_backends": ("naive_interface",),
        "mesh_modes": ("material_shell",),
        "diagnosis_scope": "auto",
        "thresholds": {
            "mesh_boundary_iou_min": 0.5,
            "mesh_boundary_chamfer_nm_max": 3.0,
            "mesh_boundary_coverage_min": 0.2,
            "sdf_roundtrip_acc_min": 0.95,
            "render_diff_rate_max": 0.3,
            "policy_gap_max": 0.01,
        },
        "real_vti_path": None,
    }
    data.update(kwargs)
    return BenchmarkSpecV1(**data)


def test_bench_cube_roundtrip_is_high(tmp_path: Path) -> None:
    spec = _base_spec(scenarios=("cube",))
    manifest = runner.run_correspondence_benchmark(spec, tmp_path / "out")
    assert float(manifest["summary"]["sdf_roundtrip_acc_mean"]) >= 0.999


def test_bench_mesh_material_shell_beats_interface_for_solid_compare(tmp_path: Path) -> None:
    spec = _base_spec(
        scenarios=("cube",),
        mesh_modes=("material_shell", "interface_mesh"),
    )
    manifest = runner.run_correspondence_benchmark(spec, tmp_path / "out")
    rows = manifest["rows"]
    shell_iou = max(
        float(row["mesh_boundary_iou"])
        for row in rows
        if row["mesh_mode"] == "material_shell" and row["status"] == "OK"
    )
    interface_iou = max(
        float(row["mesh_boundary_iou"])
        for row in rows
        if row["mesh_mode"] == "interface_mesh" and row["status"] == "OK"
    )
    assert shell_iou >= interface_iou


def test_bench_mesh_surface_metrics_cube(tmp_path: Path) -> None:
    spec = _base_spec(
        scenarios=("cube",),
        mesh_backends=("vtk",),
        mesh_modes=("material_shell",),
    )
    manifest = runner.run_correspondence_benchmark(spec, tmp_path / "out")
    rows = [row for row in manifest["rows"] if row["status"] == "OK"]
    assert rows
    chamfer = min(float(row["mesh_boundary_chamfer_nm"]) for row in rows)
    coverage = max(float(row["mesh_boundary_coverage"]) for row in rows)
    assert chamfer < 1.0
    assert coverage > 0.9


def test_bench_nearest_majority_gap_reported(tmp_path: Path) -> None:
    spec = _base_spec(
        scenarios=("diagonal",),
        point_to_cell_policies=("nearest", "majority"),
    )
    manifest = runner.run_correspondence_benchmark(spec, tmp_path / "out")
    assert float(manifest["summary"]["policy_gap_max"]) > 0.0


def test_majority_nearest_tie_reduces_policy_gap(tmp_path: Path) -> None:
    spec_old = _base_spec(
        scenarios=("thin_shell",),
        point_to_cell_policies=("nearest", "majority"),
    )
    old_manifest = runner.run_correspondence_benchmark(spec_old, tmp_path / "old")

    spec_new = _base_spec(
        scenarios=("thin_shell",),
        point_to_cell_policies=("nearest", "majority_nearest_tie"),
    )
    new_manifest = runner.run_correspondence_benchmark(spec_new, tmp_path / "new")

    assert float(new_manifest["summary"]["policy_gap_max"]) < float(
        old_manifest["summary"]["policy_gap_max"]
    )


def test_bench_root_cause_rule_triggers_mesh_stage() -> None:
    result = diagnose_root_cause(
        {
            "sdf_roundtrip_acc_mean": 0.9995,
            "material_shell_mesh_iou_mean": 0.3,
            "interface_mesh_iou_mean": 0.2,
            "material_shell_mesh_chamfer_nm_mean": 4.2,
            "interface_mesh_chamfer_nm_mean": 3.8,
            "material_shell_mesh_coverage_mean": 0.2,
            "interface_mesh_coverage_mean": 0.25,
            "render_diff_rate_mean": 0.01,
            "policy_gap_max": 0.0,
        },
        {
            "mesh_boundary_iou_min": 0.8,
            "mesh_boundary_chamfer_nm_max": 2.0,
            "mesh_boundary_coverage_min": 0.7,
            "sdf_roundtrip_acc_min": 0.999,
            "render_diff_rate_max": 0.1,
            "policy_gap_max": 0.05,
        },
    )
    assert "mesh_extraction_or_face_annotation" in result["root_cause_candidates"]


def test_policy_gap_scope_prefers_real_vti() -> None:
    result = diagnose_root_cause(
        {
            "sdf_roundtrip_acc_mean": 1.0,
            "material_shell_mesh_iou_mean": 0.99,
            "interface_mesh_iou_mean": 0.99,
            "material_shell_mesh_chamfer_nm_mean": 0.5,
            "interface_mesh_chamfer_nm_mean": 0.6,
            "material_shell_mesh_coverage_mean": 0.99,
            "interface_mesh_coverage_mean": 0.99,
            "render_diff_rate_mean": 0.0,
            "policy_gap_max": 0.2,
            "policy_gap_real_vti": 0.01,
            "policy_gap_scope_used": "real_vti",
        },
        {
            "mesh_boundary_iou_min": 0.8,
            "mesh_boundary_chamfer_nm_max": 2.0,
            "mesh_boundary_coverage_min": 0.7,
            "sdf_roundtrip_acc_min": 0.999,
            "render_diff_rate_max": 0.1,
            "policy_gap_max": 0.05,
        },
    )
    assert "Rule C" not in result["rules_triggered"]


def test_bench_outputs_exist_even_on_partial_fail(tmp_path: Path) -> None:
    spec = _base_spec(mesh_backends=("naive_interface", "unknown"))  # type: ignore[arg-type]
    manifest = runner.run_correspondence_benchmark(spec, tmp_path / "out")
    assert (tmp_path / "out" / "benchmark_manifest.json").exists()
    assert (tmp_path / "out" / "tables" / "stage_metrics.csv").exists()
    assert manifest["status"] in {"WARN", "OK"}


def test_bench_real_vti_smoke(tmp_path: Path, monkeypatch) -> None:
    scenario_cube = load_benchmark_scenario("cube")

    def _fake_load(name: str, *, real_vti_path: str | None = None):
        if name == "real_vti":
            return scenario_cube
        return load_benchmark_scenario(name, real_vti_path=real_vti_path)

    monkeypatch.setattr(runner, "load_benchmark_scenario", _fake_load)
    spec = _base_spec(
        scenarios=("real_vti",),
        real_vti_path="dummy.vti",
    )
    manifest = runner.run_correspondence_benchmark(spec, tmp_path / "out")
    assert "summary" in manifest
    assert "diagnosis" in manifest
    saved = json.loads((tmp_path / "out" / "benchmark_manifest.json").read_text())
    assert saved["case_id"] == "bench_test"


def test_manifest_backward_compat_keys(tmp_path: Path) -> None:
    spec = _base_spec(scenarios=("cube",))
    manifest = runner.run_correspondence_benchmark(spec, tmp_path / "out")
    summary = manifest["summary"]
    assert "material_shell_mesh_iou_mean" in summary
    assert "interface_mesh_iou_mean" in summary
    assert "material_shell_mesh_chamfer_nm_mean" in summary
    assert "interface_mesh_chamfer_nm_mean" in summary
    assert "material_shell_mesh_coverage_mean" in summary
    assert "interface_mesh_coverage_mean" in summary


def test_manifest_contains_policy_gap_breakdown(tmp_path: Path, monkeypatch) -> None:
    scenario_real = replace(load_benchmark_scenario("diagonal"), name="real_vti")

    def _fake_load(name: str, *, real_vti_path: str | None = None):
        if name == "real_vti":
            return scenario_real
        return load_benchmark_scenario(name, real_vti_path=real_vti_path)

    monkeypatch.setattr(runner, "load_benchmark_scenario", _fake_load)
    spec = _base_spec(
        scenarios=("real_vti", "thin_shell"),
        point_to_cell_policies=("nearest", "majority"),
        diagnosis_scope="auto",
        real_vti_path="dummy.vti",
    )
    manifest = runner.run_correspondence_benchmark(spec, tmp_path / "out")
    summary = manifest["summary"]
    assert "policy_gap_real_vti" in summary
    assert "policy_gap_synthetic_max" in summary
    assert "policy_gap_scope_used" in summary
    assert summary["policy_gap_scope_used"] == "real_vti"


def test_strict_gate_requires_all_rows_pass(tmp_path: Path) -> None:
    spec = _base_spec(
        scenarios=("cube",),
        thresholds={
            "mesh_boundary_iou_min": 0.5,
            "mesh_boundary_chamfer_nm_max": 0.1,
            "mesh_boundary_coverage_min": 0.2,
            "sdf_roundtrip_acc_min": 0.95,
            "render_diff_rate_max": 0.3,
            "policy_gap_max": 0.01,
        },
    )
    manifest = runner.run_correspondence_benchmark(spec, tmp_path / "out")
    summary = manifest["summary"]
    assert summary["strict_overall_pass"] is False
    assert any(v > 0 for v in summary["scenario_fail_counts"].values())


def test_manifest_has_strict_fields(tmp_path: Path) -> None:
    spec = _base_spec(
        scenarios=("cube",),
        point_to_cell_policies=("nearest", "majority_nearest_tie"),
    )
    manifest = runner.run_correspondence_benchmark(spec, tmp_path / "out")
    summary = manifest["summary"]
    diagnosis = manifest["diagnosis"]
    assert "strict_overall_pass" in summary
    assert "scenario_fail_counts" in summary
    assert "scenario_pass_rate" in summary
    assert "strict_fail_scenarios" in diagnosis
