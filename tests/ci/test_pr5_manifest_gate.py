from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_manifest(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _base_audit_manifest(**kwargs: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "vti_audit/v2",
        "profile_id": "vti_standard_full_v1",
        "profile_hash": "abc123",
        "status": "OK",
        "messages": [],
        "input_path": "input.vti",
        "input_hash": "input-hash",
        "source_array_name": "MaterialIds",
        "point_to_cell_policy": "nearest",
        "material_policy": "full",
        "mesh_mode": "material_shell",
        "mesh_backend": "vtk",
        "mesh_backend_used": "vtk",
        "flat_layout_used": "vtk_x_fastest",
        "metrics": {"global": {}, "slice": {}, "pair_counts": {}},
        "postprocess": {"enabled": True, "params": {}, "metrics": {}, "status": "OK"},
        "outputs": {"figures": [], "tables": []},
    }
    payload.update(kwargs)
    return payload


def _base_preview_manifest(**kwargs: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "vti_preview/v2",
        "profile_id": "vti_standard_full_v1",
        "profile_hash": "abc123",
        "status": "WARN",
        "messages": ["sdf fallback: missing scipy"],
        "input_path": "input.vti",
        "input_hash": "input-hash",
        "created_at": "2026-03-29T00:00:00+00:00",
        "generator_version": "0.1.0",
        "outputs": {"figures": [], "tables": [], "sdf": []},
        "audit_manifest_path": "audit_manifest.json",
        "sdf": {"backend": "scipy", "shape": [1, 1, 1], "mu_nm": 20.0},
        "postprocess": {"enabled": True, "params": {}, "metrics": {}, "status": "OK"},
    }
    payload.update(kwargs)
    return payload


def _base_benchmark_manifest(**kwargs: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "correspondence_bench_manifest/v1",
        "case_id": "bench_case",
        "spec_hash": "abc123",
        "code_version": "0.1.0",
        "status": "WARN",
        "messages": ["sdf backend fallback: missing scipy"],
        "thresholds": {},
        "summary": {},
        "diagnosis": {},
        "rows": [],
        "outputs": {"tables": [], "figures": []},
    }
    payload.update(kwargs)
    return payload


def test_manifest_gate_accepts_valid_audit_and_preview(tmp_path: Path) -> None:
    audit_path = _write_manifest(tmp_path / "audit.json", _base_audit_manifest())
    preview_path = _write_manifest(
        tmp_path / "preview.json",
        _base_preview_manifest(
            messages=["sdf fallback: missing scipy", "shell extraction fallback: missing vtk"],
        ),
    )
    script = Path("scripts/pr5_manifest_gate.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(audit_path),
            "--manifest",
            str(preview_path),
            "--expected-fallback-warning-count",
            "2",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "OK"
    assert payload["fallback_warning_count"] == 2


def test_manifest_gate_accepts_benchmark_manifest_schema(tmp_path: Path) -> None:
    bench_path = _write_manifest(tmp_path / "bench.json", _base_benchmark_manifest())
    script = Path("scripts/pr5_manifest_gate.py")
    proc = subprocess.run(
        [sys.executable, str(script), "--manifest", str(bench_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "OK"
    assert payload["manifests"][0]["schema_version"] == "correspondence_bench_manifest/v1"


def test_manifest_gate_rejects_status_message_inconsistency(tmp_path: Path) -> None:
    manifest = _base_audit_manifest(status="OK", messages=["unexpected warning"])
    path = _write_manifest(tmp_path / "bad.json", manifest)
    script = Path("scripts/pr5_manifest_gate.py")
    proc = subprocess.run(
        [sys.executable, str(script), "--manifest", str(path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    payload = json.loads(proc.stdout)
    assert payload["status"] == "FAIL"
    assert any("must not carry messages" in error for error in payload["errors"])


def test_manifest_gate_rejects_missing_required_keys(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "bad.json",
        {
            "schema_version": "vti_preview/v2",
            "profile_id": "vti_standard_full_v1",
            "status": "OK",
            "messages": [],
        },
    )
    script = Path("scripts/pr5_manifest_gate.py")
    proc = subprocess.run(
        [sys.executable, str(script), "--manifest", str(path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    payload = json.loads(proc.stdout)
    assert payload["status"] == "FAIL"
    assert any("missing required keys" in error for error in payload["errors"])


def test_manifest_gate_counts_nested_fallback_messages(tmp_path: Path) -> None:
    manifest = _base_preview_manifest(
        messages=["sdf fallback: missing scipy"],
        postprocess={
            "enabled": True,
            "params": {},
            "metrics": {},
            "status": "WARN",
            "messages": ["postprocess fallback: vtk not available"],
        },
    )
    path = _write_manifest(tmp_path / "nested.json", manifest)
    script = Path("scripts/pr5_manifest_gate.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(path),
            "--expected-fallback-warning-count",
            "2",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["fallback_warning_count"] == 2
