from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Any, cast

import numpy as np

import wafergeo
from wafergeo.bench.correspondence.generator import (
    BenchmarkScenarioData,
    as_label_volume_for_policy,
    load_benchmark_scenario,
)
from wafergeo.bench.correspondence.metrics import compute_stage_metrics, diagnose_root_cause
from wafergeo.bench.correspondence.spec import BenchmarkSpecV1, benchmark_spec_hash
from wafergeo.core.meta import Meta
from wafergeo.core.types import LabelVolume, MaterialSpec, TSDFVolume
from wafergeo.mesh.build import build_mesh_from_tsdf
from wafergeo.mesh.config import MeshBuildConfig
from wafergeo.mesh.errors import MeshOptionalDependencyError
from wafergeo.sdf.build import build_tsdf_volume
from wafergeo.sdf.config import SDFBuildConfig
from wafergeo.sdf.errors import OptionalDependencyUnavailableError
from wafergeo.sdf.tsdf import label_from_tsdf


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "matplotlib is required for benchmark figures. Install with: pip install -e '.[viz]'"
        ) from exc
    return plt


def _spacing_origin_from_label(
    label,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    spacing_zyx = (
        float(label.grid.spacing[0]),
        float(label.grid.spacing[1]),
        float(label.grid.spacing[2]),
    )
    origin_zyx = (
        float(label.grid.origin[0]),
        float(label.grid.origin[1]),
        float(label.grid.origin[2]),
    )
    return spacing_zyx, origin_zyx


def _write_stage_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "scenario",
        "policy",
        "mesh_backend",
        "mesh_backend_used",
        "mesh_mode",
        "sdf_backend_used",
        "flat_layout_used",
        "status",
        "point_to_cell_match",
        "sdf_roundtrip_acc",
        "mesh_boundary_iou",
        "mesh_boundary_dice",
        "mesh_boundary_chamfer_nm",
        "mesh_boundary_coverage",
        "bbox_center_shift_nm",
        "bbox_size_l2_nm",
        "surface_area_rel_error",
        "render_diff_rate",
        "row_pass_all_thresholds",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})


def _write_summary_table(path: Path, summary: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key in sorted(summary.keys()):
            writer.writerow([key, summary[key]])


def _plot_summary_figure(path: Path, rows: list[dict[str, object]]) -> None:
    plt = _require_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = []
    values = []
    for row in rows:
        if row["status"] != "OK":
            continue
        if row["mesh_boundary_iou"] is None:
            continue
        labels.append(
            f"{row['scenario']}|{row['policy']}|{row['mesh_backend']}|{row['mesh_mode']}"
        )
        value_raw = row["mesh_boundary_iou"]
        if not isinstance(value_raw, (int, float, np.floating)):
            continue
        values.append(float(value_raw))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
    if values:
        x = np.arange(len(values))
        ax.bar(x, values, color="#4c78a8")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mesh Boundary IoU")
    ax.set_title("Correspondence Benchmark")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _normalize_label_to_selected_ids(
    label: LabelVolume,
    *,
    selected_ids: list[int],
    outside_id: int,
) -> LabelVolume:
    selected = [int(v) for v in selected_ids]
    arr = label.material_id.copy()
    keep = np.asarray(selected, dtype=np.int64)
    arr[~np.isin(arr, keep)] = int(outside_id)

    old_name = {
        int(mid): name
        for mid, name in zip(label.material.ids, label.material.names, strict=True)
    }
    old_prio = {
        int(mid): int(prio)
        for mid, prio in zip(label.material.ids, label.material.priority, strict=True)
    }
    names = [old_name.get(mid, f"material_{mid}") for mid in selected]
    priority = [old_prio.get(mid, 1000 - i) for i, mid in enumerate(selected)]
    ignore = [mid == int(outside_id) for mid in selected]
    material = MaterialSpec(
        ids=selected,
        names=names,
        void_id=int(outside_id),
        priority=priority,
        ignore_in_exposure=ignore,
    )
    dtype = np.uint8 if max(selected) <= 255 else np.uint16
    return LabelVolume(
        grid=label.grid,
        material=material,
        material_id=arr.astype(dtype, copy=False),
        meta=label.meta,
    )


def _proxy_tsdf_from_label(
    label: LabelVolume,
    *,
    selected_ids: list[int],
    mu_nm: float,
) -> TSDFVolume:
    m = len(selected_ids)
    shape_zyx = label.material_id.shape
    tsdf = np.ones((m,) + shape_zyx, dtype=np.float32)
    for channel, material_id in enumerate(selected_ids):
        tsdf[channel, label.material_id == int(material_id)] = -1.0

    extra = dict(label.meta.extra)
    extra["sdf_backend"] = "proxy_numpy"
    extra["sdf_engine_name"] = "proxy_numpy"
    extra["sdf_engine_version"] = "1.0.0"
    extra["selected_material_ids"] = ",".join(str(v) for v in selected_ids)
    meta = Meta(
        schema_version=label.meta.schema_version,
        profile_id=label.meta.profile_id,
        config_hash=label.meta.config_hash,
        generator_version=label.meta.generator_version,
        git_commit=label.meta.git_commit,
        input_hash=label.meta.input_hash,
        created_at=label.meta.created_at,
        extra=extra,
    )
    present_mask = np.array(
        [np.any(label.material_id == int(material_id)) for material_id in selected_ids],
        dtype=bool,
    )
    return TSDFVolume(
        grid=label.grid,
        material=label.material,
        mu_nm=mu_nm,
        tsdf=tsdf,
        present_mask=present_mask,
        meta=meta,
    )


def _build_tsdf_with_fallback(
    label: LabelVolume,
    *,
    warnings: list[str],
) -> TSDFVolume:
    primary_cfg = SDFBuildConfig(
        mu_nm=20.0,
        backend="scipy",
        include_void_channel=True,
        boundary_features=True,
        compute_present_mask=True,
    )
    try:
        tsdf, _ = build_tsdf_volume(label, primary_cfg)
        return tsdf
    except (OptionalDependencyUnavailableError, ImportError) as exc:
        warnings.append(f"sdf backend fallback: {exc}")
        selected_ids = list(label.material.ids)
        return _proxy_tsdf_from_label(
            label,
            selected_ids=selected_ids,
            mu_nm=float(primary_cfg.mu_nm),
        )


def _run_single_combo(
    *,
    scenario: BenchmarkScenarioData,
    policy: str,
    mesh_backend: str,
    mesh_mode: str,
    fixed_selected_ids: list[int] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario": scenario.name,
        "policy": policy,
        "mesh_backend": mesh_backend,
        "mesh_backend_used": mesh_backend,
        "mesh_mode": mesh_mode,
        "sdf_backend_used": "scipy",
        "status": "OK",
        "point_to_cell_match": None,
        "flat_layout_used": "",
        "error": "",
    }
    warnings: list[str] = []
    try:
        label, ingest_info = as_label_volume_for_policy(
            scenario,
            point_to_cell_policy=policy,
            max_materials=5,
        )
        if fixed_selected_ids is not None:
            label = _normalize_label_to_selected_ids(
                label,
                selected_ids=fixed_selected_ids,
                outside_id=int(scenario.outside_material_id),
            )
        row["point_to_cell_match"] = ingest_info["point_to_cell_match"]

        tsdf = _build_tsdf_with_fallback(label, warnings=warnings)
        row["sdf_backend_used"] = (
            str(tsdf.meta.extra.get("sdf_backend", "scipy")) if tsdf.meta is not None else "scipy"
        )
        if tsdf.meta is not None and "selected_material_ids" in tsdf.meta.extra:
            selected_ids = [int(v) for v in tsdf.meta.extra["selected_material_ids"].split(",")]
        else:
            selected_ids = list(tsdf.material.ids[: tsdf.tsdf.shape[0]])
        void_index = tsdf.material.ids.index(tsdf.material.void_id)
        roundtrip = label_from_tsdf(
            tsdf.tsdf,
            tsdf.material,
            void_index=void_index,
            selected_material_ids=selected_ids,
        )

        mesh_cfg = MeshBuildConfig(
            backend=cast(Any, mesh_backend),
            mode=cast(Any, mesh_mode),
            sample_points_n=1024,
            sample_seed=0,
        )
        try:
            mesh, _, _ = build_mesh_from_tsdf(tsdf, mesh_cfg)
        except MeshOptionalDependencyError as exc:
            if mesh_backend != "vtk":
                raise
            warnings.append(f"mesh backend fallback: {exc}")
            mesh_cfg = MeshBuildConfig(
                backend="naive_interface",
                mode=cast(Any, mesh_mode),
                sample_points_n=1024,
                sample_seed=0,
            )
            row["mesh_backend_used"] = "naive_interface"
            mesh, _, _ = build_mesh_from_tsdf(tsdf, mesh_cfg)

        spacing_zyx, origin_zyx = _spacing_origin_from_label(label)
        metrics = compute_stage_metrics(
            reference_label_zyx=label.material_id,
            sdf_roundtrip_label_zyx=roundtrip,
            spacing_zyx=spacing_zyx,
            origin_zyx=origin_zyx,
            mesh_vertices=mesh.vertices,
            mesh_faces=mesh.faces,
        )
        row.update(metrics)
        row["selected_material_ids"] = selected_ids
        row["material_count_source"] = ingest_info["material_count_source"]
        if "flat_layout_used" in ingest_info:
            row["flat_layout_used"] = ingest_info["flat_layout_used"]
        if warnings:
            row["warnings"] = list(warnings)
    except Exception as exc:
        row["status"] = "FAIL"
        row["error"] = str(exc)
        row.update(
            {
                "sdf_roundtrip_acc": None,
                "mesh_boundary_iou": None,
                "mesh_boundary_dice": None,
                "mesh_boundary_chamfer_nm": None,
                "mesh_boundary_coverage": None,
                "bbox_center_shift_nm": None,
                "bbox_size_l2_nm": None,
                "surface_area_rel_error": None,
                "render_diff_rate": None,
            }
        )
    return row


def _row_passes_thresholds(
    row: Mapping[str, object],
    *,
    thresholds: Mapping[str, float],
) -> bool:
    if row.get("status") != "OK":
        return False

    sdf = _as_float(row.get("sdf_roundtrip_acc"))
    iou = _as_float(row.get("mesh_boundary_iou"))
    chamfer = _as_float(row.get("mesh_boundary_chamfer_nm"))
    coverage = _as_float(row.get("mesh_boundary_coverage"))
    if sdf is None or iou is None or chamfer is None or coverage is None:
        return False

    return (
        sdf >= float(thresholds["sdf_roundtrip_acc_min"])
        and iou >= float(thresholds["mesh_boundary_iou_min"])
        and chamfer <= float(thresholds["mesh_boundary_chamfer_nm_max"])
        and coverage >= float(thresholds["mesh_boundary_coverage_min"])
    )


def _aggregate_metrics(
    rows: list[dict[str, object]],
    *,
    diagnosis_scope: str,
    thresholds: Mapping[str, float],
    policy_order: tuple[str, ...],
) -> dict[str, object]:
    for row in rows:
        row["row_pass_all_thresholds"] = _row_passes_thresholds(
            row,
            thresholds=thresholds,
        )

    scenarios = sorted({str(row["scenario"]) for row in rows})
    scenario_pass_rate: dict[str, float] = {}
    scenario_fail_counts: dict[str, int] = {}
    for scenario in scenarios:
        rows_s = [row for row in rows if str(row["scenario"]) == scenario]
        if not rows_s:
            continue
        pass_count = sum(1 for row in rows_s if bool(row.get("row_pass_all_thresholds")))
        fail_count = len(rows_s) - pass_count
        scenario_pass_rate[scenario] = float(pass_count) / float(len(rows_s))
        scenario_fail_counts[scenario] = int(fail_count)
    strict_overall_pass = all(bool(row.get("row_pass_all_thresholds")) for row in rows)

    ok_rows = [row for row in rows if row["status"] == "OK"]
    if not ok_rows:
        scope = diagnosis_scope
        if scope == "auto":
            scope = "real_vti" if "real_vti" in scenarios else "global_max"
        return {
            "sdf_roundtrip_acc_mean": 0.0,
            "material_shell_mesh_iou_mean": 0.0,
            "interface_mesh_iou_mean": 0.0,
            "material_shell_mesh_chamfer_nm_mean": math.inf,
            "interface_mesh_chamfer_nm_mean": math.inf,
            "material_shell_mesh_coverage_mean": 0.0,
            "interface_mesh_coverage_mean": 0.0,
            "render_diff_rate_mean": 1.0,
            "policy_gap_real_vti": 0.0,
            "policy_gap_synthetic_max": 0.0,
            "policy_gap_scope_used": scope,
            "policy_gap_max": 1.0,
            "scenario_pass_rate": scenario_pass_rate,
            "scenario_fail_counts": scenario_fail_counts,
            "strict_overall_pass": strict_overall_pass,
        }

    def _mean_of(key: str, where=None) -> float:
        values = []
        for row in ok_rows:
            if where is not None and not where(row):
                continue
            value = _as_float(row.get(key))
            if value is None:
                continue
            values.append(value)
        return float(mean(values)) if values else 0.0

    policy_to_matches: dict[tuple[str, str], float] = {}
    for row in ok_rows:
        if row["point_to_cell_match"] is None:
            continue
        key = (str(row["scenario"]), str(row["policy"]))
        value = _as_float(row["point_to_cell_match"])
        if value is not None:
            policy_to_matches[key] = value

    scenario_to_gap: dict[str, float] = {}
    scenarios = sorted({str(row["scenario"]) for row in rows})
    for scenario in scenarios:
        per_policy = {
            policy: value
            for (scn, policy), value in policy_to_matches.items()
            if scn == scenario
        }
        if not per_policy:
            continue
        baseline_policy = "nearest" if "nearest" in per_policy else None
        if baseline_policy is None:
            for policy in policy_order:
                if policy in per_policy:
                    baseline_policy = policy
                    break
        if baseline_policy is None:
            baseline_policy = next(iter(per_policy.keys()))

        baseline = per_policy[baseline_policy]
        others = [v for p, v in per_policy.items() if p != baseline_policy]
        scenario_to_gap[scenario] = max([abs(baseline - v) for v in others], default=0.0)

    policy_gap_real_vti = float(scenario_to_gap.get("real_vti", 0.0))
    synthetic_gaps = [v for k, v in scenario_to_gap.items() if k != "real_vti"]
    policy_gap_synthetic_max = max(synthetic_gaps) if synthetic_gaps else 0.0
    policy_gap_global_max = max(scenario_to_gap.values()) if scenario_to_gap else 0.0

    scope = diagnosis_scope
    if scope == "auto":
        scope = "real_vti" if "real_vti" in scenario_to_gap else "global_max"
    if scope == "real_vti":
        policy_gap_effective = policy_gap_real_vti
    else:
        policy_gap_effective = policy_gap_global_max

    return {
        "sdf_roundtrip_acc_mean": _mean_of("sdf_roundtrip_acc"),
        "material_shell_mesh_iou_mean": _mean_of(
            "mesh_boundary_iou",
            where=lambda row: row["mesh_mode"] == "material_shell",
        ),
        "interface_mesh_iou_mean": _mean_of(
            "mesh_boundary_iou",
            where=lambda row: row["mesh_mode"] == "interface_mesh",
        ),
        "material_shell_mesh_chamfer_nm_mean": _mean_of(
            "mesh_boundary_chamfer_nm",
            where=lambda row: row["mesh_mode"] == "material_shell",
        ),
        "interface_mesh_chamfer_nm_mean": _mean_of(
            "mesh_boundary_chamfer_nm",
            where=lambda row: row["mesh_mode"] == "interface_mesh",
        ),
        "material_shell_mesh_coverage_mean": _mean_of(
            "mesh_boundary_coverage",
            where=lambda row: row["mesh_mode"] == "material_shell",
        ),
        "interface_mesh_coverage_mean": _mean_of(
            "mesh_boundary_coverage",
            where=lambda row: row["mesh_mode"] == "interface_mesh",
        ),
        "render_diff_rate_mean": _mean_of("render_diff_rate"),
        "policy_gap_real_vti": policy_gap_real_vti,
        "policy_gap_synthetic_max": policy_gap_synthetic_max,
        "policy_gap_scope_used": scope,
        "policy_gap_max": policy_gap_effective,
        "scenario_pass_rate": scenario_pass_rate,
        "scenario_fail_counts": scenario_fail_counts,
        "strict_overall_pass": strict_overall_pass,
    }


def _as_float(value: object | None) -> float | None:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def run_correspondence_benchmark(
    spec: BenchmarkSpecV1,
    out_dir: str | Path,
) -> dict[str, object]:
    out_root = Path(out_dir)
    figures_dir = out_root / "figures"
    tables_dir = out_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    messages: list[str] = []
    status = "OK"

    for scenario_name in spec.scenarios:
        scenario = load_benchmark_scenario(
            scenario_name,
            real_vti_path=spec.real_vti_path,
        )
        baseline_label, _ = as_label_volume_for_policy(
            scenario,
            point_to_cell_policy="nearest",
            max_materials=5,
        )
        fixed_selected_ids = [int(v) for v in baseline_label.material.ids]
        for policy in spec.point_to_cell_policies:
            for backend in spec.mesh_backends:
                for mode in spec.mesh_modes:
                    row = _run_single_combo(
                        scenario=scenario,
                        policy=policy,
                        mesh_backend=backend,
                        mesh_mode=mode,
                        fixed_selected_ids=fixed_selected_ids,
                    )
                    rows.append(row)
                    if row["status"] != "OK":
                        status = "WARN"
                        messages.append(
                            f"{scenario_name}/{policy}/{backend}/{mode}: {row['error']}"
                        )
                    else:
                        warnings = row.get("warnings")
                        if isinstance(warnings, list) and warnings:
                            status = "WARN"
                            messages.extend(
                                f"{scenario_name}/{policy}/{backend}/{mode}: {str(message)}"
                                for message in warnings
                            )

    summary = _aggregate_metrics(
        rows,
        diagnosis_scope=spec.diagnosis_scope,
        thresholds=spec.thresholds,
        policy_order=tuple(str(v) for v in spec.point_to_cell_policies),
    )
    synthetic_gap = summary.get("policy_gap_synthetic_max", 0.0)
    scope_used = summary.get("policy_gap_scope_used", "global_max")
    if (
        isinstance(synthetic_gap, (int, float, np.floating))
        and float(synthetic_gap) > float(spec.thresholds["policy_gap_max"])
        and scope_used == "real_vti"
    ):
        messages.append(
            "synthetic policy gap exceeded threshold while diagnosis scope was real_vti"
        )
    diagnosis = diagnose_root_cause(summary, spec.thresholds)

    _write_stage_table(tables_dir / "stage_metrics.csv", rows)
    _write_summary_table(tables_dir / "summary_metrics.csv", summary)
    try:
        _plot_summary_figure(figures_dir / "mesh_boundary_iou.png", rows)
    except Exception as exc:  # pragma: no cover - env dependent
        status = "WARN"
        messages.append(f"figure warning: {exc}")

    manifest: dict[str, object] = {
        "schema_version": "correspondence_bench_manifest/v1",
        "case_id": spec.case_id,
        "spec_hash": benchmark_spec_hash(spec),
        "code_version": wafergeo.__version__,
        "status": status,
        "messages": messages,
        "thresholds": dict(spec.thresholds),
        "summary": summary,
        "diagnosis": diagnosis,
        "rows": rows,
        "outputs": {
            "tables": sorted(path.name for path in tables_dir.glob("*.csv")),
            "figures": sorted(path.name for path in figures_dir.glob("*.png")),
        },
    }
    (out_root / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return manifest
