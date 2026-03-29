from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StageMetrics:
    sdf_roundtrip_acc: float
    mesh_boundary_iou: float | None
    mesh_boundary_dice: float | None
    mesh_boundary_chamfer_nm: float | None
    mesh_boundary_coverage: float | None
    bbox_center_shift_nm: float | None
    bbox_size_l2_nm: float | None
    surface_area_rel_error: float | None
    render_diff_rate: float

    def to_dict(self) -> dict[str, float | None]:
        return {
            "sdf_roundtrip_acc": self.sdf_roundtrip_acc,
            "mesh_boundary_iou": self.mesh_boundary_iou,
            "mesh_boundary_dice": self.mesh_boundary_dice,
            "mesh_boundary_chamfer_nm": self.mesh_boundary_chamfer_nm,
            "mesh_boundary_coverage": self.mesh_boundary_coverage,
            "bbox_center_shift_nm": self.bbox_center_shift_nm,
            "bbox_size_l2_nm": self.bbox_size_l2_nm,
            "surface_area_rel_error": self.surface_area_rel_error,
            "render_diff_rate": self.render_diff_rate,
        }


def _interface_mask_3d(label_zyx: np.ndarray) -> np.ndarray:
    out = np.zeros(label_zyx.shape, dtype=bool)
    out[:-1, :, :] |= label_zyx[:-1, :, :] != label_zyx[1:, :, :]
    out[:, :-1, :] |= label_zyx[:, :-1, :] != label_zyx[:, 1:, :]
    out[:, :, :-1] |= label_zyx[:, :, :-1] != label_zyx[:, :, 1:]
    return out


def _interface_points_xyz(
    label_zyx: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float],
    origin_zyx: tuple[float, float, float],
) -> np.ndarray:
    mask = _interface_mask_3d(label_zyx)
    idx = np.argwhere(mask)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = origin_zyx[0] + idx[:, 0].astype(np.float32) * spacing_zyx[0]
    y = origin_zyx[1] + idx[:, 1].astype(np.float32) * spacing_zyx[1]
    x = origin_zyx[2] + idx[:, 2].astype(np.float32) * spacing_zyx[2]
    return np.column_stack((x, y, z)).astype(np.float32, copy=False)


def _mesh_surface_points_xyz(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    tri = vertices[faces]
    return np.mean(tri, axis=1).astype(np.float32, copy=False)


def _surface_chamfer_and_coverage(
    ref_points_xyz: np.ndarray,
    mesh_points_xyz: np.ndarray,
    *,
    tau_nm: float,
) -> tuple[float, float]:
    if ref_points_xyz.shape[0] == 0 and mesh_points_xyz.shape[0] == 0:
        return 0.0, 1.0
    if ref_points_xyz.shape[0] == 0 or mesh_points_xyz.shape[0] == 0:
        return math.inf, 0.0

    try:
        from scipy.spatial import cKDTree

        ref_tree = cKDTree(ref_points_xyz)
        mesh_tree = cKDTree(mesh_points_xyz)
        d_mesh_to_ref = ref_tree.query(mesh_points_xyz, k=1)[0]
        d_ref_to_mesh = mesh_tree.query(ref_points_xyz, k=1)[0]
    except Exception:  # pragma: no cover - env dependent
        d_mesh_to_ref = _nearest_neighbor_distances_numpy(mesh_points_xyz, ref_points_xyz)
        d_ref_to_mesh = _nearest_neighbor_distances_numpy(ref_points_xyz, mesh_points_xyz)

    chamfer = 0.5 * float(np.mean(d_mesh_to_ref) + np.mean(d_ref_to_mesh))
    coverage = float(np.mean(d_mesh_to_ref <= float(tau_nm)))
    return chamfer, coverage


def _nearest_neighbor_distances_numpy(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    if src_xyz.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if dst_xyz.shape[0] == 0:
        return np.full((src_xyz.shape[0],), np.inf, dtype=np.float64)

    src = np.asarray(src_xyz, dtype=np.float64)
    dst = np.asarray(dst_xyz, dtype=np.float64)
    out = np.empty((src.shape[0],), dtype=np.float64)
    for start in range(0, src.shape[0], chunk_size):
        stop = min(start + chunk_size, src.shape[0])
        chunk = src[start:stop]
        diff = chunk[:, None, :] - dst[None, :, :]
        dist_sq = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
        out[start:stop] = np.sqrt(np.min(dist_sq, axis=1))
    return out


def _bbox_nm(
    points_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if points_xyz.shape[0] == 0:
        return None
    mn = points_xyz.min(axis=0).astype(np.float64)
    mx = points_xyz.max(axis=0).astype(np.float64)
    return mn, mx


def _render_diff_rate(label_a: np.ndarray, label_b: np.ndarray) -> float:
    zmid = label_a.shape[0] // 2
    ymid = label_a.shape[1] // 2
    xmid = label_a.shape[2] // 2
    a_xy = label_a[zmid]
    b_xy = label_b[zmid]
    a_zy = label_a[:, :, xmid]
    b_zy = label_b[:, :, xmid]
    a_zx = label_a[:, ymid, :]
    b_zx = label_b[:, ymid, :]
    diffs = [
        float(np.mean(a_xy != b_xy)),
        float(np.mean(a_zy != b_zy)),
        float(np.mean(a_zx != b_zx)),
    ]
    return float(np.mean(diffs))


def _surface_area_proxy(
    interface_points_xyz: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> float:
    voxel_area = min(spacing_zyx) ** 2
    return float(interface_points_xyz.shape[0] * voxel_area)


def _mesh_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    if faces.size == 0:
        return 0.0
    tri = vertices[faces]
    v1 = tri[:, 1] - tri[:, 0]
    v2 = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(v1, v2), axis=1)
    return float(np.sum(area))


def compute_stage_metrics(
    *,
    reference_label_zyx: np.ndarray,
    sdf_roundtrip_label_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    origin_zyx: tuple[float, float, float],
    mesh_vertices: np.ndarray | None,
    mesh_faces: np.ndarray | None,
    coverage_tau_nm: float | None = None,
) -> dict[str, float | None]:
    roundtrip_acc = float(np.mean(reference_label_zyx == sdf_roundtrip_label_zyx))

    render_diff = _render_diff_rate(reference_label_zyx, sdf_roundtrip_label_zyx)

    mesh_chamfer: float | None = None
    mesh_coverage: float | None = None
    mesh_iou: float | None = None
    mesh_dice: float | None = None
    center_shift: float | None = None
    size_l2: float | None = None
    surface_rel_error: float | None = None

    if mesh_vertices is not None and mesh_faces is not None and mesh_faces.size > 0:
        ref_points_xyz = _interface_points_xyz(
            reference_label_zyx,
            spacing_zyx=spacing_zyx,
            origin_zyx=origin_zyx,
        )
        mesh_points_xyz = _mesh_surface_points_xyz(mesh_vertices, mesh_faces)

        tau = (
            float(coverage_tau_nm)
            if coverage_tau_nm is not None
            else 0.75 * float(min(spacing_zyx))
        )
        mesh_chamfer, mesh_coverage = _surface_chamfer_and_coverage(
            ref_points_xyz,
            mesh_points_xyz,
            tau_nm=tau,
        )
        # backward-compatible approximation
        mesh_iou = mesh_coverage
        mesh_dice = (2.0 * mesh_coverage) / (1.0 + mesh_coverage)

        ref_bbox = _bbox_nm(ref_points_xyz)
        mesh_bbox = _bbox_nm(mesh_vertices.astype(np.float32, copy=False))
        if ref_bbox is not None and mesh_bbox is not None:
            ref_min, ref_max = ref_bbox
            mesh_min, mesh_max = mesh_bbox
            center_shift = float(
                np.linalg.norm(0.5 * (ref_min + ref_max) - 0.5 * (mesh_min + mesh_max))
            )
            size_l2 = float(np.linalg.norm((ref_max - ref_min) - (mesh_max - mesh_min)))

        ref_area = _surface_area_proxy(ref_points_xyz, spacing_zyx)
        mesh_area = _mesh_surface_area(mesh_vertices, mesh_faces)
        if ref_area > 0.0:
            surface_rel_error = abs(mesh_area - ref_area) / ref_area
        else:
            surface_rel_error = 0.0 if mesh_area == 0.0 else math.inf

    return StageMetrics(
        sdf_roundtrip_acc=roundtrip_acc,
        mesh_boundary_iou=mesh_iou,
        mesh_boundary_dice=mesh_dice,
        mesh_boundary_chamfer_nm=mesh_chamfer,
        mesh_boundary_coverage=mesh_coverage,
        bbox_center_shift_nm=center_shift,
        bbox_size_l2_nm=size_l2,
        surface_area_rel_error=surface_rel_error,
        render_diff_rate=render_diff,
    ).to_dict()


def diagnose_root_cause(
    metrics: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    def _as_float(value: object, default: float) -> float:
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    root_causes: list[str] = []
    triggered: list[str] = []

    sdf_roundtrip = _as_float(metrics.get("sdf_roundtrip_acc_mean"), 0.0)
    shell_iou = _as_float(metrics.get("material_shell_mesh_iou_mean"), 0.0)
    shell_chamfer = _as_float(metrics.get("material_shell_mesh_chamfer_nm_mean"), math.inf)
    interface_chamfer = _as_float(metrics.get("interface_mesh_chamfer_nm_mean"), math.inf)
    shell_coverage = _as_float(metrics.get("material_shell_mesh_coverage_mean"), 0.0)
    interface_coverage = _as_float(metrics.get("interface_mesh_coverage_mean"), 0.0)
    render_diff = _as_float(metrics.get("render_diff_rate_mean"), 0.0)
    policy_gap = _as_float(metrics.get("policy_gap_max"), 0.0)
    scope = str(metrics.get("policy_gap_scope_used", "global_max"))
    if scope == "real_vti":
        policy_gap = _as_float(metrics.get("policy_gap_real_vti"), policy_gap)
    elif scope == "global_max":
        policy_gap = _as_float(metrics.get("policy_gap_max"), policy_gap)
    else:
        policy_gap = _as_float(metrics.get("policy_gap_max"), policy_gap)
    strict_overall_pass = bool(metrics.get("strict_overall_pass", True))
    strict_fail_scenarios: list[str] = []
    raw_fail_counts = metrics.get("scenario_fail_counts")
    if isinstance(raw_fail_counts, Mapping):
        for key, value in raw_fail_counts.items():
            if isinstance(value, (int, float, np.floating)) and float(value) > 0.0:
                strict_fail_scenarios.append(str(key))

    sdf_min = float(thresholds["sdf_roundtrip_acc_min"])
    iou_min = float(thresholds["mesh_boundary_iou_min"])
    chamfer_max = float(thresholds["mesh_boundary_chamfer_nm_max"])
    coverage_min = float(thresholds["mesh_boundary_coverage_min"])
    render_max = float(thresholds["render_diff_rate_max"])
    policy_max = float(thresholds["policy_gap_max"])

    if sdf_roundtrip >= sdf_min and (
        shell_chamfer > chamfer_max or shell_coverage < coverage_min
    ):
        root_causes.append("mesh_extraction_or_face_annotation")
        triggered.append("Rule A")
    if (
        shell_chamfer <= chamfer_max
        and shell_coverage >= coverage_min
        and render_diff > render_max
    ):
        root_causes.append("rendering_domain_or_axis_mismatch")
        triggered.append("Rule B")
    if policy_gap > policy_max:
        root_causes.append("ingest_point_to_cell_policy")
        triggered.append("Rule C")
    if (
        shell_iou >= iou_min
        and shell_coverage >= coverage_min
        and (interface_coverage + 0.05 < shell_coverage or interface_chamfer > shell_chamfer * 1.1)
    ):
        root_causes.append("comparison_domain_mismatch_interface_vs_shell")
        triggered.append("Rule D")
    if not strict_overall_pass:
        root_causes.append("scenario_threshold_failures")
        triggered.append("Rule S")

    if not root_causes:
        root_causes.append("no_strong_signal")

    return {
        "root_cause_candidates": root_causes,
        "rules_triggered": triggered,
        "policy_gap_scope_used": scope,
        "policy_gap_used": policy_gap,
        "strict_fail_scenarios": sorted(strict_fail_scenarios),
    }
