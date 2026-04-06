from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from wafergeo.core.hashing import hash_config, sha256_file
from wafergeo.io import vti_reader
from wafergeo._matplotlib import require_matplotlib_audit_plotting
from wafergeo.io.vti_reader import (
    ArrayLocation,
    RawVtiImage,
    extract_material_ids,
    infer_flat_array_layout,
    resolve_material_array_name,
)
from wafergeo.label.normalize import convert_point_labels_to_cell_zyx
from wafergeo.mesh.config import MeshBuildConfig
from wafergeo.mesh.extractors.vtk_interface import (
    VTKInterfaceExtractor,
    apply_vtk_visual_postprocess,
)
from wafergeo.sdf.audit import build_full_material_sdf

PlaneName = Literal["x_mid", "y_mid", "z_mid"]

# Backward-compatible aliases kept for tests and older call sites that monkeypatch
# reader functions on this module directly.
read_vti = vti_reader.read_vti
read_vti_with_xml_fallback = vti_reader.read_vti_with_xml_fallback


STANDARD_VTI_PROFILE_ID = "vti_standard_full_v1"
AUDIT_SCHEMA_VERSION = "vti_audit/v2"


@dataclass(frozen=True)
class StandardVTIBundle:
    profile: dict[str, object]
    raw: RawVtiImage
    input_hash: str
    read_backend_used: str
    source_array: str
    source_location: ArrayLocation
    converted_from_point: bool
    flat_layout_used: str
    point_to_cell_match: float
    selected_ids: list[int]
    normalized_label: np.ndarray
    converted_label: np.ndarray
    tsdf_stack: np.ndarray
    spacing_zyx: tuple[float, float, float]
    origin_zyx: tuple[float, float, float]
    raw_shells: dict[int, tuple[np.ndarray, np.ndarray]]
    converted_shells: dict[int, tuple[np.ndarray, np.ndarray]]
    bundle_messages: tuple[str, ...] = ()


def get_standard_vti_profile() -> dict[str, object]:
    return {
        "point_to_cell_policy": "nearest",
        "material_policy": "full",
        "mesh_mode": "material_shell",
        "mesh_backend": "vtk",
        "compare_planes": ("x_mid", "y_mid", "z_mid"),
        "vtk_viz_postprocess_enabled": True,
        "vtk_smoothing_iterations": 20,
        "vtk_smoothing_pass_band": 0.05,
        "vtk_subdivision_levels": 1,
        "vtk_boundary_smoothing": False,
        "vtk_feature_edge_smoothing": False,
        "qa_max_bbox_shift_nm": 2.0,
        "qa_max_area_rel_error": 0.15,
        "qa_postprocess_on_exceed": "warn",
    }


def _as_bool(value: object) -> bool:
    return bool(value)


def _as_int(value: object) -> int:
    return int(cast(int | float | str, value))


def _as_float(value: object) -> float:
    return float(cast(int | float | str, value))


def _as_str(value: object) -> str:
    return str(value)


def _read_vti_result(path: Path) -> vti_reader.VtiReadResult:
    # Preserve older monkeypatch patterns that replaced `read_vti` directly on this module.
    if read_vti is not vti_reader.read_vti:
        raw = read_vti(path)
        return vti_reader.VtiReadResult(raw=raw, backend_used="vtk")
    return vti_reader.read_vti_with_xml_fallback(path)


def _canonical_cell_grid(raw) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    sx, sy, sz = (float(v) for v in raw.spacing_xyz)
    ox, oy, oz = (float(v) for v in raw.origin_xyz)
    spacing_zyx = (sz, sy, sx)
    origin_zyx = (oz + 0.5 * sz, oy + 0.5 * sy, ox + 0.5 * sx)
    return spacing_zyx, origin_zyx


def _material_ids_from_policy(
    label_zyx: np.ndarray,
    *,
    outside_material_id: int,
) -> tuple[list[int], np.ndarray]:
    ids = sorted(int(v) for v in np.unique(label_zyx).tolist())
    if outside_material_id not in ids:
        ids = [outside_material_id, *ids]
    return ids, label_zyx.copy()


def _label_from_full_sdf(
    tsdf_stack: np.ndarray,
    material_ids: list[int],
    outside_id: int,
) -> np.ndarray:
    best_index = np.argmin(tsdf_stack, axis=0)
    channel_ids = np.asarray(material_ids, dtype=np.int32)
    labels = channel_ids[best_index].astype(np.int32, copy=False)
    outside_mask = np.all(tsdf_stack > 0.0, axis=0)
    labels[outside_mask] = int(outside_id)
    dtype = np.uint8 if max(material_ids) <= 255 else np.uint16
    return labels.astype(dtype, copy=False)


def _build_converted_label_and_tsdf(
    *,
    raw,
    label_cell_zyx: np.ndarray,
    selected_ids: list[int],
    outside_material_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    spacing_zyx, _ = _canonical_cell_grid(raw)
    tsdf = build_full_material_sdf(
        label_zyx=label_cell_zyx,
        material_ids=selected_ids,
        spacing_zyx=spacing_zyx,
        mu_nm=20.0,
    )
    converted = _label_from_full_sdf(tsdf, selected_ids, outside_material_id)
    return converted, tsdf


def _proxy_tsdf_from_labels(
    *,
    label_zyx: np.ndarray,
    material_ids: list[int],
) -> np.ndarray:
    tsdf = np.ones((len(material_ids),) + label_zyx.shape, dtype=np.float32)
    for channel, material_id in enumerate(material_ids):
        tsdf[channel, label_zyx == int(material_id)] = -1.0
    return tsdf


def _vtk_polys_to_triangles(polys, vtk_to_numpy) -> np.ndarray:
    class _NumpySupportShim:
        @staticmethod
        def vtk_to_numpy(arr):
            return vtk_to_numpy(arr)

    return VTKInterfaceExtractor._vtk_polys_to_triangles(polys, _NumpySupportShim)


def _extract_label_shells(
    *,
    label_zyx: np.ndarray,
    material_ids: list[int],
    spacing_zyx: tuple[float, float, float],
    origin_zyx: tuple[float, float, float],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    except Exception as exc:  # pragma: no cover - env dependent
        raise ImportError("vtk is required. Install with: pip install 'wafergeo[vtk]'") from exc

    sz, sy, sx = spacing_zyx
    oz, oy, ox = origin_zyx
    z_size, y_size, x_size = label_zyx.shape
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for mid in material_ids:
        mask_zyx = (label_zyx == int(mid)).astype(np.uint8)
        if int(mask_zyx.sum()) == 0:
            continue
        mask_xyz = np.transpose(mask_zyx, (2, 1, 0))
        image = vtk.vtkImageData()
        image.SetDimensions(x_size, y_size, z_size)
        image.SetSpacing(float(sx), float(sy), float(sz))
        image.SetOrigin(float(ox), float(oy), float(oz))
        vtk_arr = numpy_to_vtk(
            mask_xyz.ravel(order="F"),
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR,
        )
        vtk_arr.SetName("mask")
        image.GetPointData().SetScalars(vtk_arr)

        contour = vtk.vtkFlyingEdges3D()
        contour.SetInputData(image)
        contour.SetValue(0, 0.5)
        contour.Update()
        poly = contour.GetOutput()
        points = poly.GetPoints()
        polys = poly.GetPolys()
        if points is None or polys is None:
            continue
        vertices = vtk_to_numpy(points.GetData()).astype(np.float32, copy=False)
        faces = _vtk_polys_to_triangles(polys, vtk_to_numpy)
        if faces.size == 0:
            continue
        out[int(mid)] = (vertices, faces)
    return out


def _extract_label_shells_safe(
    *,
    label_zyx: np.ndarray,
    material_ids: list[int],
    spacing_zyx: tuple[float, float, float],
    origin_zyx: tuple[float, float, float],
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[str]]:
    try:
        return (
            _extract_label_shells(
                label_zyx=label_zyx,
                material_ids=material_ids,
                spacing_zyx=spacing_zyx,
                origin_zyx=origin_zyx,
            ),
            [],
        )
    except ImportError as exc:
        return (
            {},
            [f"shell extraction fallback: {exc}"],
        )
    except Exception as exc:
        return (
            {},
            [f"shell extraction warning: {exc}"],
        )


def _collect_interface_pairs(label_zyx: np.ndarray) -> dict[str, int]:
    pairs: dict[str, int] = {}
    for axis in range(3):
        lhs = np.take(label_zyx, indices=range(label_zyx.shape[axis] - 1), axis=axis)
        rhs = np.take(label_zyx, indices=range(1, label_zyx.shape[axis]), axis=axis)
        diff = lhs != rhs
        if not np.any(diff):
            continue
        a = lhs[diff].astype(np.int32, copy=False)
        b = rhs[diff].astype(np.int32, copy=False)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        for lv, hv in zip(lo.tolist(), hi.tolist(), strict=True):
            key = f"{int(lv)}-{int(hv)}"
            pairs[key] = pairs.get(key, 0) + 1
    return pairs


def _plane_slices(
    arr_zyx: np.ndarray,
    plane: PlaneName,
    *,
    zmid: int,
    ymid: int,
    xmid: int,
) -> np.ndarray:
    if plane == "z_mid":
        return arr_zyx[zmid]
    if plane == "x_mid":
        return arr_zyx[:, :, xmid]
    return arr_zyx[:, ymid, :]


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def _boundary_chamfer(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    ba = _binary_boundary(mask_a)
    bb = _binary_boundary(mask_b)
    if not np.any(ba) and not np.any(bb):
        return 0.0
    if not np.any(ba) or not np.any(bb):
        return float("inf")
    try:
        from scipy.ndimage import distance_transform_edt

        da = distance_transform_edt(~ba)
        db = distance_transform_edt(~bb)
        cd_ab = float(np.mean(db[ba]))
        cd_ba = float(np.mean(da[bb]))
        return 0.5 * (cd_ab + cd_ba)
    except Exception:  # pragma: no cover - env dependent
        a = np.argwhere(ba).astype(np.float64, copy=False)
        b = np.argwhere(bb).astype(np.float64, copy=False)
        d_ab = _nearest_neighbor_2d_numpy(a, b)
        d_ba = _nearest_neighbor_2d_numpy(b, a)
        return 0.5 * float(np.mean(d_ab) + np.mean(d_ba))


def _nearest_neighbor_2d_numpy(
    src_yx: np.ndarray,
    dst_yx: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    if src_yx.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if dst_yx.shape[0] == 0:
        return np.full((src_yx.shape[0],), np.inf, dtype=np.float64)
    out = np.empty((src_yx.shape[0],), dtype=np.float64)
    for start in range(0, src_yx.shape[0], chunk_size):
        stop = min(start + chunk_size, src_yx.shape[0])
        chunk = src_yx[start:stop]
        diff = chunk[:, None, :] - dst_yx[None, :, :]
        dist_sq = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
        out[start:stop] = np.sqrt(np.min(dist_sq, axis=1))
    return out


def _compute_slice_metrics(
    *,
    raw_2d: np.ndarray,
    conv_2d: np.ndarray,
    outside_id: int,
) -> dict[str, float]:
    raw_fg = raw_2d != int(outside_id)
    conv_fg = conv_2d != int(outside_id)
    inter = int(np.sum(raw_fg & conv_fg))
    union = int(np.sum(raw_fg | conv_fg))
    raw_sum = int(np.sum(raw_fg))
    conv_sum = int(np.sum(conv_fg))
    iou = float(inter / union) if union > 0 else 1.0
    dice = float((2 * inter) / (raw_sum + conv_sum)) if (raw_sum + conv_sum) > 0 else 1.0
    chamfer = _boundary_chamfer(raw_fg, conv_fg)
    drift = float(np.mean(raw_2d != conv_2d))
    return {
        "iou": iou,
        "dice": dice,
        "boundary_chamfer_px": chamfer,
        "label_drift_rate": drift,
    }


def _bbox_nm(
    mask_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    origin_zyx: tuple[float, float, float],
):
    idx = np.argwhere(mask_zyx)
    if idx.size == 0:
        return None
    mn = idx.min(axis=0)
    mx = idx.max(axis=0)
    z0 = float(origin_zyx[0] + mn[0] * spacing_zyx[0])
    y0 = float(origin_zyx[1] + mn[1] * spacing_zyx[1])
    x0 = float(origin_zyx[2] + mn[2] * spacing_zyx[2])
    z1 = float(origin_zyx[0] + mx[0] * spacing_zyx[0])
    y1 = float(origin_zyx[1] + mx[1] * spacing_zyx[1])
    x1 = float(origin_zyx[2] + mx[2] * spacing_zyx[2])
    return {"min_zyx_nm": [z0, y0, x0], "max_zyx_nm": [z1, y1, x1]}


def _safe_plot(plot_fn, messages: list[str]) -> str:
    try:
        plot_fn()
        return "OK"
    except Exception as exc:  # pragma: no cover - visualization resilience
        messages.append(f"plot warning: {exc}")
        return "WARN"


def _plot_shells_translucent(
    shells: dict[int, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    *,
    title: str,
    alpha: float,
) -> tuple[int, int]:
    plt, _, Poly3DCollection = require_matplotlib_audit_plotting(
        context="audit plots",
        install_hint="pip install -e '.[viz]'",
    )
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    ]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    faces_total = 0
    faces_plotted = 0
    for i, (_mid, (verts, faces)) in enumerate(sorted(shells.items())):
        if faces.shape[0] == 0:
            continue
        faces_total += int(faces.shape[0])
        f = faces
        faces_plotted += int(f.shape[0])
        tri = verts[f]
        coll = Poly3DCollection(
            tri,
            facecolors=palette[i % len(palette)],
            edgecolors="none",
            alpha=alpha,
        )
        ax.add_collection3d(coll)
    if shells:
        all_verts = np.concatenate([v for v, _ in shells.values()], axis=0)
        ax.auto_scale_xyz(all_verts[:, 0], all_verts[:, 1], all_verts[:, 2])
    ax.set_title(title)
    ax.set_xlabel("X [nm]")
    ax.set_ylabel("Y [nm]")
    ax.set_zlabel("Z [nm]")
    ax.view_init(elev=22, azim=40)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return faces_total, faces_plotted


def _plot_overlay_shells(
    raw_shells: dict[int, tuple[np.ndarray, np.ndarray]],
    conv_shells: dict[int, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> tuple[int, int]:
    plt, _, Poly3DCollection = require_matplotlib_audit_plotting(
        context="audit plots",
        install_hint="pip install -e '.[viz]'",
    )
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    faces_total = 0
    faces_plotted = 0
    for shells, color, alpha in ((raw_shells, "#1f77b4", 0.10), (conv_shells, "#d62728", 0.10)):
        for verts, faces in shells.values():
            if faces.shape[0] == 0:
                continue
            faces_total += int(faces.shape[0])
            f = faces
            faces_plotted += int(f.shape[0])
            tri = verts[f]
            ax.add_collection3d(
                Poly3DCollection(
                    tri,
                    facecolors=color,
                    edgecolors="none",
                    alpha=alpha,
                )
            )
    verts_all = []
    for shells in (raw_shells, conv_shells):
        verts_all.extend([v for v, _ in shells.values()])
    if verts_all:
        merged = np.concatenate(verts_all, axis=0)
        ax.auto_scale_xyz(merged[:, 0], merged[:, 1], merged[:, 2])
    ax.set_title("Raw(blue) vs Converted(red) Shell Overlay")
    ax.set_xlabel("X [nm]")
    ax.set_ylabel("Y [nm]")
    ax.set_zlabel("Z [nm]")
    ax.view_init(elev=22, azim=40)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return faces_total, faces_plotted


def _postprocess_shells(
    shells: dict[int, tuple[np.ndarray, np.ndarray]],
    cfg: MeshBuildConfig,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, float], str, list[str]]:
    processed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    max_bbox = 0.0
    max_area = 0.0
    total_pre_faces = 0
    total_post_faces = 0
    messages: list[str] = []
    status = "OK"

    for mid, (verts, faces) in sorted(shells.items()):
        try:
            v2, f2, metrics = apply_vtk_visual_postprocess(verts, faces, cfg)
        except Exception as exc:  # pragma: no cover - env dependent
            v2, f2 = verts, faces
            metrics = {
                "bbox_shift_nm": 0.0,
                "area_rel_error": 0.0,
                "pre_faces": float(faces.shape[0]),
                "post_faces": float(faces.shape[0]),
            }
            status = "WARN"
            messages.append(f"postprocess fallback for material {mid}: {exc}")
        processed[int(mid)] = (v2, f2)
        bbox_shift = float(metrics.get("bbox_shift_nm", 0.0))
        area_error = float(metrics.get("area_rel_error", 0.0))
        max_bbox = max(max_bbox, bbox_shift)
        max_area = max(max_area, area_error)
        total_pre_faces += int(metrics.get("pre_faces", float(faces.shape[0])))
        total_post_faces += int(metrics.get("post_faces", float(f2.shape[0])))

    exceeds = (
        max_bbox > float(cfg.qa_max_bbox_shift_nm)
        or max_area > float(cfg.qa_max_area_rel_error)
    )
    if exceeds:
        status = "WARN"
        messages.append(
            "vtk postprocess exceeds thresholds: "
            f"bbox_shift_nm={max_bbox:.6f} (max={cfg.qa_max_bbox_shift_nm}), "
            f"area_rel_error={max_area:.6f} (max={cfg.qa_max_area_rel_error})"
        )
        if cfg.qa_postprocess_on_exceed == "fail":
            raise ValueError(messages[-1])

    summary = {
        "bbox_shift_nm": max_bbox,
        "area_rel_error": max_area,
        "pre_faces": float(total_pre_faces),
        "post_faces": float(total_post_faces),
    }
    return processed, summary, status, messages


def _plot_slice_compare(
    *,
    raw_2d: np.ndarray,
    conv_2d: np.ndarray,
    plane: PlaneName,
    output_path: Path,
) -> None:
    plt, _, _ = require_matplotlib_audit_plotting(
        context="audit plots",
        install_hint="pip install -e '.[viz]'",
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_2d, origin="lower", cmap="tab20")
    axes[0].set_title(f"Raw {plane}")
    axes[1].imshow(conv_2d, origin="lower", cmap="tab20")
    axes[1].set_title(f"Converted {plane}")
    diff = raw_2d != conv_2d
    axes[2].imshow(diff.astype(np.uint8), origin="lower", cmap="gray")
    axes[2].set_title(f"Diff {plane}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_boundary_overlay(
    *,
    raw_2d: np.ndarray,
    conv_2d: np.ndarray,
    plane: PlaneName,
    output_path: Path,
) -> None:
    plt, _, _ = require_matplotlib_audit_plotting(
        context="audit plots",
        install_hint="pip install -e '.[viz]'",
    )
    br = _binary_boundary(raw_2d)
    bc = _binary_boundary(conv_2d)
    img = np.zeros(br.shape + (3,), dtype=np.float32)
    img[..., 2] = br.astype(np.float32)  # blue raw
    img[..., 0] = bc.astype(np.float32)  # red conv
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, origin="lower")
    ax.set_title(f"Boundary overlay {plane} (red=converted, blue=raw)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _write_material_table(
    path: Path,
    *,
    material_ids: list[int],
    raw_label: np.ndarray,
    conv_label: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["material_id", "raw_count", "converted_count", "delta_count", "delta_rate"]
        )
        total = float(raw_label.size)
        for mid in material_ids:
            raw_count = int(np.sum(raw_label == int(mid)))
            conv_count = int(np.sum(conv_label == int(mid)))
            delta = conv_count - raw_count
            writer.writerow(
                [
                    int(mid),
                    raw_count,
                    conv_count,
                    delta,
                    float(delta / total),
                ]
            )


def _write_slice_metrics(path: Path, rows: dict[str, dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plane", "iou", "dice", "boundary_chamfer_px", "label_drift_rate"])
        for plane_name in ("x_mid", "y_mid", "z_mid"):
            if plane_name not in rows:
                continue
            row = rows[plane_name]
            writer.writerow(
                [
                    plane_name,
                    row["iou"],
                    row["dice"],
                    row["boundary_chamfer_px"],
                    row["label_drift_rate"],
                ]
            )


def compute_standard_vti_bundle(
    vti_path: Path,
    *,
    outside_material_id: int,
) -> StandardVTIBundle:
    profile = get_standard_vti_profile()
    point_to_cell_policy = str(profile["point_to_cell_policy"])

    read_result = _read_vti_result(vti_path)
    raw = read_result.raw
    input_hash = sha256_file(vti_path)
    flat_layout_used = infer_flat_array_layout(raw)
    source_array = resolve_material_array_name(raw)
    point_zyx, source_location = extract_material_ids(
        raw,
        source_array,
        flat_layout=flat_layout_used,
    )

    if source_location == "point":
        raw_cell = convert_point_labels_to_cell_zyx(
            point_zyx,
            policy=point_to_cell_policy,  # type: ignore[arg-type]
            majority_tie_breaker="smallest",
        )
        converted_from_point = True
    else:
        raw_cell = point_zyx
        converted_from_point = False
    zc, yc, xc = raw_cell.shape
    point_like_cell = point_zyx[:zc, :yc, :xc]
    point_to_cell_match = float(np.mean(raw_cell == point_like_cell))

    selected_ids, normalized_label = _material_ids_from_policy(
        raw_cell,
        outside_material_id=outside_material_id,
    )
    bundle_messages: list[str] = list(read_result.messages)
    try:
        converted_label, tsdf_stack = _build_converted_label_and_tsdf(
            raw=raw,
            label_cell_zyx=normalized_label,
            selected_ids=selected_ids,
            outside_material_id=outside_material_id,
        )
    except ImportError as exc:
        bundle_messages.append(f"sdf fallback: {exc}")
        tsdf_stack = _proxy_tsdf_from_labels(label_zyx=normalized_label, material_ids=selected_ids)
        converted_label = _label_from_full_sdf(tsdf_stack, selected_ids, outside_material_id)
    spacing_zyx, origin_zyx = _canonical_cell_grid(raw)
    raw_shells, raw_shell_messages = _extract_label_shells_safe(
        label_zyx=normalized_label,
        material_ids=selected_ids,
        spacing_zyx=spacing_zyx,
        origin_zyx=origin_zyx,
    )
    converted_shells, converted_shell_messages = _extract_label_shells_safe(
        label_zyx=converted_label,
        material_ids=selected_ids,
        spacing_zyx=spacing_zyx,
        origin_zyx=origin_zyx,
    )
    bundle_messages.extend(raw_shell_messages)
    bundle_messages.extend(converted_shell_messages)

    return StandardVTIBundle(
        profile=profile,
        raw=raw,
        input_hash=input_hash,
        read_backend_used=read_result.backend_used,
        source_array=source_array,
        source_location=source_location,
        converted_from_point=converted_from_point,
        flat_layout_used=flat_layout_used,
        point_to_cell_match=point_to_cell_match,
        selected_ids=selected_ids,
        normalized_label=normalized_label,
        converted_label=converted_label,
        tsdf_stack=tsdf_stack,
        spacing_zyx=spacing_zyx,
        origin_zyx=origin_zyx,
        raw_shells=raw_shells,
        converted_shells=converted_shells,
        bundle_messages=tuple(bundle_messages),
    )


def run_vti_correspondence_audit(
    vti_path: str | Path,
    output_dir: str | Path,
    *,
    outside_material_id: int = 2,
    _bundle: StandardVTIBundle | None = None,
) -> dict[str, object]:
    path = Path(vti_path)
    out = Path(output_dir)
    figs = out / "figures"
    tables = out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    messages: list[str] = []
    status = "OK"

    bundle = _bundle or compute_standard_vti_bundle(path, outside_material_id=outside_material_id)
    if bundle.bundle_messages:
        messages.extend(list(bundle.bundle_messages))
        status = "WARN"
    profile = dict(bundle.profile)
    point_to_cell_policy = str(profile["point_to_cell_policy"])
    material_policy = str(profile["material_policy"])
    mesh_mode = str(profile["mesh_mode"])
    mesh_backend = str(profile["mesh_backend"])
    compare_planes = cast(tuple[PlaneName, ...], profile["compare_planes"])
    profile_hash = hash_config(profile)

    input_hash = bundle.input_hash
    read_backend_used = bundle.read_backend_used
    source_array = bundle.source_array
    converted_from_point = bundle.converted_from_point
    flat_layout_used = bundle.flat_layout_used
    point_to_cell_match = bundle.point_to_cell_match
    selected_ids = bundle.selected_ids
    normalized_label = bundle.normalized_label
    converted_label = bundle.converted_label
    tsdf_stack = bundle.tsdf_stack
    spacing_zyx = bundle.spacing_zyx
    origin_zyx = bundle.origin_zyx
    raw_shells = bundle.raw_shells
    converted_shells_pre = bundle.converted_shells

    post_cfg = MeshBuildConfig(
        mode=mesh_mode,  # type: ignore[arg-type]
        backend=mesh_backend,  # type: ignore[arg-type]
        vtk_viz_postprocess_enabled=_as_bool(profile["vtk_viz_postprocess_enabled"]),
        vtk_smoothing_iterations=_as_int(profile["vtk_smoothing_iterations"]),
        vtk_smoothing_pass_band=_as_float(profile["vtk_smoothing_pass_band"]),
        vtk_subdivision_levels=_as_int(profile["vtk_subdivision_levels"]),
        vtk_boundary_smoothing=_as_bool(profile["vtk_boundary_smoothing"]),
        vtk_feature_edge_smoothing=_as_bool(profile["vtk_feature_edge_smoothing"]),
        qa_max_bbox_shift_nm=_as_float(profile["qa_max_bbox_shift_nm"]),
        qa_max_area_rel_error=_as_float(profile["qa_max_area_rel_error"]),
        qa_postprocess_on_exceed=_as_str(profile["qa_postprocess_on_exceed"]),  # type: ignore[arg-type]
    )
    converted_shells_post, post_metrics, post_status, post_messages = _postprocess_shells(
        converted_shells_pre,
        post_cfg,
    )
    messages.extend(post_messages)
    if post_status == "WARN":
        status = "WARN"
    converted_shells = converted_shells_post

    zmid = normalized_label.shape[0] // 2
    ymid = normalized_label.shape[1] // 2
    xmid = normalized_label.shape[2] // 2

    slice_metrics: dict[str, dict[str, float]] = {}
    for plane in cast(tuple[PlaneName, ...], compare_planes):
        raw_2d = _plane_slices(normalized_label, plane, zmid=zmid, ymid=ymid, xmid=xmid)
        conv_2d = _plane_slices(converted_label, plane, zmid=zmid, ymid=ymid, xmid=xmid)
        slice_metrics[plane] = _compute_slice_metrics(
            raw_2d=raw_2d,
            conv_2d=conv_2d,
            outside_id=outside_material_id,
        )

    raw_faces_total = sum(int(f.shape[0]) for _, f in raw_shells.values())
    conv_faces_total_pre = sum(int(f.shape[0]) for _, f in converted_shells_pre.values())
    conv_faces_total_post = sum(int(f.shape[0]) for _, f in converted_shells_post.values())

    raw_status = _safe_plot(
        lambda: _plot_shells_translucent(
            raw_shells,
            figs / "3d_raw_shell_translucent.png",
            title="Raw Cell Shell (Translucent)",
            alpha=0.14,
        ),
        messages,
    )
    conv_status = _safe_plot(
        lambda: _plot_shells_translucent(
            converted_shells_pre,
            figs / "3d_converted_shell_translucent_pre.png",
            title="Converted Shell (Pre-Postprocess, Translucent)",
            alpha=0.14,
        ),
        messages,
    )
    conv_post_status = _safe_plot(
        lambda: _plot_shells_translucent(
            converted_shells_post,
            figs / "3d_converted_shell_translucent_post.png",
            title="Converted Shell (Post-Postprocess, Translucent)",
            alpha=0.14,
        ),
        messages,
    )
    conv_compat_status = _safe_plot(
        lambda: _plot_shells_translucent(
            converted_shells_post,
            figs / "3d_converted_shell_translucent.png",
            title="Converted Shell (Translucent)",
            alpha=0.14,
        ),
        messages,
    )
    prepost_status = _safe_plot(
        lambda: _plot_overlay_shells(
            converted_shells_pre,
            converted_shells_post,
            figs / "3d_converted_shell_overlay_pre_post.png",
        ),
        messages,
    )
    ov_status = _safe_plot(
        lambda: _plot_overlay_shells(
            raw_shells,
            converted_shells,
            figs / "3d_overlay_shell_translucent.png",
        ),
        messages,
    )
    if "WARN" in (
        status,
        raw_status,
        conv_status,
        conv_post_status,
        conv_compat_status,
        prepost_status,
        ov_status,
    ):
        status = "WARN"

    for plane in cast(tuple[PlaneName, ...], compare_planes):
        raw_2d = _plane_slices(normalized_label, plane, zmid=zmid, ymid=ymid, xmid=xmid)
        conv_2d = _plane_slices(converted_label, plane, zmid=zmid, ymid=ymid, xmid=xmid)
        p1 = _safe_plot(
            lambda r=raw_2d, c=conv_2d, pl=plane: _plot_slice_compare(
                raw_2d=r,
                conv_2d=c,
                plane=pl,
                output_path=figs / f"slice_{pl}_raw_vs_conv.png",
            ),
            messages,
        )
        p2 = _safe_plot(
            lambda r=raw_2d, c=conv_2d, pl=plane: _plot_boundary_overlay(
                raw_2d=r,
                conv_2d=c,
                plane=pl,
                output_path=figs / f"slice_{pl}_boundary_overlay.png",
            ),
            messages,
        )
        if "WARN" in (p1, p2):
            status = "WARN"

    _write_material_table(
        tables / "material_volume_compare.csv",
        material_ids=selected_ids,
        raw_label=normalized_label,
        conv_label=converted_label,
    )
    _write_slice_metrics(tables / "slice_metrics.csv", rows=slice_metrics)

    raw_pairs = _collect_interface_pairs(normalized_label)
    conv_pairs = _collect_interface_pairs(converted_label)
    raw_pair_set = set(raw_pairs)
    conv_pair_set = set(conv_pairs)
    pair_inter = len(raw_pair_set & conv_pair_set)
    pair_union = len(raw_pair_set | conv_pair_set)
    pair_jaccard = float(pair_inter / pair_union) if pair_union > 0 else 1.0

    raw_fg = normalized_label != int(outside_material_id)
    conv_fg = converted_label != int(outside_material_id)
    raw_bbox = _bbox_nm(raw_fg, spacing_zyx, origin_zyx)
    conv_bbox = _bbox_nm(conv_fg, spacing_zyx, origin_zyx)

    material_ids_raw = sorted(int(v) for v in np.unique(normalized_label).tolist())
    material_ids_conv = sorted(int(v) for v in np.unique(converted_label).tolist())

    global_metrics = {
        "mean_iou": float(np.mean([slice_metrics[p]["iou"] for p in slice_metrics])),
        "mean_dice": float(np.mean([slice_metrics[p]["dice"] for p in slice_metrics])),
        "mean_boundary_chamfer_px": float(
            np.mean([slice_metrics[p]["boundary_chamfer_px"] for p in slice_metrics])
        ),
        "pair_jaccard": pair_jaccard,
        "point_to_cell_match_vs_point_like_cell": point_to_cell_match,
    }
    if raw_bbox is not None and conv_bbox is not None:
        raw_min = np.asarray(raw_bbox["min_zyx_nm"], dtype=np.float64)
        raw_max = np.asarray(raw_bbox["max_zyx_nm"], dtype=np.float64)
        conv_min = np.asarray(conv_bbox["min_zyx_nm"], dtype=np.float64)
        conv_max = np.asarray(conv_bbox["max_zyx_nm"], dtype=np.float64)
        global_metrics["bbox_center_shift_nm"] = float(
            np.linalg.norm(0.5 * (raw_min + raw_max) - 0.5 * (conv_min + conv_max))
        )
        global_metrics["bbox_size_l2_nm"] = float(
            np.linalg.norm((raw_max - raw_min) - (conv_max - conv_min))
        )

    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "profile_id": STANDARD_VTI_PROFILE_ID,
        "profile_hash": profile_hash,
        "status": status,
        "messages": messages,
        "input_path": str(path),
        "input_hash": input_hash,
        "source_array_name": source_array,
        "converted_from_point": converted_from_point,
        "point_to_cell_policy": point_to_cell_policy,
        "material_policy": material_policy,
        "mesh_mode": mesh_mode,
        "mesh_backend": mesh_backend,
        "mesh_backend_used": mesh_backend,
        "read_backend_used": read_backend_used,
        "flat_layout_used": flat_layout_used,
        "material_ids_raw": material_ids_raw,
        "material_ids_converted": material_ids_conv,
        "selected_material_ids": selected_ids,
        "mid_indices_zyx": {"z_mid": zmid, "y_mid": ymid, "x_mid": xmid},
        "metrics": {
            "slice": slice_metrics,
            "global": global_metrics,
            "pair_counts": {
                "raw": raw_pairs,
                "converted": conv_pairs,
            },
        },
        "raw_bbox_nm": raw_bbox,
        "converted_bbox_nm": conv_bbox,
        "tsdf_shape": [int(v) for v in tsdf_stack.shape],
        "mesh_faces_total": {
            "raw_shells": raw_faces_total,
            "converted_shells_pre": conv_faces_total_pre,
            "converted_shells_post": conv_faces_total_post,
            "converted_shells": conv_faces_total_post,
        },
        "mesh_faces_plotted": {
            "raw_shells": raw_faces_total,
            "converted_shells_pre": conv_faces_total_pre,
            "converted_shells_post": conv_faces_total_post,
            "converted_shells": conv_faces_total_post,
        },
        "postprocess": {
            "enabled": _as_bool(profile["vtk_viz_postprocess_enabled"]),
            "params": {
                "vtk_smoothing_iterations": _as_int(profile["vtk_smoothing_iterations"]),
                "vtk_smoothing_pass_band": _as_float(profile["vtk_smoothing_pass_band"]),
                "vtk_subdivision_levels": _as_int(profile["vtk_subdivision_levels"]),
                "vtk_boundary_smoothing": _as_bool(profile["vtk_boundary_smoothing"]),
                "vtk_feature_edge_smoothing": _as_bool(profile["vtk_feature_edge_smoothing"]),
                "qa_max_bbox_shift_nm": _as_float(profile["qa_max_bbox_shift_nm"]),
                "qa_max_area_rel_error": _as_float(profile["qa_max_area_rel_error"]),
                "qa_postprocess_on_exceed": _as_str(profile["qa_postprocess_on_exceed"]),
            },
            "metrics": post_metrics,
            "status": post_status,
        },
        "outputs": {
            "figures": sorted(str(p.name) for p in figs.glob("*.png")),
            "tables": sorted(str(p.name) for p in tables.glob("*.csv")),
        },
    }
    (out / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return manifest
