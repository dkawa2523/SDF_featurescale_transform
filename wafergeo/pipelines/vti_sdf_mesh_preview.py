from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

import wafergeo
from wafergeo.core.hashing import hash_config, sha256_file
from wafergeo.pipelines.vti_correspondence_audit import (
    STANDARD_VTI_PROFILE_ID,
    compute_standard_vti_bundle,
    get_standard_vti_profile,
    run_vti_correspondence_audit,
)

PREVIEW_SCHEMA_VERSION = "vti_preview/v2"


def _require_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "matplotlib is required for preview rendering. "
            "Install with: pip install -e '.[viz]'"
        ) from exc
    return plt


def _material_count_map(labels_zyx: np.ndarray) -> dict[str, int]:
    ids, counts = np.unique(labels_zyx, return_counts=True)
    return {str(int(mid)): int(cnt) for mid, cnt in zip(ids.tolist(), counts.tolist(), strict=True)}


def _plot_sdf_minabs_xyz_mid(tsdf: np.ndarray, output_path: Path) -> None:
    plt = _require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    min_abs = np.min(np.abs(tsdf.astype(np.float32, copy=False)), axis=0)
    zmid = min_abs.shape[0] // 2
    ymid = min_abs.shape[1] // 2
    xmid = min_abs.shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(min_abs[zmid], origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"min|tsdf| z-mid={zmid}")
    axes[1].imshow(min_abs[:, ymid, :], origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"min|tsdf| y-mid={ymid}")
    axes[2].imshow(min_abs[:, :, xmid], origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"min|tsdf| x-mid={xmid}")
    for ax in axes:
        ax.set_xlabel("X/Y")
        ax.set_ylabel("Z/Y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_sdf_channels_plane(
    tsdf: np.ndarray,
    material_ids: list[int],
    output_path: Path,
    *,
    plane: str,
) -> None:
    plt = _require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m = tsdf.shape[0]
    cols = min(4, m)
    rows = (m + cols - 1) // cols
    zmid = tsdf.shape[1] // 2
    ymid = tsdf.shape[2] // 2
    xmid = tsdf.shape[3] // 2

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
    for i, mid in enumerate(material_ids):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        if plane == "z":
            img = tsdf[i, zmid]
            title = f"mat={mid} z-mid"
        elif plane == "x":
            img = tsdf[i, :, :, xmid]
            title = f"mat={mid} x-mid"
        else:
            img = tsdf[i, :, ymid, :]
            title = f"mat={mid} y-mid"
        im = ax.imshow(img, origin="lower", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(m, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_single_vti_preview(
    vti_path: str | Path,
    output_dir: str | Path,
    *,
    outside_material_id: int = 2,
) -> dict[str, object]:
    vti_input = Path(vti_path)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)
    (out_root / "sdf").mkdir(parents=True, exist_ok=True)

    bundle = compute_standard_vti_bundle(vti_input, outside_material_id=outside_material_id)
    audit_manifest = run_vti_correspondence_audit(
        vti_path=vti_input,
        output_dir=out_root,
        outside_material_id=outside_material_id,
        _bundle=bundle,
    )

    labels_zyx = np.asarray(bundle.normalized_label)
    tsdf = np.asarray(bundle.tsdf_stack, dtype=np.float32)
    material_ids = [int(v) for v in bundle.selected_ids]
    source_array_name = bundle.source_array
    source_location = str(bundle.source_location)
    converted_from_point = bundle.converted_from_point
    flat_layout_used = bundle.flat_layout_used
    drift_rate = float(1.0 - bundle.point_to_cell_match)

    mu_nm = 20.0
    sdf_dir = out_root / "sdf"
    np.save(sdf_dir / "tsdf_full_stack.npy", tsdf)
    sdf_summary = {
        "shape": [int(v) for v in tsdf.shape],
        "dtype": str(tsdf.dtype),
        "mu_nm": mu_nm,
        "material_ids": material_ids,
        "tsdf_min": float(np.min(tsdf)),
        "tsdf_max": float(np.max(tsdf)),
        "nan_count": int(np.isnan(tsdf).sum()),
        "inf_count": int(np.isinf(tsdf).sum()),
    }
    (sdf_dir / "sdf_summary_full.json").write_text(
        json.dumps(sdf_summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    figures_dir = out_root / "figures"
    _plot_sdf_minabs_xyz_mid(tsdf, figures_dir / "sdf_minabs_xyz_mid.png")
    _plot_sdf_channels_plane(
        tsdf,
        material_ids,
        figures_dir / "sdf_channels_zmid_full.png",
        plane="z",
    )
    _plot_sdf_channels_plane(
        tsdf,
        material_ids,
        figures_dir / "sdf_channels_xmid_full.png",
        plane="x",
    )
    _plot_sdf_channels_plane(
        tsdf,
        material_ids,
        figures_dir / "sdf_channels_ymid_full.png",
        plane="y",
    )

    profile = get_standard_vti_profile()
    profile_hash = hash_config(profile)
    manifest = dict(audit_manifest)
    manifest.update(
        {
            "schema_version": PREVIEW_SCHEMA_VERSION,
            "profile_id": STANDARD_VTI_PROFILE_ID,
            "profile_hash": profile_hash,
            "input_path": str(vti_input),
            "input_hash": sha256_file(vti_input),
            "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "generator_version": wafergeo.__version__,
            "source_array_name": source_array_name,
            "source_location": source_location,
            "converted_from_point": converted_from_point,
            "point_to_cell_drift_rate_vs_point_like_cell": drift_rate,
            "outside_material_id": outside_material_id,
            "selected_material_ids": material_ids,
            "all_material_counts": _material_count_map(labels_zyx),
            "flat_layout_used": flat_layout_used,
            "point_to_cell_policy": str(profile["point_to_cell_policy"]),
            "material_policy": str(profile["material_policy"]),
            "mesh_mode": str(profile["mesh_mode"]),
            "mesh_backend_used": str(profile["mesh_backend"]),
            "audit_manifest_path": "audit_manifest.json",
            "postprocess": cast(dict[str, object], audit_manifest.get("postprocess", {})),
            "sdf": {
                "backend": "scipy",
                "shape": [int(v) for v in tsdf.shape],
                "mu_nm": mu_nm,
                "summary_json": "sdf/sdf_summary_full.json",
                "stack_npy": "sdf/tsdf_full_stack.npy",
            },
        }
    )
    manifest["outputs"] = {
        "figures": sorted(str(p.name) for p in figures_dir.glob("*.png")),
        "tables": sorted(str(p.name) for p in (out_root / "tables").glob("*.csv")),
        "sdf": [
            "sdf/sdf_summary_full.json",
            "sdf/tsdf_full_stack.npy",
        ],
    }

    (out_root / "preview_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return manifest
