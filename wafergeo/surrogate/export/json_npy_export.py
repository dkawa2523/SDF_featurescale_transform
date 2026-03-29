from __future__ import annotations

from pathlib import Path

import numpy as np

from wafergeo.core.types import Obs2D, PointCloud, TSDFVolume


def _save_array(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return str(path.as_posix())


def _rel(base_dir: Path, path: Path) -> str:
    return str(path.relative_to(base_dir).as_posix())


def pack_sample_json_npy(
    *,
    base_dir: Path,
    sample_id: str,
    tsdf: TSDFVolume | None,
    point_cloud: PointCloud | None,
    obs_targets: dict[str, Obs2D],
    include_sdf_features: dict[str, bool],
    include_obs2d_pack: bool,
) -> dict[str, str]:
    sample_root = base_dir / "packed" / sample_id
    packed_paths: dict[str, str] = {}

    if tsdf is not None:
        tsdf_path = sample_root / "sdf" / "tsdf.npy"
        _save_array(tsdf_path, tsdf.tsdf.astype(np.float32, copy=False))
        packed_paths["sdf.tsdf"] = _rel(base_dir, tsdf_path)

        if include_sdf_features.get("pair_code", False) and tsdf.pair_code is not None:
            pair_path = sample_root / "sdf" / "pair_code.npy"
            _save_array(pair_path, tsdf.pair_code.astype(np.uint8, copy=False))
            packed_paths["sdf.pair_code"] = _rel(base_dir, pair_path)

        if include_sdf_features.get("d_boundary", False) and tsdf.d_boundary is not None:
            boundary_path = sample_root / "sdf" / "d_boundary.npy"
            _save_array(boundary_path, tsdf.d_boundary.astype(np.float32, copy=False))
            packed_paths["sdf.d_boundary"] = _rel(base_dir, boundary_path)

        if include_sdf_features.get("present_mask", False) and tsdf.present_mask is not None:
            present_path = sample_root / "sdf" / "present_mask.npy"
            _save_array(present_path, tsdf.present_mask.astype(bool, copy=False))
            packed_paths["sdf.present_mask"] = _rel(base_dir, present_path)

    if point_cloud is not None:
        points_path = sample_root / "mesh" / "pc_points.npy"
        normals_path = sample_root / "mesh" / "pc_normals.npy"
        pair_path = sample_root / "mesh" / "pc_pair_code.npy"
        exposed_path = sample_root / "mesh" / "pc_is_exposed.npy"
        _save_array(points_path, point_cloud.points.astype(np.float32, copy=False))
        _save_array(normals_path, point_cloud.normals.astype(np.float32, copy=False))
        _save_array(pair_path, point_cloud.pair_code.astype(np.uint8, copy=False))
        _save_array(exposed_path, point_cloud.point_is_exposed.astype(bool, copy=False))
        packed_paths["mesh.pc_points"] = _rel(base_dir, points_path)
        packed_paths["mesh.pc_normals"] = _rel(base_dir, normals_path)
        packed_paths["mesh.pc_pair_code"] = _rel(base_dir, pair_path)
        packed_paths["mesh.pc_is_exposed"] = _rel(base_dir, exposed_path)

    if include_obs2d_pack:
        for observer_name, obs in obs_targets.items():
            mask_path = sample_root / "obs2d" / observer_name / "mask.npy"
            tsdf_path = sample_root / "obs2d" / observer_name / "tsdf.npy"
            _save_array(mask_path, obs.mask.astype(np.uint8, copy=False))
            _save_array(tsdf_path, obs.tsdf.astype(np.float32, copy=False))
            packed_paths[f"obs2d.{observer_name}.mask"] = _rel(base_dir, mask_path)
            packed_paths[f"obs2d.{observer_name}.tsdf"] = _rel(base_dir, tsdf_path)

    return packed_paths
