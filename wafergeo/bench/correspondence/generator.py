from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wafergeo.core.grid import GridSpec
from wafergeo.core.hashing import hash_config, sha256_file
from wafergeo.core.meta import Meta
from wafergeo.core.types import LabelVolume, MaterialSpec
from wafergeo.io.vti_reader import (
    RawVtiImage,
    extract_material_ids,
    infer_flat_array_layout,
    read_vti,
    read_vti_materialids_xml_fallback,
    resolve_material_array_name,
)
from wafergeo.label.normalize import convert_point_labels_to_cell_zyx


@dataclass(frozen=True)
class BenchmarkScenarioData:
    name: str
    input_data: RawVtiImage | LabelVolume
    outside_material_id: int
    expected: dict[str, float]
    input_hash: str


def _make_raw_from_point_labels(
    point_zyx: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RawVtiImage:
    point_xyz = np.transpose(point_zyx.astype(np.int32, copy=False), (2, 1, 0))
    nx, ny, nz = point_xyz.shape
    return RawVtiImage(
        spacing_xyz=spacing_xyz,
        origin_xyz=origin_xyz,
        dims_xyz=(nx, ny, nz),
        arrays={"MaterialIds": point_xyz.ravel(order="C")},
        array_location={"MaterialIds": "point"},
        vtk_meta={"source": "synthetic"},
    )


def _cube_point() -> np.ndarray:
    arr = np.full((17, 17, 17), 2, dtype=np.int32)
    arr[4:13, 4:13, 4:13] = 5
    return arr


def _layers_point() -> np.ndarray:
    arr = np.full((15, 15, 15), 2, dtype=np.int32)
    arr[4:8, :, :] = 3
    arr[8:11, :, :] = 4
    arr[11:13, :, :] = 5
    return arr


def _thin_shell_point() -> np.ndarray:
    arr = np.full((19, 19, 19), 2, dtype=np.int32)
    arr[3:16, 3:16, 3:16] = 3
    arr[4:15, 4:15, 4:15] = 2
    return arr


def _diagonal_point() -> np.ndarray:
    z_size, y_size, x_size = (17, 17, 17)
    z_idx, y_idx, x_idx = np.indices((z_size, y_size, x_size), dtype=np.int32)
    arr = np.full((z_size, y_size, x_size), 2, dtype=np.int32)
    arr[(x_idx + y_idx) > z_idx + 10] = 6
    arr[(x_idx > 9) & (y_idx < 7)] = 7
    return arr


def _make_label_volume(
    label_zyx: np.ndarray,
    *,
    outside_id: int,
) -> LabelVolume:
    uniq = sorted(int(v) for v in np.unique(label_zyx).tolist())
    if outside_id not in uniq:
        uniq = [outside_id, *uniq]
    names = [f"m_{mid}" for mid in uniq]
    priority = [0 if mid == outside_id else 1000 - i for i, mid in enumerate(uniq)]
    ignore = [mid == outside_id for mid in uniq]
    material = MaterialSpec(
        ids=uniq,
        names=names,
        void_id=outside_id,
        priority=priority,
        ignore_in_exposure=ignore,
    )
    grid = GridSpec(
        dim=3,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.5, 0.5, 0.5),
        axis_order="ZYX",
        sample_location="cell_center",
        units="nm",
    )
    meta = Meta(
        schema_version="label/v1",
        profile_id="bench_label_v1",
        config_hash="bench",
        generator_version="0.1.0",
        git_commit="bench",
        input_hash="bench",
        created_at="1970-01-01T00:00:00+00:00",
        extra={"scenario": "synthetic"},
    )
    dtype = np.uint8 if max(uniq) <= 255 else np.uint16
    return LabelVolume(
        grid=grid,
        material=material,
        material_id=label_zyx.astype(dtype, copy=False),
        meta=meta,
    )


def generate_benchmark_case(
    name: str,
    *,
    real_vti_path: str | None = None,
) -> RawVtiImage | LabelVolume:
    if name == "cube":
        return _make_raw_from_point_labels(_cube_point())
    if name == "layers3":
        return _make_raw_from_point_labels(_layers_point())
    if name == "thin_shell":
        return _make_raw_from_point_labels(_thin_shell_point())
    if name == "diagonal":
        return _make_raw_from_point_labels(_diagonal_point())
    if name == "real_vti":
        if not real_vti_path:
            raise ValueError("real_vti_path is required for scenario 'real_vti'")
        path = Path(real_vti_path)
        try:
            return read_vti(path)
        except ImportError:
            return read_vti_materialids_xml_fallback(path)
    raise ValueError(f"unknown benchmark scenario: {name}")


def load_benchmark_scenario(
    name: str,
    *,
    real_vti_path: str | None = None,
) -> BenchmarkScenarioData:
    data = generate_benchmark_case(name, real_vti_path=real_vti_path)
    outside_id = 2
    expected = {
        "sdf_roundtrip_acc_min": 0.995,
        "mesh_boundary_iou_min": 0.70,
    }
    if name == "cube":
        expected = {"sdf_roundtrip_acc_min": 0.999, "mesh_boundary_iou_min": 0.90}
    elif name == "layers3":
        expected = {"sdf_roundtrip_acc_min": 0.999, "mesh_boundary_iou_min": 0.85}
    elif name == "thin_shell":
        expected = {"sdf_roundtrip_acc_min": 0.995, "mesh_boundary_iou_min": 0.60}
    elif name == "diagonal":
        expected = {"sdf_roundtrip_acc_min": 0.990, "mesh_boundary_iou_min": 0.55}

    if isinstance(data, LabelVolume):
        input_hash = data.meta.input_hash
    elif name == "real_vti":
        assert real_vti_path is not None
        input_hash = sha256_file(Path(real_vti_path))
    else:
        input_hash = hash_config({"scenario": name, "shape": list(data.dims_xyz)})

    return BenchmarkScenarioData(
        name=name,
        input_data=data,
        outside_material_id=outside_id,
        expected=expected,
        input_hash=input_hash,
    )


def as_label_volume_for_policy(
    scenario: BenchmarkScenarioData,
    *,
    point_to_cell_policy: str,
    max_materials: int = 5,
) -> tuple[LabelVolume, dict[str, float | int | str]]:
    data = scenario.input_data
    if isinstance(data, LabelVolume):
        return data, {
            "point_to_cell_match": 1.0,
            "material_count_source": len(data.material.ids),
        }

    array_name = resolve_material_array_name(data)
    flat_layout_used = infer_flat_array_layout(data)
    point_zyx, _ = extract_material_ids(data, array_name, flat_layout=flat_layout_used)
    cell_zyx = convert_point_labels_to_cell_zyx(
        point_zyx,
        policy=point_to_cell_policy,  # type: ignore[arg-type]
        majority_tie_breaker="smallest",
    )
    zc, yc, xc = cell_zyx.shape
    point_like = point_zyx[:zc, :yc, :xc]
    point_match = float(np.mean(cell_zyx == point_like))

    uniq = np.unique(cell_zyx)
    if uniq.size > max_materials:
        selected_ids, _ = _select_material_ids(
            cell_zyx,
            outside_material_id=scenario.outside_material_id,
            max_materials=max_materials,
        )
    else:
        selected_ids = sorted(int(v) for v in uniq.tolist())
        if scenario.outside_material_id not in selected_ids:
            selected_ids = [scenario.outside_material_id, *selected_ids]

    out = cell_zyx.copy()
    keep = np.asarray(selected_ids, dtype=np.int64)
    out[~np.isin(out, keep)] = scenario.outside_material_id

    label = _make_label_volume(out, outside_id=scenario.outside_material_id)
    return label, {
        "point_to_cell_match": point_match,
        "material_count_source": int(uniq.size),
        "flat_layout_used": flat_layout_used,
    }


def _select_material_ids(
    material_id_zyx: np.ndarray,
    *,
    outside_material_id: int,
    max_materials: int,
) -> tuple[list[int], dict[int, int]]:
    all_ids, all_counts = np.unique(material_id_zyx, return_counts=True)
    counts = {
        int(mid): int(cnt)
        for mid, cnt in zip(all_ids.tolist(), all_counts.tolist(), strict=True)
    }
    non_outside = [mid for mid in counts if mid != outside_material_id]
    non_outside.sort(key=lambda mid: (-counts[mid], mid))
    selected = [outside_material_id]
    selected.extend(non_outside[: max(0, max_materials - 1)])
    selected = selected[:max_materials]
    if len(selected) <= 1:
        raise ValueError(
            "material selection produced only outside material. "
            "Input seems invalid for interface extraction."
        )
    return selected, counts
