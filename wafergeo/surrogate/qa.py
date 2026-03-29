from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from wafergeo.core.types import Obs2D, PointCloud, TSDFVolume
from wafergeo.surrogate.spec import DatasetQASpec


@dataclass(frozen=True)
class SampleQABundle:
    material_ids_present: list[int]
    missing_material_ids: list[int]
    interface_count: int
    obs_mask_area_px: dict[str, int]
    status: str
    messages: list[str]


def _as_list_obj(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    return []


def _as_dict_obj(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {}


def _as_int(value: object, default: int = 0) -> int:
    try:
        if not isinstance(value, (int, float, str, bytes, bytearray, np.integer, np.floating)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if not isinstance(value, (int, float, str, bytes, bytearray, np.integer, np.floating)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_sample_qa(
    *,
    tsdf: TSDFVolume | None,
    point_cloud: PointCloud | None,
    obs_targets: dict[str, Obs2D],
    expected_material_ids: list[int],
) -> dict[str, object]:
    messages: list[str] = []
    status = "OK"

    material_ids_present: list[int] = []
    missing_material_ids = list(expected_material_ids)
    interface_count = 0

    if tsdf is not None:
        if tsdf.present_mask is not None:
            material_ids_present = [
                mat_id
                for idx, mat_id in enumerate(expected_material_ids)
                if idx < tsdf.present_mask.shape[0] and bool(tsdf.present_mask[idx])
            ]
            missing_material_ids = [
                mat_id for mat_id in expected_material_ids if mat_id not in material_ids_present
            ]
        if tsdf.pair_code is not None:
            valid = tsdf.pair_code[tsdf.pair_code != 255]
            interface_count = int(valid.size)
        if not np.isfinite(tsdf.tsdf).all():
            status = "FAIL"
            messages.append("tsdf contains NaN/Inf")

    if point_cloud is not None:
        interface_count = max(interface_count, int(np.count_nonzero(point_cloud.pair_code != 255)))
        if point_cloud.points.shape[0] == 0:
            status = "FAIL"
            messages.append("pointcloud is empty")

    obs_mask_area_px: dict[str, int] = {}
    for observer_name, obs in obs_targets.items():
        area = int(np.count_nonzero(obs.mask))
        obs_mask_area_px[observer_name] = area
        if area == 0:
            status = "WARN" if status == "OK" else status
            messages.append(f"obs2d mask is empty for observer={observer_name}")

    bundle = SampleQABundle(
        material_ids_present=material_ids_present,
        missing_material_ids=missing_material_ids,
        interface_count=interface_count,
        obs_mask_area_px=obs_mask_area_px,
        status=status,
        messages=messages,
    )
    return {
        "material_ids_present": bundle.material_ids_present,
        "missing_material_ids": bundle.missing_material_ids,
        "interface_count": bundle.interface_count,
        "obs_mask_area_px": bundle.obs_mask_area_px,
        "status": bundle.status,
        "messages": bundle.messages,
    }


def compute_dataset_stats(
    records_qa: list[dict[str, object]],
    recipe_params_list: list[dict[str, object]],
) -> dict[str, object]:
    param_values: dict[str, list[float]] = {}
    for params in recipe_params_list:
        for key, value in params.items():
            if isinstance(value, (int, float)):
                param_values.setdefault(key, []).append(float(value))

    param_stats: dict[str, dict[str, float]] = {}
    for key, values in param_values.items():
        arr = np.asarray(values, dtype=np.float64)
        param_stats[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    missing_rates: list[float] = []
    interface_counts: list[int] = []
    status_counter: Counter[str] = Counter()
    obs_area_by_observer: dict[str, list[float]] = {}

    for qa in records_qa:
        missing_ids = _as_list_obj(qa.get("missing_material_ids", []))
        present_ids = _as_list_obj(qa.get("material_ids_present", []))
        total = len(missing_ids) + len(present_ids)
        missing_rates.append(0.0 if total == 0 else float(len(missing_ids)) / float(total))

        interface_counts.append(_as_int(qa.get("interface_count", 0), 0))
        status_counter[str(qa.get("status", "OK"))] += 1

        obs_area_map = _as_dict_obj(qa.get("obs_mask_area_px", {}))
        for observer_name, area in obs_area_map.items():
            obs_area_by_observer.setdefault(str(observer_name), []).append(_as_float(area, 0.0))

    obs_area_stats: dict[str, dict[str, float]] = {}
    for observer_name, values in obs_area_by_observer.items():
        arr = np.asarray(values, dtype=np.float64)
        obs_area_stats[observer_name] = {
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "num_samples": len(records_qa),
        "param_distribution": param_stats,
        "missing_material_rate_mean": float(np.mean(missing_rates)) if missing_rates else 0.0,
        "interface_frequency_mean": float(np.mean(interface_counts)) if interface_counts else 0.0,
        "status_counts": dict(status_counter),
        "obs_mask_area_stats": obs_area_stats,
    }


def compute_dataset_qa_summary(
    *,
    stats: dict[str, object],
    records_qa: list[dict[str, object]],
    qa_spec: DatasetQASpec,
) -> dict[str, object]:
    status = "OK"
    notes: list[str] = []

    material_counts: list[int] = []
    for qa in records_qa:
        present = _as_list_obj(qa.get("material_ids_present", []))
        missing = _as_list_obj(qa.get("missing_material_ids", []))
        material_counts.append(len(present) + len(missing))

    if material_counts and max(material_counts) > qa_spec.require_material_count_leq:
        status = "FAIL"
        notes.append(
            "material count exceeds require_material_count_leq "
            f"({qa_spec.require_material_count_leq})"
        )

    missing_rate = _as_float(stats.get("missing_material_rate_mean", 0.0), 0.0)
    if missing_rate > qa_spec.warn_missing_material_rate_gt:
        if status == "OK":
            status = "WARN"
        notes.append(
            "missing material rate is high: "
            f"{missing_rate:.4f} > {qa_spec.warn_missing_material_rate_gt:.4f}"
        )

    interface_counts = np.asarray(
        [_as_int(qa.get("interface_count", 0), 0) for qa in records_qa],
        dtype=np.float64,
    )
    if interface_counts.size > 0 and float(np.max(interface_counts)) > 0.0:
        imbalance = float(np.std(interface_counts) / (np.mean(interface_counts) + 1e-6))
        if imbalance > qa_spec.warn_interface_imbalance_gt:
            if status == "OK":
                status = "WARN"
            notes.append(
                "interface frequency imbalance is high: "
                f"{imbalance:.4f} > {qa_spec.warn_interface_imbalance_gt:.4f}"
            )

    return {
        "status": status,
        "notes": notes,
        "missing_material_rate_mean": missing_rate,
    }
