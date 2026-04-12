from __future__ import annotations

import numpy as np

from wafergeo.compare.features import ViewFeature
from wafergeo.compare.metric_types import MetricComputation, MetricContext


def _component_count(mask: np.ndarray) -> int:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"topology mask must be 2D [Y,X], got ndim={binary.ndim}")
    seen = np.zeros(binary.shape, dtype=bool)
    count = 0
    height, width = binary.shape
    for start_y, start_x in zip(*np.nonzero(binary & ~seen), strict=False):
        if seen[start_y, start_x]:
            continue
        count += 1
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        while stack:
            y, x = stack.pop()
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                if not binary[yy, xx] or seen[yy, xx]:
                    continue
                seen[yy, xx] = True
                stack.append((yy, xx))
    return count


def _label_component_rows(sim: ViewFeature, target: ViewFeature) -> list[dict[str, int]]:
    if sim.source != "label_volume" or target.source != "label_volume":
        return []
    sim_labels = np.asarray(sim.label2d)
    target_labels = np.asarray(target.label2d)
    excluded = {int(sim.void_id), int(target.void_id)}
    labels = sorted(
        int(label)
        for label in set(np.unique(sim_labels).tolist()).union(np.unique(target_labels).tolist())
        if int(label) not in excluded
    )
    rows: list[dict[str, int]] = []
    for label in labels:
        sim_count = _component_count(sim_labels == label)
        target_count = _component_count(target_labels == label)
        rows.append(
            {
                "material_id": int(label),
                "sim_components": sim_count,
                "target_components": target_count,
                "component_count_abs_diff": abs(sim_count - target_count),
            }
        )
    return rows


def compute_topology(
    sim: ViewFeature,
    target: ViewFeature,
    context: MetricContext,
) -> MetricComputation:
    del context
    sim_count = _component_count(sim.mask)
    target_count = _component_count(target.mask)
    union_diff = abs(sim_count - target_count)
    per_material = _label_component_rows(sim, target)
    material_diff = sum(int(row["component_count_abs_diff"]) for row in per_material)
    value = float(union_diff + material_diff)
    return MetricComputation(
        name="topology",
        loss=value,
        value=value,
        details={
            "metric": "topology",
            "mode": "projected_2d_component_count",
            "connectivity": 4,
            "sim_components": sim_count,
            "target_components": target_count,
            "union_component_count_abs_diff": union_diff,
            "material_component_count_abs_diff_sum": material_diff,
            "per_material": per_material,
        },
    )
