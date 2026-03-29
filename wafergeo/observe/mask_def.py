from __future__ import annotations

import numpy as np

from wafergeo.core.types import MaterialSpec
from wafergeo.observe.spec import MaskDefSpec


def _name_to_id_map(material: MaterialSpec) -> dict[str, int]:
    return {
        str(name): int(material_id)
        for material_id, name in zip(material.ids, material.names, strict=True)
    }


def _resolve_include_ids(spec: MaskDefSpec, material: MaterialSpec) -> set[int]:
    include_ids = {int(v) for v in spec.include_ids}
    name_to_id = _name_to_id_map(material)
    for name in spec.include_materials:
        if name in name_to_id:
            include_ids.add(name_to_id[name])
    return include_ids


def _resolve_ignore_ids(spec: MaskDefSpec, material: MaterialSpec) -> set[int]:
    ignore_ids = {int(v) for v in spec.ignore_ids}
    name_to_id = _name_to_id_map(material)
    for name in spec.ignore_materials:
        if name in name_to_id:
            ignore_ids.add(name_to_id[name])
    return ignore_ids


def mask2d_from_exposed_id(
    exposed_id_2d: np.ndarray,
    spec: MaskDefSpec,
    material: MaterialSpec,
) -> np.ndarray:
    exposed = np.asarray(exposed_id_2d)
    if exposed.ndim != 2:
        raise ValueError(f"exposed_id_2d must be 2D, got ndim={exposed.ndim}")

    ignore_ids = _resolve_ignore_ids(spec, material)
    ignore_ids.add(int(material.void_id))

    if spec.kind == "binary_solid":
        out = ~np.isin(exposed, np.asarray(sorted(ignore_ids), dtype=np.int32))
        return out.astype(np.uint8)

    if spec.kind in {"exposed_union", "material_union"}:
        include_ids = _resolve_include_ids(spec, material)
        if not include_ids:
            include_ids = {int(mid) for mid in material.ids if mid not in ignore_ids}
        out = np.isin(exposed, np.asarray(sorted(include_ids), dtype=np.int32))
        return out.astype(np.uint8)

    raise ValueError(f"unsupported mask_definition.kind for exposed map: {spec.kind}")


def mask2d_from_slice_label(
    label2d: np.ndarray,
    spec: MaskDefSpec,
    material: MaterialSpec,
) -> np.ndarray:
    labels = np.asarray(label2d)
    if labels.ndim != 2:
        raise ValueError(f"label2d must be 2D, got ndim={labels.ndim}")

    if spec.kind == "binary_solid":
        return (labels != int(material.void_id)).astype(np.uint8)

    if spec.kind in {"material_union", "exposed_union"}:
        include_ids = _resolve_include_ids(spec, material)
        if not include_ids:
            include_ids = {int(mid) for mid in material.ids if mid != material.void_id}
        out = np.isin(labels, np.asarray(sorted(include_ids), dtype=np.int32))
        return out.astype(np.uint8)

    raise ValueError(f"unsupported mask_definition.kind for slice map: {spec.kind}")
