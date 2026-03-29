from __future__ import annotations

import numpy as np

from wafergeo.core.types import MeshGeom


def _choose_material_id(mesh: MeshGeom, face_index: int) -> int:
    mat_in = int(mesh.face_mat_in[face_index])
    mat_out = int(mesh.face_mat_out[face_index])
    void_id = int(mesh.material.void_id)

    candidates = [material_id for material_id in (mat_in, mat_out) if material_id != void_id]
    if not candidates:
        return void_id

    priority_map = {
        int(material_id): int(priority)
        for material_id, priority in zip(mesh.material.ids, mesh.material.priority, strict=True)
    }
    return max(
        candidates,
        key=lambda material_id: (priority_map.get(material_id, -10_000), -material_id),
    )


def voxelize_mesh_to_label(
    mesh: MeshGeom,
    spacing_zyx: tuple[float, float, float],
    origin_zyx: tuple[float, float, float],
    shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    if len(spacing_zyx) != 3 or len(origin_zyx) != 3 or len(shape_zyx) != 3:
        raise ValueError("spacing_zyx/origin_zyx/shape_zyx must each be length 3")

    z_size, y_size, x_size = (int(v) for v in shape_zyx)
    if min(z_size, y_size, x_size) <= 0:
        raise ValueError(f"shape_zyx must be positive, got {shape_zyx}")

    sz, sy, sx = (float(v) for v in spacing_zyx)
    oz, oy, ox = (float(v) for v in origin_zyx)

    void_id = int(mesh.material.void_id)
    label = np.full((z_size, y_size, x_size), void_id, dtype=np.int32)
    priority_grid = np.full((z_size, y_size, x_size), -10_000, dtype=np.int32)

    priority_map = {
        int(material_id): int(priority)
        for material_id, priority in zip(mesh.material.ids, mesh.material.priority, strict=True)
    }

    for face_index, face in enumerate(mesh.faces):
        tri_xyz = mesh.vertices[np.asarray(face, dtype=np.int64)]
        min_xyz = np.min(tri_xyz, axis=0)
        max_xyz = np.max(tri_xyz, axis=0)

        x0 = max(0, int(np.floor((float(min_xyz[0]) - ox) / sx)))
        x1 = min(x_size - 1, int(np.ceil((float(max_xyz[0]) - ox) / sx)))
        y0 = max(0, int(np.floor((float(min_xyz[1]) - oy) / sy)))
        y1 = min(y_size - 1, int(np.ceil((float(max_xyz[1]) - oy) / sy)))
        z0 = max(0, int(np.floor((float(min_xyz[2]) - oz) / sz)))
        z1 = min(z_size - 1, int(np.ceil((float(max_xyz[2]) - oz) / sz)))

        if x0 > x1 or y0 > y1 or z0 > z1:
            continue

        material_id = _choose_material_id(mesh, face_index)
        prio = int(priority_map.get(material_id, -10_000))

        region_prio = priority_grid[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
        update = prio >= region_prio
        if not np.any(update):
            continue

        region_label = label[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
        region_label[update] = material_id
        region_prio[update] = prio

    max_id = int(max(mesh.material.ids))
    out_dtype = np.uint8 if max_id <= 255 else np.uint16
    return label.astype(out_dtype, copy=False)
