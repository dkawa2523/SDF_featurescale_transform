from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wafergeo.mesh.config import MeshMode


@dataclass(frozen=True)
class FaceAttributes:
    face_mat_in: np.ndarray
    face_mat_out: np.ndarray
    face_is_exposed: np.ndarray
    face_pair_code: np.ndarray
    face_normals: np.ndarray


def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    v1 = tri[:, 1] - tri[:, 0]
    v2 = tri[:, 2] - tri[:, 0]
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    safe = norms > 0.0
    normals[safe] /= norms[safe][:, np.newaxis]
    normals[~safe] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normals.astype(np.float32, copy=False)


def _sample_tsdf_channels(
    points_xyz: np.ndarray,
    tsdf_4d: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    origin_zyx: tuple[float, float, float],
) -> np.ndarray:
    sz, sy, sx = spacing_zyx
    oz, oy, ox = origin_zyx

    x_idx = np.rint((points_xyz[:, 0] - ox) / sx).astype(np.int64)
    y_idx = np.rint((points_xyz[:, 1] - oy) / sy).astype(np.int64)
    z_idx = np.rint((points_xyz[:, 2] - oz) / sz).astype(np.int64)

    z_idx = np.clip(z_idx, 0, tsdf_4d.shape[1] - 1)
    y_idx = np.clip(y_idx, 0, tsdf_4d.shape[2] - 1)
    x_idx = np.clip(x_idx, 0, tsdf_4d.shape[3] - 1)
    return tsdf_4d[:, z_idx, y_idx, x_idx].transpose(1, 0).astype(np.float32, copy=False)


def _classify_points_from_tsdf(
    sampled_tsdf: np.ndarray,
    *,
    channel_material_ids: list[int],
    material_ids: list[int],
    material_priority: list[int],
    void_id: int,
) -> np.ndarray:
    id_to_priority = {
        int(mid): int(priority)
        for mid, priority in zip(material_ids, material_priority, strict=True)
    }
    priority = np.asarray([id_to_priority[mid] for mid in channel_material_ids], dtype=np.int32)
    best_idx = np.argmin(sampled_tsdf, axis=1)
    best_val = sampled_tsdf[np.arange(sampled_tsdf.shape[0]), best_idx]

    # deterministic tie-break by priority then channel index
    for channel in range(sampled_tsdf.shape[1]):
        ties = np.isclose(sampled_tsdf[:, channel], best_val, rtol=0.0, atol=1e-7)
        better = ties & (priority[channel] > priority[best_idx])
        same = ties & (priority[channel] == priority[best_idx]) & (channel < best_idx)
        update = better | same
        best_idx[update] = channel
        best_val[update] = sampled_tsdf[update, channel]

    labels = np.asarray(channel_material_ids, dtype=np.int32)[best_idx]
    outside = np.all(sampled_tsdf > 0.0, axis=1)
    labels[outside] = int(void_id)
    return labels.astype(np.int32, copy=False)


def annotate_faces_from_tsdf(
    vertices: np.ndarray,
    faces: np.ndarray,
    tsdf,
    material_ids: list[int],
    *,
    epsilon_scale: float = 0.25,
) -> FaceAttributes:
    if faces.size == 0:
        empty_i = np.zeros((0,), dtype=np.int32)
        empty_b = np.zeros((0,), dtype=bool)
        empty_n = np.zeros((0, 3), dtype=np.float32)
        return FaceAttributes(
            face_mat_in=empty_i,
            face_mat_out=empty_i,
            face_is_exposed=empty_b,
            face_pair_code=empty_i,
            face_normals=empty_n,
        )

    material = tsdf.material
    channel_material_ids = [int(v) for v in material_ids]

    normals = _compute_face_normals(vertices, faces)
    triangles = vertices[faces]
    centroids = np.mean(triangles, axis=1)
    min_spacing = float(min(tsdf.grid.spacing))
    eps = max(epsilon_scale * min_spacing, 0.2 * min_spacing)

    in_points = centroids - eps * normals
    out_points = centroids + eps * normals

    spacing_zyx = (
        float(tsdf.grid.spacing[0]),
        float(tsdf.grid.spacing[1]),
        float(tsdf.grid.spacing[2]),
    )
    origin_zyx = (
        float(tsdf.grid.origin[0]),
        float(tsdf.grid.origin[1]),
        float(tsdf.grid.origin[2]),
    )

    in_tsdf = _sample_tsdf_channels(in_points, tsdf.tsdf, spacing_zyx, origin_zyx)
    out_tsdf = _sample_tsdf_channels(out_points, tsdf.tsdf, spacing_zyx, origin_zyx)
    mat_in = _classify_points_from_tsdf(
        in_tsdf,
        channel_material_ids=channel_material_ids,
        material_ids=material.ids,
        material_priority=material.priority,
        void_id=int(material.void_id),
    )
    mat_out = _classify_points_from_tsdf(
        out_tsdf,
        channel_material_ids=channel_material_ids,
        material_ids=material.ids,
        material_priority=material.priority,
        void_id=int(material.void_id),
    )

    ambiguous = mat_in == mat_out
    if np.any(ambiguous):
        eps2 = 0.49 * min_spacing
        in_points2 = centroids[ambiguous] - eps2 * normals[ambiguous]
        out_points2 = centroids[ambiguous] + eps2 * normals[ambiguous]
        in_tsdf2 = _sample_tsdf_channels(in_points2, tsdf.tsdf, spacing_zyx, origin_zyx)
        out_tsdf2 = _sample_tsdf_channels(out_points2, tsdf.tsdf, spacing_zyx, origin_zyx)
        mat_in[ambiguous] = _classify_points_from_tsdf(
            in_tsdf2,
            channel_material_ids=channel_material_ids,
            material_ids=material.ids,
            material_priority=material.priority,
            void_id=int(material.void_id),
        )
        mat_out[ambiguous] = _classify_points_from_tsdf(
            out_tsdf2,
            channel_material_ids=channel_material_ids,
            material_ids=material.ids,
            material_priority=material.priority,
            void_id=int(material.void_id),
        )

    void_id = int(material.void_id)
    is_exposed = (mat_in == void_id) | (mat_out == void_id)

    canonical_pairs = [
        tuple(sorted((int(a), int(b)))) for a, b in zip(mat_in, mat_out, strict=True)
    ]
    unique_pairs = sorted(set(canonical_pairs))
    pair_to_code = {pair: idx for idx, pair in enumerate(unique_pairs)}
    pair_code = np.asarray([pair_to_code[pair] for pair in canonical_pairs], dtype=np.int32)

    return FaceAttributes(
        face_mat_in=mat_in,
        face_mat_out=mat_out,
        face_is_exposed=is_exposed.astype(bool, copy=False),
        face_pair_code=pair_code,
        face_normals=normals,
    )


def apply_mesh_mode(
    vertices: np.ndarray,
    faces: np.ndarray,
    attrs: FaceAttributes,
    *,
    mode: MeshMode,
) -> tuple[np.ndarray, np.ndarray, FaceAttributes]:
    if mode == "material_shell":
        return vertices, faces, attrs

    mat_in = attrs.face_mat_in.copy()
    mat_out = attrs.face_mat_out.copy()

    swap = mat_in > mat_out
    tmp = mat_in[swap].copy()
    mat_in[swap] = mat_out[swap]
    mat_out[swap] = tmp

    keep = mat_in != mat_out
    if not np.any(keep):
        return (
            vertices,
            np.zeros((0, 3), dtype=np.int32),
            FaceAttributes(
                face_mat_in=np.zeros((0,), dtype=np.int32),
                face_mat_out=np.zeros((0,), dtype=np.int32),
                face_is_exposed=np.zeros((0,), dtype=bool),
                face_pair_code=np.zeros((0,), dtype=np.int32),
                face_normals=np.zeros((0, 3), dtype=np.float32),
            ),
        )

    kept_faces = faces[keep]
    kept_in = mat_in[keep]
    kept_out = mat_out[keep]
    kept_exposed = attrs.face_is_exposed[keep]
    kept_normals = attrs.face_normals[keep]

    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    selected: list[int] = []
    for idx, face in enumerate(kept_faces):
        key = (int(kept_in[idx]), int(kept_out[idx]), tuple(sorted(int(v) for v in face.tolist())))
        if key in seen:
            continue
        seen.add(key)
        selected.append(idx)

    selected_idx = np.asarray(selected, dtype=np.int64)
    out_faces = kept_faces[selected_idx].astype(np.int32, copy=False)
    out_in = kept_in[selected_idx].astype(np.int32, copy=False)
    out_out = kept_out[selected_idx].astype(np.int32, copy=False)
    out_exposed = kept_exposed[selected_idx].astype(bool, copy=False)
    out_normals = kept_normals[selected_idx].astype(np.float32, copy=False)

    pairs = [tuple(sorted((int(a), int(b)))) for a, b in zip(out_in, out_out, strict=True)]
    unique_pairs = sorted(set(pairs))
    pair_to_code = {pair: idx for idx, pair in enumerate(unique_pairs)}
    out_pair_code = np.asarray([pair_to_code[p] for p in pairs], dtype=np.int32)

    return (
        vertices,
        out_faces,
        FaceAttributes(
            face_mat_in=out_in,
            face_mat_out=out_out,
            face_is_exposed=out_exposed,
            face_pair_code=out_pair_code,
            face_normals=out_normals,
        ),
    )
