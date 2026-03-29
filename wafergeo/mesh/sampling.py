from __future__ import annotations

import numpy as np

from wafergeo.core.types import PointCloud


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def sample_pointcloud(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    face_pair_code: np.ndarray,
    face_is_exposed: np.ndarray,
    n_points: int,
    seed: int,
    meta,
) -> PointCloud:
    if n_points <= 0:
        raise ValueError(f"n_points must be > 0, got {n_points}")
    if faces.shape[0] == 0:
        raise ValueError("cannot sample point cloud from empty mesh")

    areas = triangle_areas(vertices, faces)
    total_area = float(np.sum(areas))
    if total_area <= 0.0:
        raise ValueError("mesh total area must be > 0 for point sampling")

    prob = areas / total_area
    rng = np.random.default_rng(seed)
    picked = rng.choice(faces.shape[0], size=n_points, replace=True, p=prob)

    tri = vertices[faces[picked]]
    r1 = rng.random(n_points)
    r2 = rng.random(n_points)
    sr1 = np.sqrt(r1)
    w0 = 1.0 - sr1
    w1 = sr1 * (1.0 - r2)
    w2 = sr1 * r2

    points = (
        tri[:, 0] * w0[:, np.newaxis]
        + tri[:, 1] * w1[:, np.newaxis]
        + tri[:, 2] * w2[:, np.newaxis]
    ).astype(np.float32, copy=False)

    normals = face_normals[picked].astype(np.float32, copy=False)
    pair_code = face_pair_code[picked].astype(np.int32, copy=False)
    is_exposed = face_is_exposed[picked].astype(bool, copy=False)

    return PointCloud(
        points=points,
        normals=normals,
        pair_code=pair_code,
        point_is_exposed=is_exposed,
        meta=meta,
    )
