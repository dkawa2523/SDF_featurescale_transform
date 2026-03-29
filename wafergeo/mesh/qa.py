from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from wafergeo.mesh.sampling import triangle_areas


@dataclass(frozen=True)
class MeshQA:
    num_vertices: int
    num_faces: int
    degenerate_face_count: int
    degenerate_face_rate: float
    interface_area_total: float
    exposed_face_ratio: float
    interface_area_by_pair: dict[str, float]
    post_bbox_shift_nm: float | None = None
    post_area_rel_error: float | None = None
    postprocess_status: Literal["OK", "WARN", "FAIL"] | None = None
    notes: list[str] = field(default_factory=list)


def compute_mesh_qa(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_mat_in: np.ndarray,
    face_mat_out: np.ndarray,
    face_is_exposed: np.ndarray,
    degenerate_area_eps: float,
    post_bbox_shift_nm: float | None = None,
    post_area_rel_error: float | None = None,
    postprocess_status: Literal["OK", "WARN", "FAIL"] | None = None,
    postprocess_notes: list[str] | None = None,
) -> MeshQA:
    num_vertices = int(vertices.shape[0])
    num_faces = int(faces.shape[0])

    notes: list[str] = list(postprocess_notes or [])
    if num_faces == 0:
        notes.append("empty mesh")
        return MeshQA(
            num_vertices=num_vertices,
            num_faces=num_faces,
            degenerate_face_count=0,
            degenerate_face_rate=0.0,
            interface_area_total=0.0,
            exposed_face_ratio=0.0,
            interface_area_by_pair={},
            post_bbox_shift_nm=post_bbox_shift_nm,
            post_area_rel_error=post_area_rel_error,
            postprocess_status=postprocess_status,
            notes=notes,
        )

    areas = triangle_areas(vertices, faces)
    degenerate = areas <= float(degenerate_area_eps)
    degenerate_count = int(np.sum(degenerate))
    degenerate_rate = float(degenerate_count) / float(num_faces)

    if degenerate_count > 0:
        notes.append(f"degenerate_face_count={degenerate_count}")

    interface_area_total = float(np.sum(areas))
    exposed_face_ratio = float(np.mean(face_is_exposed.astype(np.float32)))

    by_pair: dict[str, float] = {}
    for idx, area in enumerate(areas.tolist()):
        a = int(face_mat_in[idx])
        b = int(face_mat_out[idx])
        pair = tuple(sorted((a, b)))
        key = f"{pair[0]}-{pair[1]}"
        by_pair[key] = by_pair.get(key, 0.0) + float(area)

    return MeshQA(
        num_vertices=num_vertices,
        num_faces=num_faces,
        degenerate_face_count=degenerate_count,
        degenerate_face_rate=degenerate_rate,
        interface_area_total=interface_area_total,
        exposed_face_ratio=exposed_face_ratio,
        interface_area_by_pair=by_pair,
        post_bbox_shift_nm=post_bbox_shift_nm,
        post_area_rel_error=post_area_rel_error,
        postprocess_status=postprocess_status,
        notes=notes,
    )
