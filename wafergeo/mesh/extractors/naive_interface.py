from __future__ import annotations

import numpy as np

from wafergeo.mesh.config import MeshBuildConfig
from wafergeo.mesh.extractors.base import (
    MeshCapabilities,
    MeshExtractorProtocol,
    MeshMethodCard,
    RawMesh,
)
from wafergeo.sdf.tsdf import label_from_tsdf


class NaiveInterfaceExtractor(MeshExtractorProtocol):
    """Deterministic voxel-interface triangulation backend (NumPy only)."""

    name = "naive_interface"
    version = "1.0.0"
    capabilities = MeshCapabilities(
        input_types=("tsdf",),
        supports_anisotropic_spacing=True,
        deterministic=True,
    )
    method_card = MeshMethodCard(
        summary="Voxel interface triangulation by neighbor label differences.",
        dependencies=("numpy",),
        limitations=("Blocky surfaces; not smooth isocontouring.",),
        install_hint="",
    )

    def extract_from_tsdf(
        self,
        tsdf,
        cfg: MeshBuildConfig,
        material_ids: list[int],
    ) -> RawMesh:
        _ = cfg
        if len(material_ids) != tsdf.tsdf.shape[0]:
            raise ValueError("material_ids length must match tsdf channels")

        material = tsdf.material
        void_index = material.ids.index(material.void_id)
        label_zyx = label_from_tsdf(
            tsdf.tsdf,
            material,
            void_index=void_index,
            selected_material_ids=material_ids,
        )

        sz, sy, sx = (float(v) for v in tsdf.grid.spacing)
        oz, oy, ox = (float(v) for v in tsdf.grid.origin)
        z_size, y_size, x_size = label_zyx.shape

        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        vertex_map: dict[tuple[float, float, float], int] = {}

        def center_xyz(z: int, y: int, x: int) -> tuple[float, float, float]:
            return (ox + x * sx, oy + y * sy, oz + z * sz)

        def add_vertex(v: tuple[float, float, float]) -> int:
            idx = vertex_map.get(v)
            if idx is not None:
                return idx
            idx = len(vertices)
            vertices.append(v)
            vertex_map[v] = idx
            return idx

        def add_quad(corners: tuple[tuple[float, float, float], ...]) -> None:
            v0, v1, v2, v3 = (add_vertex(corner) for corner in corners)
            faces.append((v0, v1, v2))
            faces.append((v0, v2, v3))

        # X-normal interface planes.
        for z in range(z_size):
            for y in range(y_size):
                for x in range(x_size - 1):
                    if int(label_zyx[z, y, x]) == int(label_zyx[z, y, x + 1]):
                        continue
                    cx, cy, cz = center_xyz(z, y, x)
                    xb = cx + 0.5 * sx
                    corners = (
                        (xb, cy - 0.5 * sy, cz - 0.5 * sz),
                        (xb, cy + 0.5 * sy, cz - 0.5 * sz),
                        (xb, cy + 0.5 * sy, cz + 0.5 * sz),
                        (xb, cy - 0.5 * sy, cz + 0.5 * sz),
                    )
                    add_quad(corners)

        # Y-normal interface planes.
        for z in range(z_size):
            for y in range(y_size - 1):
                for x in range(x_size):
                    if int(label_zyx[z, y, x]) == int(label_zyx[z, y + 1, x]):
                        continue
                    cx, cy, cz = center_xyz(z, y, x)
                    yb = cy + 0.5 * sy
                    corners = (
                        (cx - 0.5 * sx, yb, cz - 0.5 * sz),
                        (cx + 0.5 * sx, yb, cz - 0.5 * sz),
                        (cx + 0.5 * sx, yb, cz + 0.5 * sz),
                        (cx - 0.5 * sx, yb, cz + 0.5 * sz),
                    )
                    add_quad(corners)

        # Z-normal interface planes.
        for z in range(z_size - 1):
            for y in range(y_size):
                for x in range(x_size):
                    if int(label_zyx[z, y, x]) == int(label_zyx[z + 1, y, x]):
                        continue
                    cx, cy, cz = center_xyz(z, y, x)
                    zb = cz + 0.5 * sz
                    corners = (
                        (cx - 0.5 * sx, cy - 0.5 * sy, zb),
                        (cx + 0.5 * sx, cy - 0.5 * sy, zb),
                        (cx + 0.5 * sx, cy + 0.5 * sy, zb),
                        (cx - 0.5 * sx, cy + 0.5 * sy, zb),
                    )
                    add_quad(corners)

        if not faces:
            return RawMesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
            )

        return RawMesh(
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int32),
        )
