from __future__ import annotations

import numpy as np

from wafergeo.core.geometry import vtk_polys_to_triangles
from wafergeo.mesh.config import MeshBuildConfig
from wafergeo.mesh.errors import MeshOptionalDependencyError
from wafergeo.mesh.extractors.base import (
    MeshCapabilities,
    MeshExtractorProtocol,
    MeshMethodCard,
    RawMesh,
)
from wafergeo.mesh.sampling import triangle_areas


def _bbox_center(vertices: np.ndarray) -> np.ndarray:
    if vertices.shape[0] == 0:
        return np.zeros((3,), dtype=np.float64)
    vmin = vertices.min(axis=0).astype(np.float64, copy=False)
    vmax = vertices.max(axis=0).astype(np.float64, copy=False)
    return 0.5 * (vmin + vmax)


def _surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    if faces.shape[0] == 0:
        return 0.0
    return float(np.sum(triangle_areas(vertices, faces)))


def _to_vtk_polydata(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    vtk,
    numpy_support,
):
    poly = vtk.vtkPolyData()
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return poly

    points = vtk.vtkPoints()
    points.SetData(
        numpy_support.numpy_to_vtk(
            num_array=vertices.astype(np.float32, copy=False),
            deep=True,
        )
    )
    poly.SetPoints(points)

    faces_i64 = faces.astype(np.int64, copy=False)
    connectivity = faces_i64.reshape(-1)
    offsets = np.arange(0, 3 * (faces_i64.shape[0] + 1), 3, dtype=np.int64)
    cell_array = vtk.vtkCellArray()
    cell_array.SetData(
        numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_support.numpy_to_vtkIdTypeArray(connectivity, deep=True),
    )
    poly.SetPolys(cell_array)
    return poly


def _from_vtk_polydata(poly, numpy_support) -> tuple[np.ndarray, np.ndarray]:
    points = poly.GetPoints()
    polys = poly.GetPolys()
    if points is None or polys is None:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
        )
    vertices = numpy_support.vtk_to_numpy(points.GetData()).astype(np.float32, copy=False)
    faces = vtk_polys_to_triangles(polys, numpy_support.vtk_to_numpy)
    return vertices, faces


def apply_vtk_visual_postprocess(
    vertices: np.ndarray,
    faces: np.ndarray,
    cfg: MeshBuildConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    metrics = {
        "bbox_shift_nm": 0.0,
        "area_rel_error": 0.0,
        "pre_faces": float(faces.shape[0]),
        "post_faces": float(faces.shape[0]),
    }
    if not cfg.vtk_viz_postprocess_enabled or faces.shape[0] == 0:
        return vertices, faces, metrics

    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError as exc:
        raise MeshOptionalDependencyError(
            "Mesh backend 'vtk' requires vtk. Install with: pip install 'wafergeo[vtk]'"
        ) from exc

    pre_center = _bbox_center(vertices)
    pre_area = _surface_area(vertices, faces)

    poly = _to_vtk_polydata(vertices, faces, vtk=vtk, numpy_support=numpy_support)
    if poly.GetNumberOfPoints() == 0:
        return vertices, faces, metrics

    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputData(poly)
    smooth.SetNumberOfIterations(int(max(cfg.vtk_smoothing_iterations, 0)))
    smooth.SetPassBand(float(max(cfg.vtk_smoothing_pass_band, 1e-6)))
    smooth.SetBoundarySmoothing(bool(cfg.vtk_boundary_smoothing))
    smooth.SetFeatureEdgeSmoothing(bool(cfg.vtk_feature_edge_smoothing))
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()
    current = smooth.GetOutput()

    levels = int(max(cfg.vtk_subdivision_levels, 0))
    if levels > 0:
        subdiv = vtk.vtkLoopSubdivisionFilter()
        subdiv.SetInputData(current)
        subdiv.SetNumberOfSubdivisions(levels)
        subdiv.Update()
        current = subdiv.GetOutput()

    post_vertices, post_faces = _from_vtk_polydata(current, numpy_support)
    if post_faces.shape[0] == 0:
        return vertices, faces, metrics

    post_center = _bbox_center(post_vertices)
    post_area = _surface_area(post_vertices, post_faces)
    area_denom = max(pre_area, 1e-12)
    metrics["bbox_shift_nm"] = float(np.linalg.norm(post_center - pre_center))
    metrics["area_rel_error"] = float(abs(post_area - pre_area) / area_denom)
    metrics["post_faces"] = float(post_faces.shape[0])
    return post_vertices, post_faces, metrics


class VTKInterfaceExtractor(MeshExtractorProtocol):
    """Optional VTK backend using FlyingEdges on all selected channels."""

    name = "vtk"
    version = "0.1.0"
    capabilities = MeshCapabilities(
        input_types=("tsdf",),
        supports_anisotropic_spacing=True,
        deterministic=True,
    )
    method_card = MeshMethodCard(
        summary="VTK FlyingEdges based extractor for TSDF channels with merged interfaces.",
        dependencies=("vtk",),
        limitations=("Geometry deduplication is tolerance-based.",),
        install_hint="pip install 'wafergeo[vtk]'",
    )

    @staticmethod
    def _extract_channel_surface(
        *,
        field_zyx: np.ndarray,
        iso_value: float,
        spacing_xyz: tuple[float, float, float],
        origin_xyz: tuple[float, float, float],
        vtk,
        numpy_support,
    ) -> RawMesh:
        z_size, y_size, x_size = field_zyx.shape
        image = vtk.vtkImageData()
        image.SetDimensions(x_size, y_size, z_size)
        image.SetSpacing(*spacing_xyz)
        image.SetOrigin(*origin_xyz)

        field_xyz = np.transpose(field_zyx.astype(np.float32, copy=False), (2, 1, 0))
        vtk_arr = numpy_support.numpy_to_vtk(
            num_array=field_xyz.ravel(order="F"),
            deep=True,
            array_type=vtk.VTK_FLOAT,
        )
        vtk_arr.SetName("tsdf")
        image.GetPointData().SetScalars(vtk_arr)

        contour = vtk.vtkFlyingEdges3D()
        contour.SetInputData(image)
        contour.SetValue(0, float(iso_value))
        contour.Update()

        poly = contour.GetOutput()
        points = poly.GetPoints()
        polys = poly.GetPolys()
        if points is None or polys is None:
            return RawMesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
            )

        vertices = numpy_support.vtk_to_numpy(points.GetData()).astype(np.float32, copy=False)
        faces = vtk_polys_to_triangles(polys, numpy_support.vtk_to_numpy)
        if faces.size == 0:
            return RawMesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
            )
        return RawMesh(vertices=vertices, faces=faces)

    @staticmethod
    def _deduplicate_geometry(
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        spacing_xyz: tuple[float, float, float],
    ) -> RawMesh:
        if faces.size == 0:
            return RawMesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
            )

        tolerance = max(min(spacing_xyz) * 1e-6, 1e-6)
        inv_tol = 1.0 / tolerance

        vertex_lookup: dict[tuple[int, int, int], int] = {}
        out_vertices: list[np.ndarray] = []
        out_faces: list[tuple[int, int, int]] = []
        seen_triangles: set[tuple[tuple[int, int, int], ...]] = set()

        for face in faces:
            tri = vertices[face]
            tri_keys = [
                (
                    int(np.rint(float(point[0]) * inv_tol)),
                    int(np.rint(float(point[1]) * inv_tol)),
                    int(np.rint(float(point[2]) * inv_tol)),
                )
                for point in tri
            ]
            tri_canonical = tuple(sorted(tri_keys))
            if tri_canonical in seen_triangles:
                continue
            seen_triangles.add(tri_canonical)

            remapped: list[int] = []
            for key, point in zip(tri_keys, tri, strict=True):
                index = vertex_lookup.get(key)
                if index is None:
                    index = len(out_vertices)
                    vertex_lookup[key] = index
                    out_vertices.append(point.astype(np.float32, copy=False))
                remapped.append(index)
            if len({remapped[0], remapped[1], remapped[2]}) < 3:
                continue
            out_faces.append((remapped[0], remapped[1], remapped[2]))

        if not out_faces:
            return RawMesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
            )
        return RawMesh(
            vertices=np.asarray(out_vertices, dtype=np.float32),
            faces=np.asarray(out_faces, dtype=np.int32),
        )

    def extract_from_tsdf(
        self,
        tsdf,
        cfg: MeshBuildConfig,
        material_ids: list[int],
    ) -> RawMesh:
        if len(material_ids) != tsdf.tsdf.shape[0]:
            raise ValueError("material_ids length must match tsdf channels")

        try:
            import vtk
            from vtk.util import numpy_support
        except ImportError as exc:
            raise MeshOptionalDependencyError(
                "Mesh backend 'vtk' requires vtk. Install with: pip install 'wafergeo[vtk]'"
            ) from exc

        sx = float(tsdf.grid.spacing[2])
        sy = float(tsdf.grid.spacing[1])
        sz = float(tsdf.grid.spacing[0])
        ox = float(tsdf.grid.origin[2])
        oy = float(tsdf.grid.origin[1])
        oz = float(tsdf.grid.origin[0])
        spacing_xyz = (sx, sy, sz)
        origin_xyz = (ox, oy, oz)

        active_channels = list(range(tsdf.tsdf.shape[0]))
        if tsdf.present_mask is not None:
            mask = np.asarray(tsdf.present_mask, dtype=bool)
            if mask.shape[0] == len(active_channels):
                active_channels = [channel for channel in active_channels if bool(mask[channel])]
        if not active_channels:
            return RawMesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
            )

        merged_vertices: list[np.ndarray] = []
        merged_faces: list[np.ndarray] = []
        vertex_offset = 0
        for channel in active_channels:
            field_zyx = np.asarray(tsdf.tsdf[channel], dtype=np.float32)
            channel_mesh = self._extract_channel_surface(
                field_zyx=field_zyx,
                iso_value=cfg.iso_value,
                spacing_xyz=spacing_xyz,
                origin_xyz=origin_xyz,
                vtk=vtk,
                numpy_support=numpy_support,
            )
            if channel_mesh.faces.size == 0:
                continue
            merged_vertices.append(channel_mesh.vertices)
            merged_faces.append(channel_mesh.faces + vertex_offset)
            vertex_offset += int(channel_mesh.vertices.shape[0])

        if not merged_faces:
            return RawMesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
            )

        vertices = np.concatenate(merged_vertices, axis=0).astype(np.float32, copy=False)
        faces = np.concatenate(merged_faces, axis=0).astype(np.int32, copy=False)
        return self._deduplicate_geometry(
            vertices,
            faces,
            spacing_xyz=spacing_xyz,
        )
