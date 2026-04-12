from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wafergeo.core.grid import GridSpec
from wafergeo.core.meta import Meta


@dataclass(frozen=True)
class MaterialSpec:
    ids: list[int]
    names: list[str]
    void_id: int
    priority: list[int]
    ignore_in_exposure: list[bool]

    def __post_init__(self) -> None:
        length = len(self.ids)
        if length == 0:
            raise ValueError("ids must be non-empty")
        if len(set(self.ids)) != length:
            raise ValueError("ids must be unique")
        if len(self.names) != length:
            raise ValueError("names length must match ids length")
        if len(self.priority) != length:
            raise ValueError("priority length must match ids length")
        if len(self.ignore_in_exposure) != length:
            raise ValueError("ignore_in_exposure length must match ids length")
        if self.void_id not in self.ids:
            raise ValueError(f"void_id={self.void_id} must exist in ids")


@dataclass(frozen=True)
class LabelVolume:
    grid: GridSpec
    material: MaterialSpec
    material_id: np.ndarray
    meta: Meta

    def __post_init__(self) -> None:
        if self.material_id.ndim != 3:
            raise ValueError(f"material_id must be 3D (Z,Y,X), got ndim={self.material_id.ndim}")
        if not np.issubdtype(self.material_id.dtype, np.integer):
            raise ValueError("material_id dtype must be integer")
        unique_ids = set(np.unique(self.material_id).tolist())
        allowed = set(self.material.ids)
        if not unique_ids.issubset(allowed):
            unknown = sorted(unique_ids.difference(allowed))
            raise ValueError(f"material_id contains unknown ids: {unknown}")


@dataclass(frozen=True)
class TSDFVolume:
    grid: GridSpec
    material: MaterialSpec
    mu_nm: float
    tsdf: np.ndarray
    d_boundary: np.ndarray | None = None
    pair_code: np.ndarray | None = None
    present_mask: np.ndarray | None = None
    meta: Meta | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.mu_nm) or self.mu_nm <= 0.0:
            raise ValueError(f"mu_nm must be finite and > 0, got {self.mu_nm}")
        if self.tsdf.ndim != 4:
            raise ValueError(f"tsdf must be 4D (M,Z,Y,X), got ndim={self.tsdf.ndim}")
        if not np.issubdtype(self.tsdf.dtype, np.floating):
            raise ValueError("tsdf dtype must be floating")
        max_m = len(self.material.ids)
        m = self.tsdf.shape[0]
        if m <= 0 or m > max_m:
            raise ValueError(
                "tsdf first axis must be between 1 and material count "
                f"({max_m}), got {self.tsdf.shape[0]}"
            )
        if not np.isfinite(self.tsdf).all():
            raise ValueError("tsdf contains NaN/Inf")
        if self.tsdf.min() < -1.0001 or self.tsdf.max() > 1.0001:
            raise ValueError("tsdf values must be within [-1,1]")
        zyx_shape = self.tsdf.shape[1:]
        if self.grid.dim == 2 and zyx_shape[0] != 1:
            raise ValueError("for dim=2 compatibility, TSDFVolume expects Z=1")
        if self.d_boundary is not None and self.d_boundary.shape != zyx_shape:
            raise ValueError("d_boundary must have shape (Z,Y,X)")
        if self.pair_code is not None and self.pair_code.shape != zyx_shape:
            raise ValueError("pair_code must have shape (Z,Y,X)")
        if self.present_mask is not None:
            present = np.asarray(self.present_mask)
            if present.shape != (m,):
                raise ValueError(f"present_mask must be shape ({m},), got {present.shape}")
            if not np.issubdtype(present.dtype, np.bool_):
                bool_cast = present.astype(bool)
                if not np.array_equal(present, bool_cast.astype(present.dtype)):
                    raise ValueError("present_mask must be boolean-like")


@dataclass(frozen=True)
class MeshGeom:
    vertices: np.ndarray
    faces: np.ndarray
    face_mat_in: np.ndarray
    face_mat_out: np.ndarray
    face_is_exposed: np.ndarray
    grid: GridSpec
    material: MaterialSpec
    meta: Meta

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be shape (V,3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must be shape (F,3)")
        f = self.faces.shape[0]
        for name, arr in {
            "face_mat_in": self.face_mat_in,
            "face_mat_out": self.face_mat_out,
            "face_is_exposed": self.face_is_exposed,
        }.items():
            if arr.shape != (f,):
                raise ValueError(f"{name} must be shape ({f},)")


@dataclass(frozen=True)
class PointCloud:
    points: np.ndarray
    normals: np.ndarray
    pair_code: np.ndarray
    point_is_exposed: np.ndarray
    meta: Meta

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must be shape (N,3)")
        if self.normals.shape != self.points.shape:
            raise ValueError("normals shape must match points shape")
        n = self.points.shape[0]
        if self.pair_code.shape != (n,):
            raise ValueError("pair_code must be shape (N,)")
        if self.point_is_exposed.shape != (n,):
            raise ValueError("point_is_exposed must be shape (N,)")
        if not np.issubdtype(self.point_is_exposed.dtype, np.bool_):
            bool_cast = self.point_is_exposed.astype(bool)
            if not np.array_equal(
                self.point_is_exposed,
                bool_cast.astype(self.point_is_exposed.dtype),
                ):
                    raise ValueError("point_is_exposed must be boolean-like")
