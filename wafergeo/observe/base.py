from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from wafergeo.core.grid import GridSpec
from wafergeo.core.meta import Meta
from wafergeo.core.types import LabelVolume, MaterialSpec, MeshGeom, Obs2D, TSDFVolume
from wafergeo.observe.spec import ObserverSpecV2
from wafergeo.sdf.tsdf import label_from_tsdf

GeomInput = LabelVolume | TSDFVolume | MeshGeom


@runtime_checkable
class ObserverProtocol(Protocol):
    kind: str

    def observe(self, geom: GeomInput, spec: ObserverSpecV2) -> Obs2D:
        ...


@dataclass(frozen=True)
class LabelizedInput:
    label_zyx: np.ndarray
    grid: GridSpec
    material: MaterialSpec
    parent_meta: Meta
    source_kind: str


def _parse_selected_material_ids(raw: str) -> list[int]:
    items = [v.strip() for v in raw.split(",") if v.strip()]
    if not items:
        raise ValueError("selected_material_ids is empty")
    return [int(v) for v in items]


def _infer_mesh_shape_zyx(mesh: MeshGeom, spec: ObserverSpecV2) -> tuple[int, int, int]:
    from_params = spec.params.get("mesh_shape_zyx")
    if isinstance(from_params, (list, tuple)) and len(from_params) == 3:
        return (int(from_params[0]), int(from_params[1]), int(from_params[2]))

    if mesh.vertices.shape[0] == 0:
        raise ValueError("mesh has no vertices; specify params.mesh_shape_zyx")

    sz, sy, sx = (float(v) for v in mesh.grid.spacing)
    oz, oy, ox = (float(v) for v in mesh.grid.origin)

    x_max = float(np.max(mesh.vertices[:, 0]))
    y_max = float(np.max(mesh.vertices[:, 1]))
    z_max = float(np.max(mesh.vertices[:, 2]))

    x_size = max(1, int(np.ceil((x_max - ox) / sx)) + 1)
    y_size = max(1, int(np.ceil((y_max - oy) / sy)) + 1)
    z_size = max(1, int(np.ceil((z_max - oz) / sz)) + 1)
    return (z_size, y_size, x_size)


def coerce_geom_to_label(geom: GeomInput, spec: ObserverSpecV2) -> LabelizedInput:
    if isinstance(geom, LabelVolume):
        return LabelizedInput(
            label_zyx=np.asarray(geom.material_id),
            grid=geom.grid,
            material=geom.material,
            parent_meta=geom.meta,
            source_kind="label",
        )

    if isinstance(geom, TSDFVolume):
        channel_count = geom.tsdf.shape[0]
        selected_ids: list[int]
        if geom.meta is not None:
            selected_raw = geom.meta.extra.get("selected_material_ids")
            if selected_raw:
                parsed = _parse_selected_material_ids(selected_raw)
                if len(parsed) == channel_count:
                    selected_ids = parsed
                else:
                    selected_ids = list(geom.material.ids[:channel_count])
            else:
                selected_ids = list(geom.material.ids[:channel_count])
            parent_meta = geom.meta
        else:
            selected_ids = list(geom.material.ids[:channel_count])
            parent_meta = Meta(
                schema_version="observer/v2",
                profile_id="observer_unknown_parent",
                config_hash="unknown",
                generator_version="0.1.0",
                git_commit="unknown",
                input_hash="unknown",
                created_at="1970-01-01T00:00:00+00:00",
                extra={},
            )

        void_index = geom.material.ids.index(geom.material.void_id)
        label_zyx = label_from_tsdf(
            geom.tsdf,
            geom.material,
            void_index=void_index,
            selected_material_ids=selected_ids,
        )
        return LabelizedInput(
            label_zyx=label_zyx,
            grid=geom.grid,
            material=geom.material,
            parent_meta=parent_meta,
            source_kind="tsdf",
        )

    if isinstance(geom, MeshGeom):
        from wafergeo.observe.rasterize import voxelize_mesh_to_label

        shape_zyx = _infer_mesh_shape_zyx(geom, spec)
        spacing_zyx = (
            float(geom.grid.spacing[0]),
            float(geom.grid.spacing[1]),
            float(geom.grid.spacing[2]),
        )
        origin_zyx = (
            float(geom.grid.origin[0]),
            float(geom.grid.origin[1]),
            float(geom.grid.origin[2]),
        )
        label_zyx = voxelize_mesh_to_label(
            geom,
            spacing_zyx=spacing_zyx,
            origin_zyx=origin_zyx,
            shape_zyx=shape_zyx,
        )
        return LabelizedInput(
            label_zyx=label_zyx,
            grid=geom.grid,
            material=geom.material,
            parent_meta=geom.meta,
            source_kind="mesh",
        )

    raise TypeError(f"unsupported geometry type: {type(geom).__name__}")
