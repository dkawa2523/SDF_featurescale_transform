from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from wafergeo.core.meta import Meta
from wafergeo.core.types import Obs2D
from wafergeo.observe.base import GeomInput, coerce_geom_to_label
from wafergeo.observe.contour_extract import extract_contours_from_tsdf
from wafergeo.observe.errors import ObserverOptionalDependencyError
from wafergeo.observe.mask_def import mask2d_from_slice_label
from wafergeo.observe.qa import run_observer_qa
from wafergeo.observe.spec import ObserverSpecV2, observer_spec_hash
from wafergeo.observe.tsdf2d import tsdf2d_from_mask


class SliceObserver:
    kind = "slice"

    def _param_float(self, params: dict[str, object], key: str, default: float) -> float:
        value = params.get(key, default)
        if isinstance(value, (int, float, str)):
            return float(value)
        return float(default)

    def _axis_spacing_origin(
        self,
        axis: str,
        spacing_zyx: tuple[float, float, float],
        origin_zyx: tuple[float, float, float],
    ) -> tuple[float, float]:
        if axis == "z":
            return float(spacing_zyx[0]), float(origin_zyx[0])
        if axis == "y":
            return float(spacing_zyx[1]), float(origin_zyx[1])
        if axis == "x":
            return float(spacing_zyx[2]), float(origin_zyx[2])
        raise ValueError(f"unsupported axis: {axis}")

    def _slice_index(
        self,
        axis: str,
        coord_nm: float,
        spacing_zyx: tuple[float, float, float],
        origin_zyx: tuple[float, float, float],
        shape_zyx: tuple[int, int, int],
    ) -> int:
        spacing, origin = self._axis_spacing_origin(axis, spacing_zyx, origin_zyx)
        idx = int(np.rint((float(coord_nm) - origin) / spacing))
        max_idx = {"z": shape_zyx[0] - 1, "y": shape_zyx[1] - 1, "x": shape_zyx[2] - 1}[axis]
        return int(np.clip(idx, 0, max_idx))

    def _extract_plane(self, label_zyx: np.ndarray, axis: str, index: int) -> np.ndarray:
        if axis == "z":
            return label_zyx[index, :, :]
        if axis == "y":
            return label_zyx[:, index, :]
        if axis == "x":
            return label_zyx[:, :, index]
        raise ValueError(f"unsupported axis: {axis}")

    def _build_meta(
        self,
        parent_meta: Meta,
        spec: ObserverSpecV2,
        *,
        source_kind: str,
        axis: str,
        center_index: int,
        slice_count: int,
        qa_status: str,
        qa_notes: list[str],
    ) -> Meta:
        spec_hash = observer_spec_hash(spec)
        extra = dict(parent_meta.extra)
        extra.update(
            {
                "observer_name": spec.name,
                "observer_kind": self.kind,
                "observer_spec_hash": spec_hash,
                "observer_spec_version": spec.schema_version,
                "source_kind": source_kind,
                "slice_axis": axis,
                "slice_center_index": str(center_index),
                "slice_count": str(slice_count),
                "observer_qa_status": qa_status,
                "observer_qa_notes": " | ".join(qa_notes),
            }
        )

        return Meta(
            schema_version=spec.schema_version,
            profile_id=spec.name,
            config_hash=spec_hash,
            generator_version=parent_meta.generator_version,
            git_commit=parent_meta.git_commit,
            input_hash=parent_meta.input_hash,
            created_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            extra={str(k): str(v) for k, v in extra.items()},
        )

    def observe(self, geom: GeomInput, spec: ObserverSpecV2) -> Obs2D:
        data = coerce_geom_to_label(geom, spec)

        axis = str(spec.params.get("axis", "z")).lower()
        coord_nm = self._param_float(spec.params, "coord_nm", 0.0)
        slab_thickness_nm = self._param_float(
            spec.params,
            "slab_thickness_nm",
            spec.mask_definition.slab_thickness_nm,
        )

        spacing_zyx = (
            float(data.grid.spacing[0]),
            float(data.grid.spacing[1]),
            float(data.grid.spacing[2]),
        )
        origin_zyx = (
            float(data.grid.origin[0]),
            float(data.grid.origin[1]),
            float(data.grid.origin[2]),
        )
        shape_zyx = (
            int(data.label_zyx.shape[0]),
            int(data.label_zyx.shape[1]),
            int(data.label_zyx.shape[2]),
        )

        center_index = self._slice_index(axis, coord_nm, spacing_zyx, origin_zyx, shape_zyx)
        spacing_axis, _ = self._axis_spacing_origin(axis, spacing_zyx, origin_zyx)
        radius = int(np.floor(max(0.0, slab_thickness_nm) / max(spacing_axis, 1e-12) / 2.0))

        index_min = max(0, center_index - radius)
        index_max = min(
            {"z": shape_zyx[0] - 1, "y": shape_zyx[1] - 1, "x": shape_zyx[2] - 1}[axis],
            center_index + radius,
        )
        indices = list(range(index_min, index_max + 1))

        mask2d: np.ndarray | None = None
        for idx in indices:
            label2d = self._extract_plane(data.label_zyx, axis, idx)
            current = mask2d_from_slice_label(
                label2d, spec.mask_definition, data.material
            ).astype(bool)
            if mask2d is None:
                mask2d = current
            else:
                mask2d |= current

        if mask2d is None:
            raise ValueError("slice extraction produced no planes")

        tsdf2d = tsdf2d_from_mask(
            mask2d,
            spec.target_grid_2d,
            mu_nm=spec.tsdf2d.mu_nm,
            backend=spec.tsdf2d.engine,
        )

        loops = []
        contour_notes: list[str] = []
        if spec.contour.resample_points != 0:
            try:
                loops = extract_contours_from_tsdf(
                    tsdf2d,
                    spec.target_grid_2d,
                    level=spec.contour.level,
                    resample_points=spec.contour.resample_points,
                    backend=spec.contour.backend,
                )
            except ObserverOptionalDependencyError:
                if spec.contour.allow_missing_backend:
                    contour_notes.append("contour_backend_missing")
                    loops = []
                else:
                    raise

        debug_maps: dict[str, np.ndarray] = {}
        debug_maps["slice_center_index"] = np.full(mask2d.shape, center_index, dtype=np.int32)
        debug_maps["slice_count"] = np.full(mask2d.shape, len(indices), dtype=np.int32)

        provisional_meta = self._build_meta(
            data.parent_meta,
            spec,
            source_kind=data.source_kind,
            axis=axis,
            center_index=center_index,
            slice_count=len(indices),
            qa_status="OK",
            qa_notes=contour_notes,
        )

        obs = Obs2D(
            grid2d=spec.target_grid_2d,
            mask=mask2d.astype(np.uint8, copy=False),
            tsdf=tsdf2d.astype(np.float32, copy=False),
            loops=loops,
            weight=None,
            transform=None,
            debug_maps=debug_maps,
            meta=provisional_meta,
        )

        qa = run_observer_qa(obs, spec)
        final_meta = self._build_meta(
            data.parent_meta,
            spec,
            source_kind=data.source_kind,
            axis=axis,
            center_index=center_index,
            slice_count=len(indices),
            qa_status=qa.status,
            qa_notes=qa.notes + contour_notes,
        )

        return Obs2D(
            grid2d=obs.grid2d,
            mask=obs.mask,
            tsdf=obs.tsdf,
            loops=obs.loops,
            weight=obs.weight,
            transform=obs.transform,
            debug_maps=obs.debug_maps,
            meta=final_meta,
        )
