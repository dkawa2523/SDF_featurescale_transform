from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from wafergeo.core.meta import Meta
from wafergeo.core.types import Obs2D
from wafergeo.observe.base import GeomInput, coerce_geom_to_label
from wafergeo.observe.contour_extract import extract_contours_from_tsdf
from wafergeo.observe.errors import ObserverOptionalDependencyError
from wafergeo.observe.mask_def import mask2d_from_exposed_id
from wafergeo.observe.qa import run_observer_qa
from wafergeo.observe.spec import ObserverSpecV2, observer_spec_hash
from wafergeo.observe.tsdf2d import tsdf2d_from_mask


class TopDownExposedObserver:
    kind = "topdown_exposed"

    def _build_ignore_ids(self, material, spec: ObserverSpecV2) -> set[int]:
        ignore_ids = {
            int(material_id)
            for material_id, ignore in zip(
                material.ids, material.ignore_in_exposure, strict=True
            )
            if ignore
        }

        name_to_id = {
            str(name): int(material_id)
            for material_id, name in zip(material.ids, material.names, strict=True)
        }
        for name in spec.mask_definition.ignore_materials:
            if name in name_to_id:
                ignore_ids.add(name_to_id[name])
        for material_id in spec.mask_definition.ignore_ids:
            ignore_ids.add(int(material_id))
        return ignore_ids

    def _compute_exposed_maps(
        self,
        label_zyx: np.ndarray,
        *,
        spacing_zyx: tuple[float, float, float],
        origin_zyx: tuple[float, float, float],
        ignore_ids: set[int],
        void_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        z_size = label_zyx.shape[0]
        valid = ~np.isin(label_zyx, np.asarray(sorted(ignore_ids), dtype=np.int32))
        valid_rev = valid[::-1, :, :]

        any_hit = np.any(valid_rev, axis=0)
        hit_idx_from_top = np.argmax(valid_rev, axis=0)
        z_hit = (z_size - 1) - hit_idx_from_top

        exposed_id = np.full(label_zyx.shape[1:], int(void_id), dtype=np.int32)
        height_map = np.full(label_zyx.shape[1:], np.nan, dtype=np.float32)

        y_idx, x_idx = np.nonzero(any_hit)
        if y_idx.size > 0:
            z_idx = z_hit[any_hit]
            exposed_id[any_hit] = label_zyx[z_idx, y_idx, x_idx]
            height_nm = float(origin_zyx[0]) + z_idx.astype(np.float32) * float(spacing_zyx[0])
            height_map[any_hit] = height_nm

        return exposed_id, height_map

    def _build_meta(
        self,
        parent_meta: Meta,
        spec: ObserverSpecV2,
        *,
        source_kind: str,
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
        ignore_ids = self._build_ignore_ids(data.material, spec)

        exposed_id, height_map = self._compute_exposed_maps(
            data.label_zyx,
            spacing_zyx=spacing_zyx,
            origin_zyx=origin_zyx,
            ignore_ids=ignore_ids,
            void_id=int(data.material.void_id),
        )

        mask2d = mask2d_from_exposed_id(exposed_id, spec.mask_definition, data.material)
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
        if bool(spec.debug.get("save_exposed_id", True)):
            debug_maps["exposed_id"] = exposed_id.astype(np.int32, copy=False)
        if bool(spec.debug.get("save_height_map", True)):
            height_out = np.nan_to_num(height_map, nan=-1.0).astype(np.float32, copy=False)
            debug_maps["height_map_nm"] = height_out

        provisional_meta = self._build_meta(
            data.parent_meta,
            spec,
            source_kind=data.source_kind,
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
