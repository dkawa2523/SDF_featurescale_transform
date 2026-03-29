from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from wafergeo.core.grid import GridSpec
from wafergeo.core.hashing import hash_config
from wafergeo.core.meta import Meta
from wafergeo.core.types import Obs2D, TSDFVolume
from wafergeo.sdf.edt import signed_distance_from_mask
from wafergeo.sdf.tsdf import label_from_tsdf, to_tsdf


class DummyTopDownObserver:
    """Phase 2.1 temporary observer for benchmark-level 3D->2D comparison."""

    def observe_from_tsdf(
        self,
        tsdf_volume: TSDFVolume,
        *,
        mu_nm: float,
        backend: str,
    ) -> Obs2D:
        material = tsdf_volume.material
        void_index = material.ids.index(material.void_id)

        label_zyx = label_from_tsdf(
            tsdf_volume.tsdf,
            material,
            void_index=void_index,
            selected_material_ids=list(material.ids)[: tsdf_volume.tsdf.shape[0]],
        )
        mask2d = np.any(label_zyx != material.void_id, axis=0).astype(np.uint8)

        spacing_yx = (float(tsdf_volume.grid.spacing[1]), float(tsdf_volume.grid.spacing[2]))
        phi2d = signed_distance_from_mask(
            mask2d[np.newaxis, :, :].astype(bool),
            (1.0, spacing_yx[0], spacing_yx[1]),
            backend,
        )[0]
        tsdf2d = to_tsdf(phi2d, mu_nm=mu_nm, out_dtype=np.float32)

        grid2d = GridSpec(
            dim=2,
            spacing=spacing_yx,
            origin=(float(tsdf_volume.grid.origin[1]), float(tsdf_volume.grid.origin[2])),
            axis_order="YX",
            sample_location="cell_center",
            units=tsdf_volume.grid.units,
        )

        parent_meta = tsdf_volume.meta
        if parent_meta is None:
            generator_version = "0.1.0"
            git_commit = "unknown"
            input_hash = "unknown"
        else:
            generator_version = parent_meta.generator_version
            git_commit = parent_meta.git_commit
            input_hash = parent_meta.input_hash

        config_hash = hash_config({"kind": "dummy_topdown", "mu_nm": mu_nm, "backend": backend})
        meta = Meta(
            schema_version="observer/v2",
            profile_id="dummy_topdown_v1",
            config_hash=config_hash,
            generator_version=generator_version,
            git_commit=git_commit,
            input_hash=input_hash,
            created_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            extra={"observer_kind": "dummy_topdown"},
        )

        return Obs2D(
            grid2d=grid2d,
            mask=mask2d,
            tsdf=tsdf2d,
            loops=[],
            weight=None,
            transform=None,
            debug_maps={},
            meta=meta,
        )
