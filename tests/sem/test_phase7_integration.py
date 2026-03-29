from __future__ import annotations

import numpy as np

from tests.sdf.helpers import register_bruteforce_engine
from tests.sem.helpers import build_sem_spec
from wafergeo.core.meta import Meta
from wafergeo.core.types import ContourLoop, Obs2D
from wafergeo.io.artifact_store import LocalDiskArtifactStore
from wafergeo.metrics.aggregate import build_metric_context, compute_objective
from wafergeo.metrics.spec import MetricEntrySpec, MetricSpecV2
from wafergeo.sem.artifact import read_sem_obs_artifact, write_sem_obs_artifact
from wafergeo.sem.build_obs import build_sem_obs2d
from wafergeo.sem.normalize import build_transform_chain, normalize_contours
from wafergeo.sem.readers import RawContourLoop, RawContourSet


def _closed_square_raw() -> RawContourSet:
    return RawContourSet(
        coord_system="nm",
        units="nm",
        loops_raw=[
            RawContourLoop(
                loop_id="outer_0",
                role="outer",
                points_xy=np.array(
                    [[30.0, 30.0], [110.0, 30.0], [110.0, 110.0], [30.0, 110.0], [30.0, 30.0]],
                    dtype=np.float32,
                ),
                is_closed_hint=True,
            )
        ],
    )


def _shift_obs_x(obs: Obs2D, shift_px: int) -> Obs2D:
    sx = float(obs.grid2d.spacing[1])
    shifted_loops = [
        ContourLoop(
            points_xy=(loop.points_xy + np.array([shift_px * sx, 0.0], dtype=np.float32)).astype(
                np.float32,
                copy=False,
            ),
            is_hole=loop.is_hole,
            label=loop.label,
            meta=dict(loop.meta),
        )
        for loop in obs.loops
    ]
    shifted_meta = Meta.from_dict(obs.meta.to_dict())
    return Obs2D(
        grid2d=obs.grid2d,
        mask=np.roll(obs.mask, shift=shift_px, axis=1).astype(np.uint8, copy=False),
        tsdf=np.roll(obs.tsdf, shift=shift_px, axis=1).astype(np.float32, copy=False),
        loops=shifted_loops,
        weight=None if obs.weight is None else np.roll(obs.weight, shift=shift_px, axis=1),
        transform=obs.transform,
        debug_maps={k: np.roll(v, shift=shift_px, axis=1) for k, v in obs.debug_maps.items()},
        meta=shifted_meta,
    )


def test_sem_obs_target_shape_yx_fixes_comparison_grid() -> None:
    backend = register_bruteforce_engine("brute_sem_shape")
    spec = build_sem_spec(
        coord_system="nm",
        units="nm",
        tsdf_mode="signed_region",
        distance_backend=backend,
        target_shape_yx=(32, 48),
    )
    chain = build_transform_chain(spec, image_shape=None)
    normalized = normalize_contours(_closed_square_raw(), spec, chain)
    obs, _, _ = build_sem_obs2d(
        normalized,
        spec,
        source_contour_path="contours.json",
        source_image_path=None,
        image_raw=None,
        transform_chain=chain,
        input_hash="shape_hash",
        generator_version="0.1.0",
        git_commit="deadbeef",
    )
    assert obs.mask.shape == (32, 48)
    assert obs.tsdf.shape == (32, 48)


def test_sem_obs_artifact_is_usable_for_phase7_metrics_objective(tmp_path) -> None:
    backend = register_bruteforce_engine("brute_sem_phase7")
    spec = build_sem_spec(
        coord_system="nm",
        units="nm",
        tsdf_mode="signed_region",
        distance_backend=backend,
        target_shape_yx=(40, 40),
    )
    chain = build_transform_chain(spec, image_shape=None)
    normalized = normalize_contours(_closed_square_raw(), spec, chain)
    obs, qa, extra_payload = build_sem_obs2d(
        normalized,
        spec,
        source_contour_path="contours.json",
        source_image_path=None,
        image_raw=None,
        transform_chain=chain,
        input_hash="phase7_hash",
        generator_version="0.1.0",
        git_commit="deadbeef",
    )

    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    artifact_id = write_sem_obs_artifact(store, obs, qa, extra_payload)
    sem_obs_ids = {"topdown": artifact_id}

    obs_by_observer: dict[str, Obs2D] = {}
    for observer_name, sem_artifact_id in sem_obs_ids.items():
        loaded_obs, _ = read_sem_obs_artifact(store, sem_artifact_id)
        obs_by_observer[observer_name] = loaded_obs

    metric_spec = MetricSpecV2(
        schema_version="metric/v2",
        metric_set_id="assim_phase7_contract",
        fail_penalty=1e6,
        observer_weights={"topdown": 1.0},
        metrics=(
            MetricEntrySpec(
                name="tsdf_band_robust_weight",
                weight=1.0,
                observers=("topdown",),
                params={
                    "band_mode": "obs_band",
                    "robust": {"type": "l2"},
                    "weight_mode": "uniform",
                },
            ),
        ),
    )

    ctx = build_metric_context(obs_by_observer, metric_spec, {})
    same = compute_objective(obs_by_observer, obs_by_observer, metric_spec, ctx)
    shifted_pred = {"topdown": _shift_obs_x(obs_by_observer["topdown"], shift_px=1)}
    shifted = compute_objective(shifted_pred, obs_by_observer, metric_spec, ctx)

    assert same.status == "OK"
    assert same.total_loss < 1e-6
    assert shifted.total_loss > same.total_loss
