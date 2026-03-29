from __future__ import annotations

from dataclasses import asdict
from typing import Any, cast

import numpy as np

from wafergeo.core.grid import AxisOrder, GridSpec, SampleLocation
from wafergeo.core.meta import Meta
from wafergeo.core.types import ContourLoop, Obs2D
from wafergeo.io.artifact_store import ArtifactStore
from wafergeo.sem.qa import SEMQA


def _serialize_loops(obs: Obs2D) -> list[dict[str, object]]:
    return [
        {
            "points_xy": loop.points_xy,
            "is_hole": loop.is_hole,
            "label": loop.label,
            "meta": dict(loop.meta),
        }
        for loop in obs.loops
    ]


def build_sem_obs_artifact_payload(
    obs: Obs2D,
    qa: SEMQA,
    extra_payload: dict[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "grid2d": asdict(obs.grid2d),
        "mask": obs.mask,
        "tsdf": obs.tsdf,
        "contours": _serialize_loops(obs),
        "weight": obs.weight,
        "transform": obs.transform,
        "debug_maps": dict(obs.debug_maps),
        "meta": obs.meta.to_dict(),
        "qa": asdict(qa),
    }
    payload.update(dict(extra_payload))
    return payload


def write_sem_obs_artifact(
    store: ArtifactStore,
    obs: Obs2D,
    qa: SEMQA,
    extra_payload: dict[str, object],
) -> str:
    payload: dict[str, Any] = build_sem_obs_artifact_payload(obs, qa, extra_payload)
    return store.write("sem_obs", payload, obs.meta)


def _to_grid2d(raw: dict[str, Any]) -> GridSpec:
    return GridSpec(
        dim=int(raw["dim"]),
        spacing=(float(raw["spacing"][0]), float(raw["spacing"][1])),
        origin=(float(raw["origin"][0]), float(raw["origin"][1])),
        axis_order=cast(AxisOrder, str(raw["axis_order"])),
        sample_location=cast(SampleLocation, str(raw["sample_location"])),
        units=str(raw["units"]),
    )


def _to_loops(raw: list[dict[str, Any]]) -> list[ContourLoop]:
    loops: list[ContourLoop] = []
    for row in raw:
        loops.append(
            ContourLoop(
                points_xy=np.asarray(row["points_xy"], dtype=np.float32),
                is_hole=bool(row["is_hole"]),
                label=str(row["label"]) if row.get("label") is not None else None,
                meta={str(k): v for k, v in dict(row.get("meta", {})).items()},
            )
        )
    return loops


def obs2d_from_sem_obs_payload(payload: dict[str, Any]) -> Obs2D:
    weight = payload.get("weight")
    transform = payload.get("transform")
    debug_maps_raw = dict(payload.get("debug_maps", {}))

    return Obs2D(
        grid2d=_to_grid2d(dict(payload["grid2d"])),
        mask=np.asarray(payload["mask"], dtype=np.uint8),
        tsdf=np.asarray(payload["tsdf"], dtype=np.float32),
        loops=_to_loops(list(payload.get("contours", []))),
        weight=None if weight is None else np.asarray(weight, dtype=np.float32),
        transform=None if transform is None else dict(transform),
        debug_maps={str(name): np.asarray(value) for name, value in debug_maps_raw.items()},
        meta=Meta.from_dict(dict(payload["meta"])),
    )


def read_sem_obs_artifact(store: ArtifactStore, artifact_id: str) -> tuple[Obs2D, dict[str, Any]]:
    payload = store.load(artifact_id)
    if not isinstance(payload, dict):
        raise ValueError("sem_obs payload must be a mapping")
    obs = obs2d_from_sem_obs_payload(payload)
    return obs, payload
