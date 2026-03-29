from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wafergeo.core.meta import Meta
from wafergeo.io.artifact_store import LocalDiskArtifactStore


def write_run_index_json(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"rows": rows}, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path



def _meta(seed: int) -> Meta:
    return Meta(
        schema_version="assim_trial/v1",
        profile_id="report_test",
        config_hash=f"cfg_{seed}",
        generator_version="0.1.0",
        git_commit="deadbeef",
        input_hash=f"input_{seed}",
        created_at=datetime.now(UTC).isoformat(),
        extra={},
    )



def write_assim_trial(
    store: LocalDiskArtifactStore,
    *,
    seed: int,
    total_loss: float,
    metric_loss: float,
) -> str:
    payload = {
        "schema_version": "assim_trial/v1",
        "case_id": "case_report",
        "candidate_id": f"cand_{seed}",
        "x": np.asarray([float(seed)], dtype=np.float64),
        "params": {"shift_x": seed},
        "total_loss": float(total_loss),
        "per_observer": {"topdown": float(metric_loss)},
        "status": "OK",
        "messages": [],
        "timings": {"total": 0.01},
        "metric_results": [
            {
                "name": "tsdf_band_robust_weight",
                "version": "1.0.0",
                "loss": float(metric_loss),
                "status": "OK",
                "messages": [],
                "report": {"n": 1},
                "maps": {"residual": np.full((8, 8), metric_loss, dtype=np.float32)},
                "meta": {"observer": "topdown"},
            }
        ],
        "artifacts": {},
    }
    return store.write("assim_trial", payload, _meta(seed))
