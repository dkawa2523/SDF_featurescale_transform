from __future__ import annotations

import numpy as np

from wafergeo.core.types import Status
from wafergeo.metrics.base import MetricResult

_STATUS_RANK: dict[Status, int] = {"OK": 0, "WARN": 1, "FAIL": 2}


def combine_status(a: Status, b: Status) -> Status:
    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


def fail_result(
    name: str,
    version: str,
    *,
    fail_penalty: float,
    message: str,
    report: dict[str, object] | None = None,
    maps: dict[str, np.ndarray] | None = None,
    meta: dict[str, object] | None = None,
) -> MetricResult:
    return MetricResult(
        name=name,
        version=version,
        loss=float(fail_penalty),
        report=dict(report or {}),
        maps=dict(maps or {}),
        status="FAIL",
        messages=[message],
        meta=dict(meta or {}),
    )


def ensure_finite_or_fail(
    result: MetricResult,
    *,
    fail_penalty: float,
    message_prefix: str,
) -> MetricResult:
    if np.isfinite(result.loss):
        return result
    return fail_result(
        result.name,
        result.version,
        fail_penalty=fail_penalty,
        message=f"{message_prefix}: non-finite loss",
        report=result.report,
        maps=result.maps,
        meta=result.meta,
    )


def to_message(err: Exception | str) -> str:
    if isinstance(err, Exception):
        return f"{type(err).__name__}: {err}"
    return str(err)
