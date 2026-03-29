from __future__ import annotations

from typing import Any

import numpy as np


def robust_loss(residual: np.ndarray, robust_spec: dict[str, Any] | None) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float32)
    spec = dict(robust_spec or {})
    kind = str(spec.get("type", "huber")).lower()

    if kind == "l1":
        return np.abs(values)

    if kind == "l2":
        return 0.5 * values * values

    if kind == "huber":
        delta = float(spec.get("delta", 0.1))
        if delta <= 0.0:
            raise ValueError(f"huber delta must be > 0, got {delta}")
        abs_v = np.abs(values)
        quad = 0.5 * values * values
        linear = delta * (abs_v - 0.5 * delta)
        return np.where(abs_v <= delta, quad, linear).astype(np.float32, copy=False)

    if kind == "cauchy":
        c = float(spec.get("c", 1.0))
        if c <= 0.0:
            raise ValueError(f"cauchy c must be > 0, got {c}")
        ratio = values / c
        return (0.5 * (c * c) * np.log1p(ratio * ratio)).astype(np.float32, copy=False)

    raise ValueError(f"unsupported robust type: {kind}")
