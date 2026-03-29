from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FailurePolicy:
    out_of_bounds: Literal["clamp", "penalty", "fail"] = "clamp"
    penalty: float = 1e6
    on_surrogate_exception: Literal["penalty", "fail"] = "penalty"
    on_observer_exception: Literal["penalty", "fail"] = "penalty"

    def __post_init__(self) -> None:
        if self.penalty <= 0.0:
            raise ValueError(f"penalty must be >0, got {self.penalty}")


@dataclass(frozen=True)
class TransformPolicy:
    mode: Literal["strict_sim_grid", "apply_sem_to_sim", "compare_in_sem"] = "strict_sim_grid"


@dataclass(frozen=True)
class LoggingPolicy:
    mode: Literal["none", "best_only", "periodic", "all"] = "none"
    period: int = 10
    save_pred_obs: bool = False

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError(f"period must be >0, got {self.period}")
