from __future__ import annotations

from dataclasses import dataclass

from wafergeo.core.types import LabelVolume
from wafergeo.sdf.config import SDFBuildConfig
from wafergeo.sdf.qa import SDFQA


@dataclass(frozen=True)
class QADelta:
    tsdf_min_abs_diff: float
    tsdf_max_abs_diff: float
    band_fraction_abs_diff: float
    grad_error_abs_diff: float


@dataclass(frozen=True)
class Obs2DDelta:
    mask_iou: float
    tsdf_l1: float
    contour_count_delta: int


@dataclass(frozen=True)
class EngineRunResult:
    engine_name: str
    engine_version: str
    runtime_sec: float
    peak_memory_bytes: int
    qa: SDFQA | None
    qa_delta_vs_ref: QADelta | None
    obs2d_delta_vs_ref: Obs2DDelta | None
    error_message: str | None = None


@dataclass(frozen=True)
class BenchmarkCase:
    label: LabelVolume
    base_config: SDFBuildConfig
    engine_names: tuple[str, ...]
    reference_engine: str | None = None
    observer_mu_nm: float | None = None
    observer_backend: str | None = None


@dataclass(frozen=True)
class BenchmarkReport:
    reference_engine: str | None
    runs: list[EngineRunResult]
