from __future__ import annotations

import tracemalloc
from dataclasses import replace
from time import perf_counter

import numpy as np

from wafergeo.core.types import Obs2D
from wafergeo.sdf.bench.observer_dummy import DummyTopDownObserver
from wafergeo.sdf.bench.types import (
    BenchmarkCase,
    BenchmarkReport,
    EngineRunResult,
    Obs2DDelta,
    QADelta,
)
from wafergeo.sdf.build import build_tsdf_volume
from wafergeo.sdf.engines.registry import get_sdf_engine


def _mask_iou(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_bool = lhs.astype(bool)
    rhs_bool = rhs.astype(bool)
    inter = np.logical_and(lhs_bool, rhs_bool).sum()
    union = np.logical_or(lhs_bool, rhs_bool).sum()
    if int(union) == 0:
        return 1.0
    return float(inter) / float(union)


def _make_qa_delta(current: EngineRunResult, reference: EngineRunResult) -> QADelta | None:
    if current.qa is None or reference.qa is None:
        return None
    return QADelta(
        tsdf_min_abs_diff=abs(current.qa.tsdf_min - reference.qa.tsdf_min),
        tsdf_max_abs_diff=abs(current.qa.tsdf_max - reference.qa.tsdf_max),
        band_fraction_abs_diff=abs(current.qa.band_fraction - reference.qa.band_fraction),
        grad_error_abs_diff=abs(
            current.qa.grad_unit_error_rate - reference.qa.grad_unit_error_rate
        ),
    )


def _find_reference(
    runs: list[EngineRunResult],
    explicit_ref: str | None,
) -> EngineRunResult | None:
    if explicit_ref is not None:
        for run in runs:
            if run.engine_name == explicit_ref and run.error_message is None:
                return run
        return None

    for run in runs:
        if run.error_message is None:
            return run
    return None


def benchmark_engines_on_label(case: BenchmarkCase) -> BenchmarkReport:
    if not case.engine_names:
        raise ValueError("engine_names must be non-empty")

    observer = DummyTopDownObserver()
    preliminary: list[tuple[EngineRunResult, Obs2D | None]] = []

    for engine_name in case.engine_names:
        runtime_sec = 0.0
        peak_memory_bytes = 0
        qa = None
        obs = None
        error_message: str | None = None

        try:
            engine = get_sdf_engine(engine_name)
            engine_version = engine.version
        except Exception as exc:
            preliminary.append(
                (
                    EngineRunResult(
                        engine_name=engine_name,
                        engine_version="unknown",
                        runtime_sec=runtime_sec,
                        peak_memory_bytes=peak_memory_bytes,
                        qa=qa,
                        qa_delta_vs_ref=None,
                        obs2d_delta_vs_ref=None,
                        error_message=str(exc),
                    ),
                    None,
                )
            )
            continue

        cfg = replace(case.base_config, backend=engine_name)

        tracemalloc.start()
        start = perf_counter()
        try:
            tsdf_volume, qa = build_tsdf_volume(case.label, cfg)
            observer_mu = float(
                case.observer_mu_nm if case.observer_mu_nm is not None else cfg.mu_nm
            )
            observer_backend = case.observer_backend or engine_name
            obs = observer.observe_from_tsdf(
                tsdf_volume,
                mu_nm=observer_mu,
                backend=observer_backend,
            )
        except Exception as exc:
            error_message = str(exc)
        finally:
            runtime_sec = perf_counter() - start
            _, peak_memory_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        preliminary.append(
            (
                EngineRunResult(
                    engine_name=engine_name,
                    engine_version=engine_version,
                    runtime_sec=runtime_sec,
                    peak_memory_bytes=int(peak_memory_bytes),
                    qa=qa,
                    qa_delta_vs_ref=None,
                    obs2d_delta_vs_ref=None,
                    error_message=error_message,
                ),
                obs,
            )
        )

    raw_runs = [run for run, _ in preliminary]
    reference = _find_reference(raw_runs, case.reference_engine)

    final_runs: list[EngineRunResult] = []
    ref_obs = None
    if reference is not None:
        for run, obs in preliminary:
            if run.engine_name == reference.engine_name and run.error_message is None:
                ref_obs = obs
                break

    for run, obs in preliminary:
        qa_delta = None
        obs_delta = None
        if reference is not None and run.error_message is None and reference.error_message is None:
            qa_delta = _make_qa_delta(run, reference)
            if obs is not None and ref_obs is not None:
                tsdf_l1 = float(np.mean(np.abs(obs.tsdf - ref_obs.tsdf)))
                obs_delta = Obs2DDelta(
                    mask_iou=_mask_iou(obs.mask, ref_obs.mask),
                    tsdf_l1=tsdf_l1,
                    contour_count_delta=abs(len(obs.loops) - len(ref_obs.loops)),
                )

        final_runs.append(
            EngineRunResult(
                engine_name=run.engine_name,
                engine_version=run.engine_version,
                runtime_sec=run.runtime_sec,
                peak_memory_bytes=run.peak_memory_bytes,
                qa=run.qa,
                qa_delta_vs_ref=qa_delta,
                obs2d_delta_vs_ref=obs_delta,
                error_message=run.error_message,
            )
        )

    ref_name = reference.engine_name if reference is not None else None
    return BenchmarkReport(reference_engine=ref_name, runs=final_runs)
