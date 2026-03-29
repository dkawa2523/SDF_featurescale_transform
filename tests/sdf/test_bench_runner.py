from __future__ import annotations

import numpy as np

from tests.sdf.helpers import build_label_volume, register_bruteforce_engine
from wafergeo.sdf.bench.runner import benchmark_engines_on_label
from wafergeo.sdf.bench.types import BenchmarkCase
from wafergeo.sdf.config import SDFBuildConfig


def _label() -> np.ndarray:
    data = np.zeros((1, 4, 4), dtype=np.uint8)
    data[:, :, :2] = 1
    data[:, :, 2:] = 2
    return data


def test_benchmark_runner_collects_time_memory_qa_and_obs2d_deltas() -> None:
    e1 = register_bruteforce_engine("bench_engine_a")
    e2 = register_bruteforce_engine("bench_engine_b")
    label = build_label_volume(_label())

    report = benchmark_engines_on_label(
        BenchmarkCase(
            label=label,
            base_config=SDFBuildConfig(mu_nm=20.0, backend=e1),
            engine_names=(e1, e2),
            reference_engine=e1,
            observer_backend=e1,
        )
    )

    assert report.reference_engine == e1
    assert len(report.runs) == 2
    for run in report.runs:
        assert run.runtime_sec >= 0.0
        assert run.peak_memory_bytes >= 0
        assert run.error_message is None
        assert run.qa is not None
        assert run.qa_delta_vs_ref is not None
        assert run.obs2d_delta_vs_ref is not None


def test_benchmark_runner_keeps_results_when_one_engine_fails() -> None:
    ok_engine = register_bruteforce_engine("bench_engine_ok")
    label = build_label_volume(_label())

    report = benchmark_engines_on_label(
        BenchmarkCase(
            label=label,
            base_config=SDFBuildConfig(mu_nm=20.0, backend=ok_engine),
            engine_names=(ok_engine, "itk_maurer"),
            reference_engine=ok_engine,
            observer_backend=ok_engine,
        )
    )

    assert len(report.runs) == 2
    ok_run = next(run for run in report.runs if run.engine_name == ok_engine)
    fail_run = next(run for run in report.runs if run.engine_name == "itk_maurer")

    assert ok_run.error_message is None
    assert ok_run.qa is not None
    assert fail_run.error_message is not None
