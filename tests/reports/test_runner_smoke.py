from __future__ import annotations

import importlib.util

import pytest

from tests.reports.helpers import write_assim_trial, write_run_index_json
from wafergeo.io.artifact_store import LocalDiskArtifactStore
from wafergeo.reports.registry import register_report_plot
from wafergeo.reports.runner import build_report
from wafergeo.reports.schema import (
    HeavySelector,
    PlotRequestSpec,
    PlotRunResult,
    ReportOutputSpec,
    ReportSpecV1,
)


def _require_viz_deps() -> None:
    missing = [
        name
        for name in ("pandas", "matplotlib", "pyarrow")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        pytest.skip("missing optional dependencies: " + ", ".join(missing))



def _build_spec(
    run_index_path: str,
    *,
    html: bool = False,
    write_tables_parquet: bool = True,
) -> ReportSpecV1:
    return ReportSpecV1(
        schema_version="report/v1",
        report_id="smoke_report",
        title="Smoke Report",
        run_index_path=run_index_path,
        output=ReportOutputSpec(
            write_png=True,
            write_svg=False,
            write_tables_parquet=write_tables_parquet,
            write_html_index=html,
            dpi=120,
        ),
        plots=(
            PlotRequestSpec(name="metrics.loss_breakdown", params={}),
            PlotRequestSpec(
                name="metrics.residual_map",
                params={},
                heavy_selector=HeavySelector(top_k=1),
            ),
        ),
    )



def test_report_runner_smoke_outputs_png_parquet_manifest(tmp_path) -> None:
    _require_viz_deps()

    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    trial_a = write_assim_trial(store, seed=0, total_loss=0.2, metric_loss=0.2)
    trial_b = write_assim_trial(store, seed=1, total_loss=0.8, metric_loss=0.8)

    run_index = write_run_index_json(
        tmp_path / "run_index.json",
        [
            {"sample_id": "s0", "group_id": "g0", "trial_artifact_id": trial_a},
            {"sample_id": "s1", "group_id": "g1", "trial_artifact_id": trial_b},
        ],
    )

    spec = _build_spec(str(run_index), html=True)
    out_dir = tmp_path / "report"
    manifest = build_report(spec, store, out_dir)

    assert (out_dir / "report_manifest.json").exists()
    assert manifest.spec_hash
    assert manifest.index_hash
    assert manifest.code_version

    assert len(manifest.tables) >= 1
    assert len(manifest.figures) >= 1
    for table in manifest.tables:
        if table.path:
            assert (out_dir / table.path).exists()
    all_figs = [path for row in manifest.figures for path in row.figure_paths]
    assert all_figs
    assert any(path.endswith(".png") for path in all_figs)
    for path in all_figs:
        assert (out_dir / path).exists()

    assert (out_dir / "html" / "index.html").exists()



def test_report_runner_uses_cache_on_second_run(tmp_path) -> None:
    _require_viz_deps()

    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    trial_id = write_assim_trial(store, seed=3, total_loss=0.4, metric_loss=0.4)
    run_index = write_run_index_json(
        tmp_path / "run_index.json",
        [{"sample_id": "s0", "group_id": "g0", "trial_artifact_id": trial_id}],
    )

    spec = _build_spec(str(run_index), html=False)
    out_dir = tmp_path / "report"

    first = build_report(spec, store, out_dir)
    second = build_report(spec, store, out_dir)

    assert all(not row.cache_hit for row in first.tables)
    assert all(not row.cache_hit for row in first.figures)
    assert all(row.cache_hit for row in second.tables)
    assert all(row.cache_hit for row in second.figures)


def test_report_runner_cache_preserves_warn_status(tmp_path) -> None:
    class _WarnNoFigurePlot:
        name = "dummy.warn_cache"
        version = "1.0.0"
        required_tables: tuple[str, ...] = ()
        is_heavy = False

        def run(self, ctx, request):
            _ = (ctx, request)
            return PlotRunResult(
                plot_name=self.name,
                figure_paths=[],
                status="WARN",
                messages=["no data"],
            )

    register_report_plot(_WarnNoFigurePlot())

    run_index = write_run_index_json(
        tmp_path / "run_index.json",
        [{"sample_id": "s0", "group_id": "g0", "trial_artifact_id": "missing"}],
    )
    spec = ReportSpecV1(
        schema_version="report/v1",
        report_id="warn_cache",
        title="Warn Cache",
        run_index_path=str(run_index),
        output=ReportOutputSpec(write_png=True, write_svg=False, write_tables_parquet=False),
        plots=(PlotRequestSpec(name="dummy.warn_cache", params={}),),
    )
    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")

    first = build_report(spec, store, tmp_path / "out")
    second = build_report(spec, store, tmp_path / "out")
    assert first.figures[0].status == "WARN"
    assert second.figures[0].cache_hit is True
    assert second.figures[0].status == "WARN"
    assert second.figures[0].messages == ["no data"]


def test_report_runner_sanitizes_unsafe_suffix(tmp_path) -> None:
    class _UnsafeSuffixPlot:
        name = "dummy.unsafe_suffix"
        version = "1.0.0"
        required_tables: tuple[str, ...] = ()
        is_heavy = False

        def run(self, ctx, request):
            _ = request
            base = ctx.figure_base(self.name, suffix="1_a/b c")
            rel = f"figures/{base}.txt"
            out = ctx.out_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("ok", encoding="utf-8")
            return PlotRunResult(plot_name=self.name, figure_paths=[rel], status="OK")

    register_report_plot(_UnsafeSuffixPlot())
    run_index = write_run_index_json(
        tmp_path / "run_index.json",
        [{"sample_id": "s0", "group_id": "g0", "trial_artifact_id": "missing"}],
    )
    spec = ReportSpecV1(
        schema_version="report/v1",
        report_id="unsafe_suffix",
        title="Unsafe Suffix",
        run_index_path=str(run_index),
        output=ReportOutputSpec(write_png=True, write_svg=False, write_tables_parquet=False),
        plots=(PlotRequestSpec(name="dummy.unsafe_suffix", params={}),),
    )
    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    manifest = build_report(spec, store, tmp_path / "out")
    rel = manifest.figures[0].figure_paths[0]
    assert "/" in rel
    assert "a_b_c" in rel
    assert "a/b c" not in rel
    assert (tmp_path / "out" / rel).exists()


def test_report_runner_respects_write_tables_parquet_false(tmp_path) -> None:
    _require_viz_deps()
    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    trial_id = write_assim_trial(store, seed=7, total_loss=0.3, metric_loss=0.3)
    run_index = write_run_index_json(
        tmp_path / "run_index.json",
        [{"sample_id": "s0", "group_id": "g0", "trial_artifact_id": trial_id}],
    )
    spec = _build_spec(str(run_index), html=False, write_tables_parquet=False)
    out_dir = tmp_path / "report"
    manifest = build_report(spec, store, out_dir)
    assert len(manifest.tables) >= 1
    assert all(row.path == "" for row in manifest.tables)
    assert list((out_dir / "tables").glob("*.parquet")) == []


def test_metrics_long_table_is_parquet_compatible_shape(tmp_path) -> None:
    _require_viz_deps()
    import pandas as pd  # type: ignore[import-not-found]

    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    trial_id = write_assim_trial(store, seed=9, total_loss=0.9, metric_loss=0.9)
    run_index = write_run_index_json(
        tmp_path / "run_index.json",
        [{"sample_id": "s0", "group_id": "g0", "trial_artifact_id": trial_id}],
    )
    spec = _build_spec(str(run_index), html=False, write_tables_parquet=True)
    out_dir = tmp_path / "report"
    manifest = build_report(spec, store, out_dir)
    table_paths = {
        row.table_name: out_dir / row.path for row in manifest.tables if row.path
    }
    long_df = pd.read_parquet(table_paths["metrics.long"], engine="pyarrow")
    maps_df = pd.read_parquet(table_paths["metrics.maps"], engine="pyarrow")
    assert "report_json" in long_df.columns
    assert "maps" not in long_df.columns
    assert "report" not in long_df.columns
    assert "map_path" in maps_df.columns
    map_path = maps_df.iloc[0]["map_path"]
    assert isinstance(map_path, str)
    assert (out_dir / map_path).exists()
