from __future__ import annotations

from tests.reports.helpers import write_run_index_json
from wafergeo.io.artifact_store import LocalDiskArtifactStore
from wafergeo.reports.registry import register_report_plot
from wafergeo.reports.runner import build_report
from wafergeo.reports.schema import PlotRequestSpec, PlotRunResult, ReportSpecV1


def test_heavy_plot_without_selector_records_fail_and_continues(tmp_path) -> None:
    class _HeavyPlot:
        name = "dummy.heavy"
        version = "1.0.0"
        required_tables: tuple[str, ...] = ()
        is_heavy = True

        def run(self, ctx, request):
            return PlotRunResult(plot_name=self.name, figure_paths=[], status="OK")

    class _NormalPlot:
        name = "dummy.normal"
        version = "1.0.0"
        required_tables: tuple[str, ...] = ()
        is_heavy = False

        def run(self, ctx, request):
            _ = (ctx, request)
            return PlotRunResult(plot_name=self.name, figure_paths=[], status="WARN")

    register_report_plot(_HeavyPlot())
    register_report_plot(_NormalPlot())

    run_index = write_run_index_json(
        tmp_path / "run_index.json",
        [{"sample_id": "s0", "group_id": "g0", "trial_artifact_id": "missing"}],
    )

    spec = ReportSpecV1(
        schema_version="report/v1",
        report_id="heavy_gate",
        title="Heavy Gate",
        run_index_path=str(run_index),
        plots=(
            PlotRequestSpec(name="dummy.heavy", params={}),
            PlotRequestSpec(name="dummy.normal", params={}),
        ),
    )

    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    manifest = build_report(spec, store, tmp_path / "out")
    assert len(manifest.figures) == 2
    assert manifest.figures[0].plot_name == "dummy.heavy"
    assert manifest.figures[0].status == "FAIL"
    assert any("heavy plot requires top_k or sample_ids" in v for v in manifest.figures[0].messages)
    assert manifest.figures[1].plot_name == "dummy.normal"
    assert manifest.figures[1].status == "WARN"
