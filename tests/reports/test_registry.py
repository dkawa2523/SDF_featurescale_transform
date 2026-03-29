from __future__ import annotations

from wafergeo.reports.registry import (
    get_report_extractor,
    get_report_plot,
    list_report_extractors,
    list_report_plots,
    register_report_extractor,
    register_report_plot,
)
from wafergeo.reports.schema import PlotRunResult


def test_report_registry_allows_custom_plugins() -> None:
    class _DummyExtractor:
        name = "dummy_extractor"
        version = "1.0.0"
        produces_table = "dummy.table"
        required_tables: tuple[str, ...] = ()

        def build(self, ctx):
            return ctx.tables["index.run_index"]

    class _DummyPlot:
        name = "dummy.plot"
        version = "1.0.0"
        required_tables: tuple[str, ...] = ()
        is_heavy = False

        def run(self, ctx, request):
            return PlotRunResult(plot_name=self.name, figure_paths=[], status="OK")

    register_report_extractor(_DummyExtractor())
    register_report_plot(_DummyPlot())

    ext = get_report_extractor("dummy.table")
    plot = get_report_plot("dummy.plot")
    assert ext.version == "1.0.0"
    assert plot.version == "1.0.0"
    assert "dummy.table" in list_report_extractors()
    assert "dummy.plot" in list_report_plots()
