from __future__ import annotations

from typing import cast

from wafergeo.core.registry import report_extractor_registry, report_plot_registry
from wafergeo.reports.errors import ReportRegistryError
from wafergeo.reports.extract.base import TableExtractorProtocol
from wafergeo.reports.extract.index_build import RunIndexExtractor
from wafergeo.reports.extract.metrics_tables import MetricsLongExtractor, MetricsMapsExtractor
from wafergeo.reports.plots.base import PlotTaskProtocol
from wafergeo.reports.plots.metrics.loss_breakdown import LossBreakdownPlot
from wafergeo.reports.plots.metrics.residual_map import ResidualMapPlot


def register_report_extractor(
    extractor: TableExtractorProtocol,
    *,
    aliases: tuple[str, ...] = (),
) -> None:
    report_extractor_registry.register(extractor.produces_table, extractor, override=True)
    for alias in aliases:
        report_extractor_registry.register(alias, extractor, override=True)



def get_report_extractor(name: str) -> TableExtractorProtocol:
    register_default_report_components()
    try:
        raw = report_extractor_registry.get(name)
    except KeyError as exc:
        raise ReportRegistryError(f"unknown report extractor: {name}") from exc
    return cast(TableExtractorProtocol, raw)



def list_report_extractors() -> tuple[str, ...]:
    register_default_report_components()
    return report_extractor_registry.list()



def register_report_plot(
    plot: PlotTaskProtocol,
    *,
    aliases: tuple[str, ...] = (),
) -> None:
    report_plot_registry.register(plot.name, plot, override=True)
    for alias in aliases:
        report_plot_registry.register(alias, plot, override=True)



def get_report_plot(name: str) -> PlotTaskProtocol:
    register_default_report_components()
    try:
        raw = report_plot_registry.get(name)
    except KeyError as exc:
        raise ReportRegistryError(f"unknown report plot: {name}") from exc
    return cast(PlotTaskProtocol, raw)



def list_report_plots() -> tuple[str, ...]:
    register_default_report_components()
    return report_plot_registry.list()



def register_default_report_components() -> None:
    if "index.run_index" not in report_extractor_registry.list():
        register_report_extractor(RunIndexExtractor())
    if "metrics.long" not in report_extractor_registry.list():
        register_report_extractor(MetricsLongExtractor())
    if "metrics.maps" not in report_extractor_registry.list():
        register_report_extractor(MetricsMapsExtractor())

    if "metrics.loss_breakdown" not in report_plot_registry.list():
        register_report_plot(LossBreakdownPlot())
    if "metrics.residual_map" not in report_plot_registry.list():
        register_report_plot(ResidualMapPlot())
