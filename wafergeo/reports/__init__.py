"""Reports/Viz subsystem."""

from wafergeo.reports.registry import (
    get_report_extractor,
    get_report_plot,
    list_report_extractors,
    list_report_plots,
    register_report_extractor,
    register_report_plot,
)
from wafergeo.reports.runner import build_report, validate_report_inputs
from wafergeo.reports.schema import (
    FigureManifestEntry,
    HeavySelector,
    PlotRequestSpec,
    PlotRunResult,
    ReportManifestV1,
    ReportOutputSpec,
    ReportSpecV1,
    TableManifestEntry,
)
from wafergeo.reports.spec import load_report_spec_yaml, report_spec_hash

__all__ = [
    "HeavySelector",
    "PlotRequestSpec",
    "ReportOutputSpec",
    "ReportSpecV1",
    "PlotRunResult",
    "TableManifestEntry",
    "FigureManifestEntry",
    "ReportManifestV1",
    "load_report_spec_yaml",
    "report_spec_hash",
    "register_report_extractor",
    "get_report_extractor",
    "list_report_extractors",
    "register_report_plot",
    "get_report_plot",
    "list_report_plots",
    "validate_report_inputs",
    "build_report",
]
