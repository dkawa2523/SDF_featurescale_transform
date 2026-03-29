from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from wafergeo.reports.schema import PlotRequestSpec, PlotRunResult

if TYPE_CHECKING:
    from wafergeo.reports.context import ReportContext


class PlotTaskProtocol(Protocol):
    name: str
    version: str
    required_tables: tuple[str, ...]
    is_heavy: bool

    def run(self, ctx: ReportContext, request: PlotRequestSpec) -> PlotRunResult:
        ...
