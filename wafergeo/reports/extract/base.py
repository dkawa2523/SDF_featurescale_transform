from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pandas as pd

    from wafergeo.reports.context import ReportContext


class TableExtractorProtocol(Protocol):
    name: str
    version: str
    produces_table: str
    required_tables: tuple[str, ...]

    def build(self, ctx: ReportContext) -> pd.DataFrame:
        ...
