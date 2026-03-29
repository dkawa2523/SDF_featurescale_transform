from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wafergeo.io.artifact_store import ArtifactStore
from wafergeo.reports.cache import sanitize_name
from wafergeo.reports.schema import FigureManifestEntry, ReportSpecV1, TableManifestEntry


@dataclass
class ReportContext:
    store: ArtifactStore
    out_dir: Path
    spec: ReportSpecV1
    spec_hash: str
    index_hash: str
    code_version: str
    tables: dict[str, object] = field(default_factory=dict)
    table_entries: dict[str, TableManifestEntry] = field(default_factory=dict)
    figure_entries: list[FigureManifestEntry] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    current_plot_key: str = ""
    scratch: dict[str, object] = field(default_factory=dict)

    def table_path(self, table_name: str, cache_key: str) -> Path:
        safe = sanitize_name(table_name)
        return self.out_dir / "tables" / f"{safe}__{cache_key}.parquet"

    def figure_base(self, plot_name: str, suffix: str = "") -> str:
        safe_name = sanitize_name(plot_name)
        if suffix:
            suffix_part = f"__{sanitize_name(suffix)}"
        else:
            suffix_part = ""
        return f"{safe_name}__{self.current_plot_key}{suffix_part}"

    def add_message(self, message: str) -> None:
        self.messages.append(message)

    def as_extra(self) -> dict[str, Any]:
        return {
            "title": self.spec.title,
            "run_index_path": self.spec.run_index_path,
        }
