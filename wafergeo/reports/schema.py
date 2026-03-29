from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wafergeo.core.types import Status


@dataclass(frozen=True)
class HeavySelector:
    top_k: int | None = None
    only_failures: bool = False
    sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("heavy_selector.top_k must be > 0 when set")

    def to_hash_payload(self) -> dict[str, object]:
        return {
            "top_k": self.top_k,
            "only_failures": self.only_failures,
            "sample_ids": list(self.sample_ids),
        }


@dataclass(frozen=True)
class PlotRequestSpec:
    name: str
    params: dict[str, object] = field(default_factory=dict)
    heavy_selector: HeavySelector | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("plot name must be non-empty")

    def to_hash_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "heavy_selector": (
                None if self.heavy_selector is None else self.heavy_selector.to_hash_payload()
            ),
        }


@dataclass(frozen=True)
class ReportOutputSpec:
    write_png: bool = True
    write_svg: bool = False
    write_tables_parquet: bool = True
    write_html_index: bool = False
    dpi: int = 200

    def __post_init__(self) -> None:
        if self.dpi <= 0:
            raise ValueError("output.dpi must be > 0")
        if not self.write_png and not self.write_svg:
            raise ValueError("at least one figure format must be enabled")

    def to_hash_payload(self) -> dict[str, object]:
        return {
            "write_png": self.write_png,
            "write_svg": self.write_svg,
            "write_tables_parquet": self.write_tables_parquet,
            "write_html_index": self.write_html_index,
            "dpi": self.dpi,
        }


@dataclass(frozen=True)
class ReportSpecV1:
    schema_version: str
    report_id: str
    title: str
    run_index_path: str
    output: ReportOutputSpec = field(default_factory=ReportOutputSpec)
    plots: tuple[PlotRequestSpec, ...] = ()
    filters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != "report/v1":
            raise ValueError(f"schema_version must be 'report/v1', got {self.schema_version}")
        if not self.report_id:
            raise ValueError("report_id must be non-empty")
        if not self.title:
            raise ValueError("title must be non-empty")
        if not self.run_index_path:
            raise ValueError("run_index_path must be non-empty")
        if not self.plots:
            raise ValueError("plots must be non-empty")

    def to_hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "title": self.title,
            "run_index_path": self.run_index_path,
            "output": self.output.to_hash_payload(),
            "plots": [plot.to_hash_payload() for plot in self.plots],
            "filters": dict(self.filters),
        }


@dataclass(frozen=True)
class PlotRunResult:
    plot_name: str
    figure_paths: list[str] = field(default_factory=list)
    status: Status = "OK"
    messages: list[str] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TableManifestEntry:
    table_name: str
    version: str
    cache_key: str
    path: str
    rows: int
    columns: list[str]
    cache_hit: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "table_name": self.table_name,
            "version": self.version,
            "cache_key": self.cache_key,
            "path": self.path,
            "rows": self.rows,
            "columns": list(self.columns),
            "cache_hit": self.cache_hit,
        }


@dataclass(frozen=True)
class FigureManifestEntry:
    plot_name: str
    version: str
    cache_key: str
    figure_paths: list[str]
    status: Status
    cache_hit: bool
    messages: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "plot_name": self.plot_name,
            "version": self.version,
            "cache_key": self.cache_key,
            "figure_paths": list(self.figure_paths),
            "status": self.status,
            "cache_hit": self.cache_hit,
            "messages": list(self.messages),
        }


@dataclass(frozen=True)
class ReportManifestV1:
    schema_version: str
    report_id: str
    created_at: str
    spec_hash: str
    index_hash: str
    code_version: str
    tables: list[TableManifestEntry]
    figures: list[FigureManifestEntry]
    status: Status
    messages: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != "report_manifest/v1":
            raise ValueError(
                "schema_version must be 'report_manifest/v1', "
                f"got {self.schema_version}"
            )
        if not self.report_id:
            raise ValueError("report_id must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "created_at": self.created_at,
            "spec_hash": self.spec_hash,
            "index_hash": self.index_hash,
            "code_version": self.code_version,
            "tables": [row.to_dict() for row in self.tables],
            "figures": [row.to_dict() for row in self.figures],
            "status": self.status,
            "messages": list(self.messages),
            "extra": dict(self.extra),
        }
