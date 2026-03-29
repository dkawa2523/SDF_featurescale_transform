from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from wafergeo.core.hashing import hash_config, sha256_file
from wafergeo.core.types import Status
from wafergeo.io.artifact_store import ArtifactStore
from wafergeo.reports.cache import (
    build_plot_cache_key,
    build_table_cache_key,
    sanitize_name,
)
from wafergeo.reports.context import ReportContext
from wafergeo.reports.errors import ReportRegistryError
from wafergeo.reports.export.html_export import write_index_html
from wafergeo.reports.export.table_export import read_table_parquet, write_table_parquet
from wafergeo.reports.registry import (
    get_report_extractor,
    get_report_plot,
    register_default_report_components,
)
from wafergeo.reports.schema import (
    FigureManifestEntry,
    ReportManifestV1,
    ReportSpecV1,
    TableManifestEntry,
)
from wafergeo.reports.spec import report_spec_hash


def _reports_source_hash(root_dir: Path | None = None) -> str:
    root = root_dir or Path(__file__).resolve().parent
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _code_version() -> str:
    try:
        base_version = version("wafergeo")
    except PackageNotFoundError:
        base_version = "0.1.0"
    source_hash = _reports_source_hash()[:12]
    return f"{base_version}+src.{source_hash}"


def validate_report_inputs(spec: ReportSpecV1) -> list[str]:
    messages: list[str] = []
    if not Path(spec.run_index_path).exists():
        messages.append(f"run_index_path does not exist: {spec.run_index_path}")
    return messages



def _status_rank(status: Status) -> int:
    return {"OK": 0, "WARN": 1, "FAIL": 2}[status]



def _combine_status(a: Status, b: Status) -> Status:
    return a if _status_rank(a) >= _status_rank(b) else b



def _selector_hash(request) -> str:
    if request.heavy_selector is None:
        return hash_config({"heavy_selector": None})
    return hash_config(request.heavy_selector.to_hash_payload())



def _ensure_table(ctx: ReportContext, table_name: str) -> None:
    if table_name in ctx.tables:
        return

    extractor = get_report_extractor(table_name)

    for dep in extractor.required_tables:
        _ensure_table(ctx, dep)

    cache_key = build_table_cache_key(
        spec_hash=ctx.spec_hash,
        index_hash=ctx.index_hash,
        code_version=ctx.code_version,
        table_name=table_name,
        table_version=extractor.version,
    )
    table_path = ctx.table_path(table_name, cache_key)
    should_write_parquet = ctx.spec.output.write_tables_parquet
    cache_hit = should_write_parquet and table_path.exists()
    if cache_hit and should_write_parquet:
        df = read_table_parquet(table_path)
    else:
        df = extractor.build(ctx)
        if should_write_parquet:
            write_table_parquet(df, table_path)

    ctx.tables[table_name] = df
    if table_name not in ctx.table_entries:
        ctx.table_entries[table_name] = TableManifestEntry(
            table_name=table_name,
            version=extractor.version,
            cache_key=cache_key,
            path=(
                str(table_path.relative_to(ctx.out_dir).as_posix())
                if should_write_parquet
                else ""
            ),
            rows=int(len(df)),
            columns=[str(v) for v in list(df.columns)],
            cache_hit=cache_hit,
        )



def _heavy_selector_is_valid(request) -> bool:
    selector = request.heavy_selector
    if selector is None:
        return False
    if selector.top_k is not None:
        return True
    if selector.sample_ids:
        return True
    return False



def _plot_cache_marker_path(ctx: ReportContext, plot_name: str, cache_key: str) -> Path:
    safe_plot = sanitize_name(plot_name)
    return ctx.out_dir / "figures" / f"{safe_plot}__{cache_key}.cache.json"



def _figure_cache_hit(ctx: ReportContext, marker_path: Path) -> dict[str, Any] | None:
    if not marker_path.exists():
        return None
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    paths = payload.get("figure_paths")
    if not isinstance(paths, list):
        return None
    rel_paths: list[str] = [str(v) for v in paths]
    for rel in rel_paths:
        if not (ctx.out_dir / rel).exists():
            return None

    status_raw = str(payload.get("status", "OK"))
    status: Status = "OK"
    if status_raw in {"OK", "WARN", "FAIL"}:
        status = cast(Status, status_raw)
    messages_raw = payload.get("messages", [])
    if isinstance(messages_raw, list):
        messages = [str(v) for v in messages_raw]
    else:
        messages = []
    version_raw = payload.get("version")
    version_value = "" if version_raw is None else str(version_raw)

    return {
        "figure_paths": rel_paths,
        "status": status,
        "messages": messages,
        "version": version_value,
    }



def build_report(
    spec: ReportSpecV1,
    store: ArtifactStore,
    out_dir: str | Path,
) -> ReportManifestV1:
    issues = validate_report_inputs(spec)
    if issues:
        raise ValueError(" | ".join(issues))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "figures").mkdir(parents=True, exist_ok=True)
    (out_path / "tables").mkdir(parents=True, exist_ok=True)

    spec_hash = report_spec_hash(spec)
    index_hash = sha256_file(spec.run_index_path)

    ctx = ReportContext(
        store=store,
        out_dir=out_path,
        spec=spec,
        spec_hash=spec_hash,
        index_hash=index_hash,
        code_version=_code_version(),
    )

    register_default_report_components()

    overall_status: Status = "OK"

    for request in spec.plots:
        try:
            plot = get_report_plot(request.name)
        except ReportRegistryError as exc:
            overall_status = _combine_status(overall_status, "FAIL")
            ctx.figure_entries.append(
                FigureManifestEntry(
                    plot_name=request.name,
                    version="unknown",
                    cache_key="",
                    figure_paths=[],
                    status="FAIL",
                    cache_hit=False,
                    messages=[str(exc)],
                )
            )
            continue

        if plot.is_heavy and not _heavy_selector_is_valid(request):
            overall_status = _combine_status(overall_status, "FAIL")
            ctx.figure_entries.append(
                FigureManifestEntry(
                    plot_name=plot.name,
                    version=plot.version,
                    cache_key="",
                    figure_paths=[],
                    status="FAIL",
                    cache_hit=False,
                    messages=["heavy plot requires top_k or sample_ids"],
                )
            )
            continue

        params_hash = hash_config(dict(request.params))
        selector_hash = _selector_hash(request)
        plot_key = build_plot_cache_key(
            spec_hash=ctx.spec_hash,
            index_hash=ctx.index_hash,
            code_version=ctx.code_version,
            plot_name=plot.name,
            plot_version=plot.version,
            params_hash=params_hash,
            heavy_selector_hash=selector_hash,
        )

        try:
            for table_name in plot.required_tables:
                _ensure_table(ctx, table_name)
        except Exception as exc:
            overall_status = _combine_status(overall_status, "FAIL")
            ctx.figure_entries.append(
                FigureManifestEntry(
                    plot_name=plot.name,
                    version=plot.version,
                    cache_key=plot_key,
                    figure_paths=[],
                    status="FAIL",
                    cache_hit=False,
                    messages=[f"table resolution failed: {type(exc).__name__}: {exc}"],
                )
            )
            continue

        marker = _plot_cache_marker_path(ctx, plot.name, plot_key)
        cache_payload = _figure_cache_hit(ctx, marker)
        if cache_payload is not None:
            cached_status_raw = str(cache_payload["status"])
            cached_status: Status = "OK"
            if cached_status_raw in {"OK", "WARN", "FAIL"}:
                cached_status = cast(Status, cached_status_raw)
            ctx.figure_entries.append(
                FigureManifestEntry(
                    plot_name=plot.name,
                    version=str(cache_payload["version"] or plot.version),
                    cache_key=plot_key,
                    figure_paths=list(cache_payload["figure_paths"]),
                    status=cached_status,
                    cache_hit=True,
                    messages=list(cache_payload["messages"]),
                )
            )
            overall_status = _combine_status(overall_status, cached_status)
            continue

        ctx.current_plot_key = plot_key
        try:
            result = plot.run(ctx, request)
            marker.write_text(
                json.dumps(
                    {
                        "figure_paths": list(result.figure_paths),
                        "status": result.status,
                        "messages": list(result.messages),
                        "version": plot.version,
                    },
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            ctx.figure_entries.append(
                FigureManifestEntry(
                    plot_name=plot.name,
                    version=plot.version,
                    cache_key=plot_key,
                    figure_paths=list(result.figure_paths),
                    status=result.status,
                    cache_hit=False,
                    messages=list(result.messages),
                )
            )
            overall_status = _combine_status(overall_status, result.status)
        except Exception as exc:
            overall_status = _combine_status(overall_status, "FAIL")
            ctx.figure_entries.append(
                FigureManifestEntry(
                    plot_name=plot.name,
                    version=plot.version,
                    cache_key=plot_key,
                    figure_paths=[],
                    status="FAIL",
                    cache_hit=False,
                    messages=[f"plot failed: {type(exc).__name__}: {exc}"],
                )
            )

    manifest = ReportManifestV1(
        schema_version="report_manifest/v1",
        report_id=spec.report_id,
        created_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        spec_hash=spec_hash,
        index_hash=index_hash,
        code_version=ctx.code_version,
        tables=[ctx.table_entries[name] for name in sorted(ctx.table_entries.keys())],
        figures=ctx.figure_entries,
        status=overall_status,
        messages=list(ctx.messages),
        extra=ctx.as_extra(),
    )

    if spec.output.write_html_index:
        html_rel_path = write_index_html(out_path, manifest)
        ctx.add_message(f"html_index={html_rel_path}")
        manifest = ReportManifestV1(
            schema_version=manifest.schema_version,
            report_id=manifest.report_id,
            created_at=manifest.created_at,
            spec_hash=manifest.spec_hash,
            index_hash=manifest.index_hash,
            code_version=manifest.code_version,
            tables=manifest.tables,
            figures=manifest.figures,
            status=manifest.status,
            messages=ctx.messages,
            extra={**manifest.extra, "html_index": html_rel_path},
        )

    manifest_path = out_path / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest
