from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from wafergeo.core.hashing import hash_config
from wafergeo.reports.schema import (
    HeavySelector,
    PlotRequestSpec,
    ReportOutputSpec,
    ReportSpecV1,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Report spec YAML root must be a mapping")
    return {str(k): v for k, v in raw.items()}


def _as_mapping(
    parent: dict[str, Any],
    key: str,
    *,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if key not in parent:
        if default is None:
            raise ValueError(f"missing required key: {key}")
        return default
    value = parent[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return {str(k): v for k, v in value.items()}


def _as_list(parent: dict[str, Any], key: str, *, default: list[Any] | None = None) -> list[Any]:
    if key not in parent:
        return default or []
    value = parent[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return list(value)


def _as_str(parent: dict[str, Any], key: str, default: str | None = None) -> str:
    if key not in parent:
        if default is None:
            raise ValueError(f"missing required key: {key}")
        return default
    return str(parent[key])


def _as_int(parent: dict[str, Any], key: str, default: int) -> int:
    if key not in parent:
        return int(default)
    return int(parent[key])


def _as_bool(parent: dict[str, Any], key: str, default: bool) -> bool:
    if key not in parent:
        return bool(default)
    value = parent[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def load_report_spec_yaml(path: str | Path) -> ReportSpecV1:
    raw = _read_yaml(Path(path))
    output_raw = _as_mapping(raw, "output", default={})
    plots_raw = _as_list(raw, "plots")

    plots: list[PlotRequestSpec] = []
    for idx, item in enumerate(plots_raw):
        if not isinstance(item, dict):
            raise ValueError(f"plots[{idx}] must be mapping")
        row = {str(k): v for k, v in item.items()}

        selector = None
        if "heavy_selector" in row and row["heavy_selector"] is not None:
            selector_raw = _as_mapping(row, "heavy_selector")
            sample_ids = tuple(str(v) for v in _as_list(selector_raw, "sample_ids", default=[]))
            top_k: int | None
            if "top_k" in selector_raw and selector_raw["top_k"] is not None:
                top_k = int(selector_raw["top_k"])
            else:
                top_k = None
            selector = HeavySelector(
                top_k=top_k,
                only_failures=_as_bool(selector_raw, "only_failures", False),
                sample_ids=sample_ids,
            )

        plots.append(
            PlotRequestSpec(
                name=_as_str(row, "name"),
                params=_as_mapping(row, "params", default={}),
                heavy_selector=selector,
            )
        )

    return ReportSpecV1(
        schema_version=_as_str(raw, "schema_version"),
        report_id=_as_str(raw, "report_id"),
        title=_as_str(raw, "title"),
        run_index_path=_as_str(raw, "run_index_path"),
        output=ReportOutputSpec(
            write_png=_as_bool(output_raw, "write_png", True),
            write_svg=_as_bool(output_raw, "write_svg", False),
            write_tables_parquet=_as_bool(output_raw, "write_tables_parquet", True),
            write_html_index=_as_bool(output_raw, "write_html_index", False),
            dpi=_as_int(output_raw, "dpi", 200),
        ),
        plots=tuple(plots),
        filters=_as_mapping(raw, "filters", default={}),
    )


def report_spec_hash(spec: ReportSpecV1) -> str:
    return hash_config(spec.to_hash_payload())
