from __future__ import annotations

import re

from wafergeo.core.hashing import hash_config

_DEF_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def sanitize_name(name: str, *, max_len: int = 128) -> str:
    cleaned = _DEF_RE.sub("_", name.strip())
    sanitized = cleaned.strip("_") or "unnamed"
    if max_len <= 0:
        return "unnamed"
    if len(sanitized) <= max_len:
        return sanitized
    return sanitized[:max_len].rstrip("_.-") or "unnamed"


def build_table_cache_key(
    *,
    spec_hash: str,
    index_hash: str,
    code_version: str,
    table_name: str,
    table_version: str,
) -> str:
    return hash_config(
        {
            "spec_hash": spec_hash,
            "index_hash": index_hash,
            "code_version": code_version,
            "table_name": table_name,
            "table_version": table_version,
        }
    )


def build_plot_cache_key(
    *,
    spec_hash: str,
    index_hash: str,
    code_version: str,
    plot_name: str,
    plot_version: str,
    params_hash: str,
    heavy_selector_hash: str,
) -> str:
    return hash_config(
        {
            "spec_hash": spec_hash,
            "index_hash": index_hash,
            "code_version": code_version,
            "plot_name": plot_name,
            "plot_version": plot_version,
            "params_hash": params_hash,
            "heavy_selector_hash": heavy_selector_hash,
        }
    )
