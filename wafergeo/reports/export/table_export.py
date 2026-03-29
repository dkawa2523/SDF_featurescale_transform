from __future__ import annotations

from pathlib import Path
from typing import Any

from wafergeo.reports.errors import ReportDependencyError


def _load_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore[import-not-found]

        return pd
    except ImportError as exc:
        raise ReportDependencyError(
            "pandas is required. Install: pip install 'wafergeo[viz]'"
        ) from exc



def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ReportDependencyError(
            "parquet writer requires pyarrow. Install: pip install 'wafergeo[parquet]'"
        ) from exc



def write_table_parquet(df, path: Path) -> None:
    _ = _load_pandas()
    _require_pyarrow()

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")



def read_table_parquet(path: Path):
    pd = _load_pandas()
    _require_pyarrow()
    return pd.read_parquet(path, engine="pyarrow")
