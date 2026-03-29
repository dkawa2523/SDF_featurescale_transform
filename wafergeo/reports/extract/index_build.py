from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wafergeo.reports.errors import ReportDependencyError

if TYPE_CHECKING:
    import pandas as pd

    from wafergeo.reports.context import ReportContext


def _load_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore[import-not-found]

        return pd
    except ImportError as exc:
        raise ReportDependencyError(
            "pandas is required. Install: pip install 'wafergeo[viz]'"
        ) from exc


class RunIndexExtractor:
    name = "run_index"
    version = "1.0.0"
    produces_table = "index.run_index"
    required_tables: tuple[str, ...] = ()

    def build(self, ctx: ReportContext) -> pd.DataFrame:
        pd = _load_pandas()
        path = Path(ctx.spec.run_index_path)
        if not path.exists():
            raise FileNotFoundError(f"run_index_path not found: {path}")

        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                rows = raw
            elif isinstance(raw, dict):
                rows_obj = raw.get("rows", raw.get("samples", []))
                if not isinstance(rows_obj, list):
                    raise ValueError("run_index JSON must contain list at 'rows' or 'samples'")
                rows = rows_obj
            else:
                raise ValueError("run_index JSON must be list or mapping")
            df = pd.DataFrame(rows)

        required = {"sample_id", "trial_artifact_id"}
        missing = sorted(required.difference(set(df.columns)))
        if missing:
            raise ValueError(f"run_index missing required columns: {missing}")

        if "group_id" not in df.columns:
            df["group_id"] = "default"

        return df[["sample_id", "group_id", "trial_artifact_id"]].copy()
