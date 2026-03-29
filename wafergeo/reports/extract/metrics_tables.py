from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from wafergeo.reports.cache import sanitize_name
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


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported json value type: {type(value).__name__}")


def _as_mapping(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items()}


def _write_map_array(
    *,
    out_dir: Path,
    result_id: str,
    map_key: str,
    arr: np.ndarray,
) -> str:
    maps_dir = out_dir / "tables" / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    safe_result = sanitize_name(result_id)
    safe_map_key = sanitize_name(map_key)
    filename = f"{safe_result}__{safe_map_key}.npy"
    rel = Path("tables") / "maps" / filename
    np.save(out_dir / rel, np.asarray(arr), allow_pickle=False)
    return rel.as_posix()


def _collect_metrics_rows(
    ctx: ReportContext,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cached = ctx.scratch.get("metrics_rows")
    if isinstance(cached, tuple) and len(cached) == 2:
        cached_long_rows, cached_map_rows = cached
        if isinstance(cached_long_rows, list) and isinstance(cached_map_rows, list):
            return cached_long_rows, cached_map_rows

    index_df_any = cast(Any, ctx.tables["index.run_index"])
    long_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []

    for _, index_row in index_df_any.iterrows():
        trial_id = str(index_row["trial_artifact_id"])
        sample_id = str(index_row["sample_id"])
        group_id = str(index_row["group_id"])

        payload = ctx.store.load(trial_id)
        if not isinstance(payload, dict):
            continue

        total_loss = float(payload.get("total_loss", np.nan))
        metric_results = payload.get("metric_results", [])
        if not isinstance(metric_results, list):
            continue

        for result_idx, result in enumerate(metric_results):
            if not isinstance(result, dict):
                continue
            result_id = f"{trial_id}:{result_idx}"
            meta = _as_mapping(result.get("meta"))
            report = _as_mapping(result.get("report"))
            maps = _as_mapping(result.get("maps"))
            map_keys_sorted = sorted(maps.keys())

            long_rows.append(
                {
                    "result_id": result_id,
                    "trial_id": trial_id,
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "metric_name": str(result.get("name", "")),
                    "observer": str(meta.get("observer", "")),
                    "loss": float(result.get("loss", np.nan)),
                    "status": str(result.get("status", "FAIL")),
                    "total_loss": total_loss,
                    "report_json": json.dumps(
                        report,
                        ensure_ascii=True,
                        sort_keys=True,
                        default=_json_default,
                    ),
                    "has_maps": bool(map_keys_sorted),
                    "map_keys": ",".join(map_keys_sorted),
                    "map_count": int(len(map_keys_sorted)),
                }
            )

            for map_key, map_value in maps.items():
                arr = np.asarray(map_value)
                if arr.size == 0:
                    continue
                map_rows.append(
                    {
                        "result_id": result_id,
                        "trial_id": trial_id,
                        "sample_id": sample_id,
                        "group_id": group_id,
                        "metric_name": str(result.get("name", "")),
                        "observer": str(meta.get("observer", "")),
                        "map_key": str(map_key),
                        "map_path": _write_map_array(
                            out_dir=ctx.out_dir,
                            result_id=result_id,
                            map_key=str(map_key),
                            arr=arr,
                        ),
                        "map_shape": ",".join(str(int(v)) for v in arr.shape),
                        "map_dtype": str(arr.dtype),
                    }
                )

    ctx.scratch["metrics_rows"] = (long_rows, map_rows)
    return long_rows, map_rows


class MetricsLongExtractor:
    name = "metrics_long"
    version = "1.2.0"
    produces_table = "metrics.long"
    required_tables: tuple[str, ...] = ("index.run_index",)

    def build(self, ctx: ReportContext) -> pd.DataFrame:
        pd = _load_pandas()
        long_rows, _ = _collect_metrics_rows(ctx)
        if not long_rows:
            return pd.DataFrame(
                columns=[
                    "result_id",
                    "trial_id",
                    "sample_id",
                    "group_id",
                    "metric_name",
                    "observer",
                    "loss",
                    "status",
                    "total_loss",
                    "report_json",
                    "has_maps",
                    "map_keys",
                    "map_count",
                ]
            )
        return pd.DataFrame(long_rows)


class MetricsMapsExtractor:
    name = "metrics_maps"
    version = "1.1.0"
    produces_table = "metrics.maps"
    required_tables: tuple[str, ...] = ("index.run_index",)

    def build(self, ctx: ReportContext) -> pd.DataFrame:
        pd = _load_pandas()
        _, map_rows = _collect_metrics_rows(ctx)
        if not map_rows:
            return pd.DataFrame(
                columns=[
                    "result_id",
                    "trial_id",
                    "sample_id",
                    "group_id",
                    "metric_name",
                    "observer",
                    "map_key",
                    "map_path",
                    "map_shape",
                    "map_dtype",
                ]
            )
        return pd.DataFrame(map_rows)
