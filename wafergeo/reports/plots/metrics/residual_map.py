from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from wafergeo.core.types import Status
from wafergeo.reports.cache import sanitize_name
from wafergeo.reports.errors import ReportDependencyError
from wafergeo.reports.export.image_export import save_figure
from wafergeo.reports.schema import PlotRequestSpec, PlotRunResult

if TYPE_CHECKING:
    from wafergeo.reports.context import ReportContext


def _load_matplotlib_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]

        return plt
    except ImportError as exc:
        raise ReportDependencyError(
            "matplotlib is required. Install: pip install 'wafergeo[viz]'"
        ) from exc


def _load_map_array(*, out_dir: Path, map_path: object) -> np.ndarray | None:
    if not isinstance(map_path, str) or not map_path:
        return None
    abs_path = out_dir / map_path
    if not abs_path.exists():
        return None
    arr = np.asarray(np.load(abs_path, allow_pickle=False), dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim >= 3:
        return arr.reshape((-1,) + arr.shape[-2:])[0]
    return None


class ResidualMapPlot:
    name = "metrics.residual_map"
    version = "1.1.0"
    required_tables: tuple[str, ...] = ("metrics.long", "metrics.maps")
    is_heavy = True

    def run(self, ctx: ReportContext, request: PlotRequestSpec) -> PlotRunResult:
        plt = _load_matplotlib_pyplot()
        df = cast(Any, ctx.tables["metrics.long"])
        maps_df = cast(Any, ctx.tables["metrics.maps"])

        selector = request.heavy_selector
        if selector is None:
            raise ValueError("heavy plot requires top_k or sample_ids")

        work = df
        if selector.only_failures:
            work = work[work["status"] != "OK"]
        if selector.sample_ids:
            work = work[work["sample_id"].isin(list(selector.sample_ids))]

        if selector.top_k is not None:
            work = work.sort_values("loss", ascending=False).head(int(selector.top_k))

        if len(work) == 0:
            return PlotRunResult(
                plot_name=self.name,
                figure_paths=[],
                status="WARN",
                messages=["no rows selected for residual_map"],
                meta={"n_rows": 0},
            )

        figure_paths: list[str] = []
        for idx, row in work.reset_index(drop=True).iterrows():
            maps_rows = maps_df[maps_df["result_id"] == row["result_id"]]
            if len(maps_rows) == 0:
                continue
            maps_rows = maps_rows.sort_values("map_key")
            arr = _load_map_array(out_dir=ctx.out_dir, map_path=maps_rows.iloc[0]["map_path"])
            if arr is None:
                continue

            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(arr, cmap="coolwarm")
            ax.set_title(
                f"Residual #{idx + 1} sample={row['sample_id']} metric={row['metric_name']}"
            )
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()

            suffix = sanitize_name(f"{idx+1}_{row['sample_id']}_{row['metric_name']}")
            figure_paths.extend(
                save_figure(
                    fig,
                    out_dir=ctx.out_dir,
                    base_name=ctx.figure_base(self.name, suffix=str(suffix)),
                    write_png=ctx.spec.output.write_png,
                    write_svg=ctx.spec.output.write_svg,
                    dpi=ctx.spec.output.dpi,
                )
            )
            plt.close(fig)

        status: Status = "OK" if figure_paths else "WARN"
        messages = [] if figure_paths else ["selected rows did not contain map arrays"]
        return PlotRunResult(
            plot_name=self.name,
            figure_paths=figure_paths,
            status=status,
            messages=messages,
            meta={"n_rows": int(len(work)), "n_figures": len(figure_paths)},
        )
