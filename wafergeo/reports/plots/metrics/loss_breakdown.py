from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

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


class LossBreakdownPlot:
    name = "metrics.loss_breakdown"
    version = "1.0.0"
    required_tables: tuple[str, ...] = ("metrics.long",)
    is_heavy = False

    def run(self, ctx: ReportContext, request: PlotRequestSpec) -> PlotRunResult:
        plt = _load_matplotlib_pyplot()
        df = cast(Any, ctx.tables["metrics.long"])

        if len(df) == 0:
            return PlotRunResult(
                plot_name=self.name,
                figure_paths=[],
                status="WARN",
                messages=["metrics.long is empty"],
                meta={"n_rows": 0},
            )

        grouped = df.groupby("metric_name", as_index=False)["loss"].mean()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(grouped["metric_name"], grouped["loss"])
        ax.set_xlabel("metric")
        ax.set_ylabel("mean loss")
        ax.set_title("Loss Breakdown")
        fig.tight_layout()

        paths = save_figure(
            fig,
            out_dir=ctx.out_dir,
            base_name=ctx.figure_base(self.name),
            write_png=ctx.spec.output.write_png,
            write_svg=ctx.spec.output.write_svg,
            dpi=ctx.spec.output.dpi,
        )
        plt.close(fig)

        return PlotRunResult(
            plot_name=self.name,
            figure_paths=paths,
            status="OK",
            messages=[],
            meta={"n_rows": int(len(df)), "n_metrics": int(len(grouped))},
        )
