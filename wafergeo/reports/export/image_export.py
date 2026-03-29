from __future__ import annotations

from pathlib import Path


def save_figure(
    fig,
    *,
    out_dir: Path,
    base_name: str,
    write_png: bool,
    write_svg: bool,
    dpi: int,
) -> list[str]:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    if write_png:
        path_png = figures_dir / f"{base_name}.png"
        fig.savefig(path_png, dpi=dpi, format="png")
        paths.append(str(path_png.relative_to(out_dir).as_posix()))

    if write_svg:
        path_svg = figures_dir / f"{base_name}.svg"
        fig.savefig(path_svg, format="svg")
        paths.append(str(path_svg.relative_to(out_dir).as_posix()))

    return paths
