from __future__ import annotations

from pathlib import Path

from wafergeo.reports.schema import ReportManifestV1


def write_index_html(out_dir: Path, manifest: ReportManifestV1) -> str:
    html_dir = out_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    html_path = html_dir / "index.html"

    lines: list[str] = []
    lines.append("<!doctype html>")
    lines.append("<html><head><meta charset='utf-8'><title>wafergeo report</title></head><body>")
    lines.append(f"<h1>{manifest.report_id}</h1>")
    lines.append(f"<p>Status: {manifest.status}</p>")

    lines.append("<h2>Figures</h2><ul>")
    for fig_entry in manifest.figures:
        for fig_path in fig_entry.figure_paths:
            rel = "../" + fig_path
            lines.append(
                "<li>"
                f"{fig_entry.plot_name} ({fig_entry.status}) "
                f"<a href='{rel}'>{fig_path}</a>"
                "</li>"
            )
    lines.append("</ul>")

    lines.append("<h2>Tables</h2><ul>")
    for table_entry in manifest.tables:
        rel = "../" + table_entry.path
        lines.append(
            f"<li>{table_entry.table_name} <a href='{rel}'>{table_entry.path}</a></li>"
        )
    lines.append("</ul>")

    lines.append("</body></html>")
    html_path.write_text("\n".join(lines), encoding="utf-8")
    return str(html_path.relative_to(out_dir).as_posix())
