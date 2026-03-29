from __future__ import annotations

from wafergeo.reports.spec import load_report_spec_yaml, report_spec_hash


def test_report_spec_load_and_hash_deterministic(tmp_path) -> None:
    spec_path = tmp_path / "report.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "schema_version: report/v1",
                "report_id: smoke_report",
                "title: Smoke Report",
                "run_index_path: run_index.json",
                "output:",
                "  write_png: true",
                "  write_svg: false",
                "  write_tables_parquet: true",
                "  write_html_index: false",
                "  dpi: 200",
                "plots:",
                "  - name: metrics.loss_breakdown",
                "    params: {}",
                "  - name: metrics.residual_map",
                "    params: {}",
                "    heavy_selector:",
                "      top_k: 2",
            ]
        ),
        encoding="utf-8",
    )

    a = load_report_spec_yaml(spec_path)
    b = load_report_spec_yaml(spec_path)
    assert a.report_id == "smoke_report"
    assert a.plots[1].heavy_selector is not None
    assert report_spec_hash(a) == report_spec_hash(b)
