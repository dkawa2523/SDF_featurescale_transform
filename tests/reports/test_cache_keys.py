from __future__ import annotations

from wafergeo.reports.cache import build_plot_cache_key, build_table_cache_key
from wafergeo.reports.runner import _reports_source_hash


def test_cache_key_is_deterministic_and_changes_on_version() -> None:
    table_a = build_table_cache_key(
        spec_hash="s",
        index_hash="i",
        code_version="0.1.0",
        table_name="metrics.long",
        table_version="1.0.0",
    )
    table_b = build_table_cache_key(
        spec_hash="s",
        index_hash="i",
        code_version="0.1.0",
        table_name="metrics.long",
        table_version="1.0.0",
    )
    table_c = build_table_cache_key(
        spec_hash="s",
        index_hash="i",
        code_version="0.1.0",
        table_name="metrics.long",
        table_version="2.0.0",
    )
    assert table_a == table_b
    assert table_a != table_c

    plot_a = build_plot_cache_key(
        spec_hash="s",
        index_hash="i",
        code_version="0.1.0",
        plot_name="metrics.loss_breakdown",
        plot_version="1.0.0",
        params_hash="p",
        heavy_selector_hash="h",
    )
    plot_b = build_plot_cache_key(
        spec_hash="s",
        index_hash="i",
        code_version="0.1.0",
        plot_name="metrics.loss_breakdown",
        plot_version="1.0.0",
        params_hash="p",
        heavy_selector_hash="h",
    )
    plot_c = build_plot_cache_key(
        spec_hash="s",
        index_hash="i",
        code_version="0.1.0",
        plot_name="metrics.loss_breakdown",
        plot_version="1.0.0",
        params_hash="p2",
        heavy_selector_hash="h",
    )
    assert plot_a == plot_b
    assert plot_a != plot_c


def test_reports_source_hash_changes_with_file_content(tmp_path) -> None:
    root = tmp_path / "reports"
    root.mkdir(parents=True, exist_ok=True)
    file_a = root / "a.py"
    file_a.write_text("VALUE = 1\n", encoding="utf-8")
    hash_a = _reports_source_hash(root)

    file_a.write_text("VALUE = 2\n", encoding="utf-8")
    hash_b = _reports_source_hash(root)
    assert hash_a != hash_b

    sub_dir = root / "sub"
    sub_dir.mkdir(parents=True, exist_ok=True)
    file_b = sub_dir / "b.py"
    file_b.write_text("X = 'ok'\n", encoding="utf-8")
    hash_c = _reports_source_hash(root)
    assert hash_b != hash_c
