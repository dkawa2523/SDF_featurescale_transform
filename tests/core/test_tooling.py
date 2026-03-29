from __future__ import annotations

from pathlib import Path


def test_makefile_pins_python311_for_test_target() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")
    assert "PY311 ?= python3.11" in text
    assert "$(PY311) -m pytest -q" in text
