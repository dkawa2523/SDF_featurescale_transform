from __future__ import annotations

import importlib
import sys


def test_wafergeo_init_uses_lazy_submodule_import() -> None:
    stale = [name for name in sys.modules if name == "wafergeo" or name.startswith("wafergeo.")]
    for name in stale:
        sys.modules.pop(name, None)

    wafergeo = importlib.import_module("wafergeo")
    assert "wafergeo.assimilation" not in sys.modules
    assert "wafergeo.reports" not in sys.modules

    _ = wafergeo.reports
    assert "wafergeo.reports" in sys.modules
    assert "wafergeo.assimilation" not in sys.modules
