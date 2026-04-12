from __future__ import annotations

import importlib
import sys


def test_wafergeo_init_uses_lazy_submodule_import() -> None:
    stale = [name for name in sys.modules if name == "wafergeo" or name.startswith("wafergeo.")]
    saved_modules = {name: sys.modules[name] for name in stale}
    for name in stale:
        sys.modules.pop(name, None)

    try:
        wafergeo = importlib.import_module("wafergeo")
        assert "wafergeo.compare" not in sys.modules
        assert "wafergeo.application" not in sys.modules

        _ = wafergeo.compare
        assert "wafergeo.compare" in sys.modules
        assert "wafergeo.application" not in sys.modules
    finally:
        current = [
            name
            for name in sys.modules
            if name == "wafergeo" or name.startswith("wafergeo.")
        ]
        for name in current:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
