from __future__ import annotations

from importlib import import_module
from typing import Any


def _load_matplotlib(
    *,
    context: str,
    install_hint: str,
    extra_imports: tuple[tuple[str, str], ...] = (),
) -> tuple[Any, ...]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        loaded: list[Any] = [import_module("matplotlib.pyplot")]
        for module_name, attr_name in extra_imports:
            module = import_module(module_name)
            loaded.append(getattr(module, attr_name))
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"matplotlib is required for {context}. Install with: {install_hint}"
        ) from exc
    return tuple(loaded)


def require_matplotlib_pyplot(*, context: str, install_hint: str) -> Any:
    return _load_matplotlib(context=context, install_hint=install_hint)[0]
