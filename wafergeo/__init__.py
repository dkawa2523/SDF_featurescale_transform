"""wafergeo package."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__version__ = "0.1.0"

_SUBMODULES = (
    "core",
    "io",
    "label",
    "sdf",
    "mesh",
    "compare",
)

__all__ = ["__version__", *_SUBMODULES]


def __getattr__(name: str) -> ModuleType:
    if name not in _SUBMODULES:
        raise AttributeError(f"module 'wafergeo' has no attribute '{name}'")
    module = import_module(f"wafergeo.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_SUBMODULES))
