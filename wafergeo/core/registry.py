from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class Registry(Generic[T]):
    """Simple keyed registry for pluggable components."""

    name: str
    _items: dict[str, T] = field(default_factory=dict)

    def register(self, key: str, value: T, *, override: bool = False) -> None:
        if not key:
            raise ValueError("registry key must be non-empty")
        if not override and key in self._items:
            raise KeyError(f"{self.name}: key already registered: {key}")
        self._items[key] = value

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError as exc:
            raise KeyError(f"{self.name}: unknown key: {key}") from exc

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(self._items.keys()))

    def create(self, key: str, *args: Any, **kwargs: Any) -> Any:
        factory = self.get(key)
        if not callable(factory):
            raise TypeError(f"{self.name}: item '{key}' is not callable")
        callable_factory = factory
        return callable_factory(*args, **kwargs)


# Domain-level registries (Phase 0 placeholders).
sdf_backend_registry: Registry[Any] = Registry("sdf_backend")
observer_registry: Registry[Any] = Registry("observer")
metric_registry: Registry[Any] = Registry("metric")
report_plot_registry: Registry[Any] = Registry("report_plot")
report_extractor_registry: Registry[Any] = Registry("report_extractor")
sem_reader_registry: Registry[Any] = Registry("sem_reader")


def register_observer(kind: str, cls: Any) -> None:
    observer_registry.register(kind, cls)


def register_metric(name: str, cls: Any) -> None:
    metric_registry.register(name, cls)


def register_sdf_backend(name: str, factory: Callable[..., Any]) -> None:
    sdf_backend_registry.register(name, factory)


def register_sdf_engine(name: str, engine: Any) -> None:
    """Register metadata-rich SDF engine object in the shared backend registry."""

    sdf_backend_registry.register(name, engine)
