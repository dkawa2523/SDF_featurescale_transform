from __future__ import annotations

from typing import cast

from wafergeo.core.registry import Registry
from wafergeo.observe.base import ObserverProtocol

_observer_registry: Registry[type[object]] = Registry("observer_v2")


def register_observer(kind: str, observer_cls: type[object]) -> None:
    _observer_registry.register(kind, observer_cls, override=True)


def get_observer(kind: str) -> type[object]:
    register_default_observers()
    return _observer_registry.get(kind)


def list_observers() -> tuple[str, ...]:
    register_default_observers()
    return _observer_registry.list()


def create_observer(kind: str) -> ObserverProtocol:
    register_default_observers()
    observer_cls = cast(type[ObserverProtocol], _observer_registry.get(kind))
    return observer_cls()


def register_default_observers() -> None:
    if "topdown_exposed" not in _observer_registry.list():
        from wafergeo.observe.topdown import TopDownExposedObserver

        register_observer("topdown_exposed", TopDownExposedObserver)
    if "slice" not in _observer_registry.list():
        from wafergeo.observe.slice import SliceObserver

        register_observer("slice", SliceObserver)
