from __future__ import annotations

import pytest

from wafergeo.core.registry import Registry


def test_registry_register_get_and_list() -> None:
    registry: Registry[object] = Registry("test")
    marker = object()
    registry.register("x", marker)
    assert registry.get("x") is marker
    assert registry.list() == ("x",)


def test_registry_duplicate_register_raises() -> None:
    registry: Registry[int] = Registry("test")
    registry.register("x", 1)
    with pytest.raises(KeyError):
        registry.register("x", 2)


def test_registry_create_invokes_callable() -> None:
    registry: Registry[object] = Registry("test")

    def factory(x: int, y: int) -> int:
        return x + y

    registry.register("adder", factory)
    result = registry.create("adder", 3, 4)
    assert result == 7


def test_registry_create_non_callable_raises() -> None:
    registry: Registry[object] = Registry("test")
    registry.register("value", 123)
    with pytest.raises(TypeError):
        registry.create("value")
