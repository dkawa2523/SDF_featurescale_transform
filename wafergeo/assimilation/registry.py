from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from wafergeo.assimilation.types import ModelLoaderProtocol, SurrogateModelProtocol
from wafergeo.core.registry import Registry

_loader_registry: Registry[object] = Registry("assim_model_loader")


@dataclass
class InMemoryModelLoader:
    name: str = "in_memory"

    def load(
        self,
        model_ref: dict[str, object],
        store: object,
    ) -> SurrogateModelProtocol:
        _ = store
        model = model_ref.get("model")
        if model is None:
            raise ValueError("in_memory loader requires model_ref['model']")
        if not hasattr(model, "predict") or not callable(model.predict):  # type: ignore[attr-defined]
            raise TypeError("model_ref['model'] must implement predict(params)")
        return cast(SurrogateModelProtocol, model)


def register_model_loader(
    loader: ModelLoaderProtocol,
    *,
    aliases: tuple[str, ...] = (),
) -> None:
    _loader_registry.register(loader.name, loader, override=True)
    for alias in aliases:
        _loader_registry.register(alias, loader, override=True)


def get_model_loader(name: str) -> ModelLoaderProtocol:
    register_default_model_loaders()
    return cast(ModelLoaderProtocol, _loader_registry.get(name))


def list_model_loaders() -> tuple[str, ...]:
    register_default_model_loaders()
    return _loader_registry.list()


def register_default_model_loaders() -> None:
    if "in_memory" not in _loader_registry.list():
        register_model_loader(InMemoryModelLoader())
