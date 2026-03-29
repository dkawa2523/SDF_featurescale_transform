from __future__ import annotations

from typing import cast

from wafergeo.core.registry import Registry
from wafergeo.mesh.extractors.base import MeshExtractorProtocol
from wafergeo.mesh.extractors.naive_interface import NaiveInterfaceExtractor
from wafergeo.mesh.extractors.vtk_interface import VTKInterfaceExtractor

mesh_extractor_registry: Registry[object] = Registry("mesh_extractor")


def register_mesh_extractor(
    extractor: MeshExtractorProtocol,
    *,
    aliases: tuple[str, ...] = (),
) -> None:
    mesh_extractor_registry.register(extractor.name, extractor, override=True)
    for alias in aliases:
        mesh_extractor_registry.register(alias, extractor, override=True)


def list_mesh_extractors() -> tuple[str, ...]:
    register_default_mesh_extractors()
    return mesh_extractor_registry.list()


def get_mesh_extractor(name: str) -> MeshExtractorProtocol:
    register_default_mesh_extractors()
    raw = mesh_extractor_registry.get(name)
    return cast(MeshExtractorProtocol, raw)


def register_default_mesh_extractors() -> None:
    if "naive_interface" not in mesh_extractor_registry.list():
        register_mesh_extractor(NaiveInterfaceExtractor())
    if "vtk" not in mesh_extractor_registry.list():
        register_mesh_extractor(VTKInterfaceExtractor())
