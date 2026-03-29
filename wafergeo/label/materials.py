from __future__ import annotations

from pathlib import Path
from typing import Any

from wafergeo.core.types import MaterialSpec


def _require_yaml() -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - import path specific
        raise ImportError("PyYAML is required for materials.yaml parsing") from exc
    return yaml


def load_materials_yaml(path: str | Path) -> MaterialSpec:
    yaml = _require_yaml()
    input_path = Path(path)
    data = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("materials.yaml must be a mapping object")

    schema_version = str(data.get("schema_version", ""))
    if schema_version != "materials/v1":
        raise ValueError(f"unsupported materials schema_version: {schema_version}")

    if "void_id" not in data:
        raise ValueError("materials.yaml is missing 'void_id'")
    if "materials" not in data:
        raise ValueError("materials.yaml is missing 'materials' list")

    material_items = data["materials"]
    if not isinstance(material_items, list) or not material_items:
        raise ValueError("materials must be a non-empty list")

    ids: list[int] = []
    names: list[str] = []
    priority: list[int] = []
    ignore_in_exposure: list[bool] = []

    for idx, item in enumerate(material_items):
        if not isinstance(item, dict):
            raise ValueError(f"materials[{idx}] must be a mapping")
        try:
            mid = int(item["id"])
            name = str(item["name"])
            prio = int(item["priority"])
            ignore = bool(item["ignore_in_exposure"])
        except KeyError as exc:
            raise ValueError(f"materials[{idx}] missing key: {exc.args[0]}") from exc
        ids.append(mid)
        names.append(name)
        priority.append(prio)
        ignore_in_exposure.append(ignore)

    return MaterialSpec(
        ids=ids,
        names=names,
        void_id=int(data["void_id"]),
        priority=priority,
        ignore_in_exposure=ignore_in_exposure,
    )


def material_id_to_index(material: MaterialSpec) -> dict[int, int]:
    return {material_id: idx for idx, material_id in enumerate(material.ids)}


def material_index_to_id(material: MaterialSpec) -> dict[int, int]:
    return {idx: material_id for idx, material_id in enumerate(material.ids)}
