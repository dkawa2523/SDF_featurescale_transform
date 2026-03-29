from __future__ import annotations

import pytest

from wafergeo.label.materials import load_materials_yaml, material_id_to_index, material_index_to_id


def test_load_materials_yaml_success(tmp_path) -> None:
    path = tmp_path / "materials.yaml"
    path.write_text(
        "\n".join(
            [
                'schema_version: "materials/v1"',
                "void_id: 0",
                "materials:",
                "  - id: 0",
                "    name: void",
                "    ignore_in_exposure: true",
                "    priority: 0",
                "  - id: 1",
                "    name: resist",
                "    ignore_in_exposure: false",
                "    priority: 10",
                "  - id: 2",
                "    name: oxide",
                "    ignore_in_exposure: false",
                "    priority: 20",
            ]
        ),
        encoding="utf-8",
    )

    material = load_materials_yaml(path)
    assert material.ids == [0, 1, 2]
    assert material_id_to_index(material) == {0: 0, 1: 1, 2: 2}
    assert material_index_to_id(material) == {0: 0, 1: 1, 2: 2}


def test_load_materials_yaml_invalid_missing_void_id(tmp_path) -> None:
    path = tmp_path / "materials.yaml"
    path.write_text(
        "\n".join(
            [
                'schema_version: "materials/v1"',
                "materials:",
                "  - id: 0",
                "    name: void",
                "    ignore_in_exposure: true",
                "    priority: 0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_materials_yaml(path)
