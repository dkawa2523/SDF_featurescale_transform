from __future__ import annotations

import numpy as np

import wafergeo.io.vti_reader as vti_reader
from wafergeo.io.vti_reader import (
    RawVtiImage,
    VtiReadResult,
    extract_material_ids,
    infer_flat_array_layout,
    read_vti,
    resolve_material_array_name,
)


def test_read_vti_without_vtk_raises_helpful_error() -> None:
    try:
        read_vti("dummy.vti")
    except ImportError as exc:
        assert "wafergeo[vtk]" in str(exc)
    except FileNotFoundError as exc:
        assert "dummy.vti" in str(exc)


def test_read_vti_with_xml_fallback_prefers_vtk(tmp_path, monkeypatch) -> None:
    vti_path = tmp_path / "dummy.vti"
    vti_path.write_bytes(b"vti")
    raw = RawVtiImage(
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        dims_xyz=(2, 2, 2),
        arrays={"MaterialIds": np.array([2, 3], dtype=np.int32)},
        array_location={"MaterialIds": "cell"},
    )
    monkeypatch.setattr(vti_reader, "read_vti", lambda _p: raw)
    monkeypatch.setattr(
        vti_reader,
        "read_vti_materialids_xml_fallback",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("xml fallback should not run")),
    )

    result = vti_reader.read_vti_with_xml_fallback(vti_path)

    assert isinstance(result, VtiReadResult)
    assert result.backend_used == "vtk"
    assert result.messages == ()
    assert result.raw is raw


def test_read_vti_with_xml_fallback_reports_fallback_reason(tmp_path, monkeypatch) -> None:
    vti_path = tmp_path / "dummy.vti"
    vti_path.write_bytes(b"vti")
    raw = RawVtiImage(
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        dims_xyz=(2, 2, 2),
        arrays={"MaterialIds": np.array([2, 3], dtype=np.int32)},
        array_location={"MaterialIds": "cell"},
        vtk_meta={"reader": "xml_fallback_materialids_only"},
    )
    monkeypatch.setattr(
        vti_reader,
        "read_vti",
        lambda _p: (_ for _ in ()).throw(ImportError("no vtk runtime")),
    )
    monkeypatch.setattr(vti_reader, "read_vti_materialids_xml_fallback", lambda *_a, **_k: raw)

    result = vti_reader.read_vti_with_xml_fallback(vti_path)

    assert result.backend_used == "xml_fallback"
    assert len(result.messages) == 1
    assert "fallback" in result.messages[0]
    assert "no vtk runtime" in result.messages[0]
    assert result.raw is raw


def test_resolve_material_array_name_prefers_candidates() -> None:
    raw = RawVtiImage(
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        dims_xyz=(2, 2, 2),
        arrays={
            "foo": np.array([0, 1], dtype=np.int32),
            "MaterialIds": np.array([2, 3], dtype=np.int32),
        },
        array_location={"foo": "cell", "MaterialIds": "cell"},
    )
    assert resolve_material_array_name(raw) == "MaterialIds"


def test_extract_material_ids_returns_zyx_int_labels() -> None:
    point_zyx = np.array(
        [
            [[2, 3], [4, 5]],
            [[6, 7], [8, 9]],
        ],
        dtype=np.int32,
    )
    point_xyz = point_zyx.transpose(2, 1, 0)
    raw = RawVtiImage(
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        dims_xyz=(2, 2, 2),
        arrays={"MaterialIds": point_xyz.ravel(order="C")},
        array_location={"MaterialIds": "point"},
    )
    labels_zyx, location = extract_material_ids(raw, "MaterialIds")
    assert location == "point"
    assert labels_zyx.dtype == np.int64
    np.testing.assert_array_equal(labels_zyx, point_zyx.astype(np.int64))


def test_extract_material_ids_vtk_x_fastest_matches_expected_axis() -> None:
    nx, ny, nz = (3, 2, 2)
    expected_zyx = np.fromfunction(
        lambda z, y, x: 100 * z + 10 * y + x,
        (nz, ny, nx),
        dtype=int,
    ).astype(np.int32)
    flat_x_fastest = expected_zyx.ravel(order="C")
    raw = RawVtiImage(
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        dims_xyz=(nx, ny, nz),
        arrays={"MaterialIds": flat_x_fastest},
        array_location={"MaterialIds": "point"},
        vtk_meta={"reader": "vtkXMLImageDataReader"},
    )
    assert infer_flat_array_layout(raw) == "vtk_x_fastest"
    labels_zyx, location = extract_material_ids(raw, "MaterialIds")
    assert location == "point"
    np.testing.assert_array_equal(labels_zyx, expected_zyx.astype(np.int64))
