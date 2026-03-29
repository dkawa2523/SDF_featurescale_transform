from __future__ import annotations

import base64
import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast
from xml.etree import ElementTree as ET

import numpy as np

ArrayLocation = Literal["cell", "point"]
FlatArrayLayout = Literal["vtk_x_fastest", "legacy_xyz_transpose", "auto"]
DEFAULT_MATERIAL_ARRAY_CANDIDATES: tuple[str, ...] = (
    "MaterialIds",
    "material_id",
    "MaterialId",
)
_VTK_TO_NUMPY_DTYPE: dict[str, str] = {
    "Int8": "<i1",
    "UInt8": "<u1",
    "Int16": "<i2",
    "UInt16": "<u2",
    "Int32": "<i4",
    "UInt32": "<u4",
    "Int64": "<i8",
    "UInt64": "<u8",
    "Float32": "<f4",
    "Float64": "<f8",
}


@dataclass(frozen=True)
class RawVtiImage:
    """Raw VTI payload with VTK-native metadata.

    Reader keeps VTK-specific layout untouched; canonical normalization is handled
    by `wafergeo.label.normalize`.
    """

    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    dims_xyz: tuple[int, int, int]
    arrays: dict[str, np.ndarray]
    array_location: dict[str, ArrayLocation]
    vtk_meta: dict[str, str] = field(default_factory=dict)


def read_vti(path: str | Path) -> RawVtiImage:
    """Read VTI and return raw data object.

    Raises:
        ImportError: when VTK is unavailable.
        FileNotFoundError: when file path does not exist.
        ValueError: when VTK reader fails.
    """

    try:
        from vtk.util.numpy_support import vtk_to_numpy
        from vtkmodules.vtkIOXML import vtkXMLImageDataReader
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "VTK is required for VTI reading. Install with: pip install wafergeo[vtk]"
        ) from exc

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"VTI file not found: {input_path}")

    reader = vtkXMLImageDataReader()
    reader.SetFileName(str(input_path))
    reader.Update()
    image = reader.GetOutput()
    if image is None:
        raise ValueError(f"Failed to load VTI image: {input_path}")

    spacing_xyz = cast(tuple[float, float, float], tuple(float(v) for v in image.GetSpacing()))
    origin_xyz = cast(tuple[float, float, float], tuple(float(v) for v in image.GetOrigin()))
    dims_xyz = cast(tuple[int, int, int], tuple(int(v) for v in image.GetDimensions()))

    arrays: dict[str, np.ndarray] = {}
    array_location: dict[str, ArrayLocation] = {}

    def add_arrays(field_data, location: ArrayLocation) -> None:
        count = int(field_data.GetNumberOfArrays())
        for i in range(count):
            vtk_array = field_data.GetArray(i)
            if vtk_array is None:
                continue
            raw_name = vtk_array.GetName() or f"unnamed_{location}_{i}"
            name = raw_name
            if name in arrays:
                name = f"{raw_name}__{location}"
            arrays[name] = np.asarray(vtk_to_numpy(vtk_array))
            array_location[name] = location

    add_arrays(image.GetCellData(), "cell")
    add_arrays(image.GetPointData(), "point")

    vtk_meta = {
        "path": str(input_path),
        "reader": "vtkXMLImageDataReader",
    }

    return RawVtiImage(
        spacing_xyz=spacing_xyz,
        origin_xyz=origin_xyz,
        dims_xyz=dims_xyz,
        arrays=arrays,
        array_location=array_location,
        vtk_meta=vtk_meta,
    )


def resolve_material_array_name(
    raw: RawVtiImage,
    candidates: tuple[str, ...] = DEFAULT_MATERIAL_ARRAY_CANDIDATES,
) -> str:
    """Resolve material-id array name from candidates."""
    for name in candidates:
        if name in raw.arrays:
            return name
    available = ", ".join(sorted(raw.arrays.keys()))
    raise ValueError(
        "Material label array was not found. "
        f"Expected one of: {candidates}. "
        f"Available arrays: [{available}]"
    )


def extract_material_ids(
    raw: RawVtiImage,
    array_name: str,
    *,
    flat_layout: FlatArrayLayout = "auto",
) -> tuple[np.ndarray, ArrayLocation]:
    """Extract integer material ids in canonical ZYX order."""
    if array_name not in raw.arrays:
        raise ValueError(f"Unknown array name: {array_name}")
    location = raw.array_location[array_name]
    resolved_layout = _resolve_flat_layout(raw, flat_layout)
    labels_zyx = _to_zyx_local(
        raw.arrays[array_name],
        raw.dims_xyz,
        location,
        flat_layout=resolved_layout,
    )
    return _coerce_scalar_labels_local(labels_zyx), location


def infer_flat_array_layout(raw: RawVtiImage) -> FlatArrayLayout:
    """Infer flat-array memory layout used for VTI scalar arrays."""
    reader_name = raw.vtk_meta.get("reader", "")
    if reader_name in {"vtkXMLImageDataReader", "xml_fallback_materialids_only"}:
        return "vtk_x_fastest"
    return "legacy_xyz_transpose"


def _resolve_flat_layout(raw: RawVtiImage, flat_layout: FlatArrayLayout) -> FlatArrayLayout:
    if flat_layout == "auto":
        return infer_flat_array_layout(raw)
    return flat_layout


def read_vti_materialids_xml_fallback(
    path: str | Path,
    candidates: tuple[str, ...] = DEFAULT_MATERIAL_ARRAY_CANDIDATES,
) -> RawVtiImage:
    """Read only one material-id array from VTI XML without vtk runtime."""
    vti_path = Path(path)
    root = ET.parse(vti_path).getroot()
    if root.attrib.get("type") != "ImageData":
        raise ValueError(f"Unsupported VTI type: {root.attrib.get('type')}")

    image = root.find("ImageData")
    piece = image.find("Piece") if image is not None else None
    if image is None or piece is None:
        raise ValueError("Invalid VTI XML: missing ImageData/Piece")

    spacing_xyz = _parse_xyz3(image.attrib["Spacing"], name="Spacing")
    origin_xyz = _parse_xyz3(image.attrib["Origin"], name="Origin")
    extent = [int(v) for v in image.attrib["WholeExtent"].split()]
    if len(extent) != 6:
        raise ValueError(f"WholeExtent must have 6 elements, got {extent}")
    dims_xyz = (
        extent[1] - extent[0] + 1,
        extent[3] - extent[2] + 1,
        extent[5] - extent[4] + 1,
    )

    entries: list[tuple[str, str, int, str]] = []
    point_data = piece.find("PointData")
    if point_data is not None:
        for array in point_data.findall("DataArray"):
            name = array.attrib.get("Name")
            offset_raw = array.attrib.get("offset")
            if name is None or offset_raw is None:
                continue
            entries.append((name, "point", int(offset_raw), array.attrib.get("type", "")))
    cell_data = piece.find("CellData")
    if cell_data is not None:
        for array in cell_data.findall("DataArray"):
            name = array.attrib.get("Name")
            offset_raw = array.attrib.get("offset")
            if name is None or offset_raw is None:
                continue
            entries.append((name, "cell", int(offset_raw), array.attrib.get("type", "")))

    target_name: str | None = None
    target_location = "cell"
    target_offset = 0
    target_dtype = ""
    for candidate in candidates:
        for name, location, offset, vtk_dtype in entries:
            if name == candidate:
                target_name = name
                target_location = location
                target_offset = offset
                target_dtype = vtk_dtype
                break
        if target_name is not None:
            break
    if target_name is None:
        names = [name for name, _, _, _ in entries]
        raise ValueError(
            "Material label array was not found in VTI file. "
            f"Expected one of: {candidates}. Available arrays: {sorted(names)}"
        )
    if target_dtype not in _VTK_TO_NUMPY_DTYPE:
        raise ValueError(f"Unsupported VTI data type for MaterialIds: {target_dtype}")

    offsets = sorted(offset_value for _, _, offset_value, _ in entries)
    next_offsets = [value for value in offsets if value > target_offset]
    appended_text = _parse_appended_base64_text(vti_path)
    next_offset = next_offsets[0] if next_offsets else len(appended_text)
    segment = appended_text[target_offset:next_offset]
    payload = _decompress_vti_appended_segment(
        segment,
        header_type=root.attrib.get("header_type", "UInt32"),
    )
    labels = np.frombuffer(payload, dtype=np.dtype(_VTK_TO_NUMPY_DTYPE[target_dtype])).copy()

    array_location: ArrayLocation = cast(
        ArrayLocation,
        "point" if target_location == "point" else "cell",
    )
    return RawVtiImage(
        spacing_xyz=spacing_xyz,
        origin_xyz=origin_xyz,
        dims_xyz=dims_xyz,
        arrays={target_name: labels},
        array_location={target_name: array_location},
        vtk_meta={"path": str(vti_path), "reader": "xml_fallback_materialids_only"},
    )


def _parse_xyz3(raw_value: str, *, name: str) -> tuple[float, float, float]:
    values = [float(v) for v in raw_value.split()]
    if len(values) != 3:
        raise ValueError(f"{name} must have 3 elements, got {values}")
    return values[0], values[1], values[2]


def _parse_appended_base64_text(vti_path: Path) -> str:
    content = vti_path.read_text(encoding="utf-8")
    match = re.search(r"<AppendedData[^>]*>(.*?)</AppendedData>", content, flags=re.S)
    if match is None:
        raise ValueError("VTI AppendedData section was not found")
    raw = "".join(match.group(1).split())
    if not raw:
        raise ValueError("VTI AppendedData section is empty")
    if raw.startswith("_"):
        raw = raw[1:]
    return raw


def _decompress_vti_appended_segment(segment_base64: str, *, header_type: str) -> bytes:
    decoded = base64.b64decode(segment_base64)
    if header_type == "UInt32":
        fmt = "<I"
    elif header_type == "UInt64":
        fmt = "<Q"
    else:
        raise ValueError(f"Unsupported VTI header_type: {header_type}")

    hsize = struct.calcsize(fmt)
    if len(decoded) < hsize * 3:
        raise ValueError("Invalid appended segment: compressed header is too short")

    num_blocks = struct.unpack_from(fmt, decoded, 0)[0]
    block_size = struct.unpack_from(fmt, decoded, hsize)[0]
    last_block_size = struct.unpack_from(fmt, decoded, hsize * 2)[0]
    compressed_sizes = [
        struct.unpack_from(fmt, decoded, hsize * (3 + idx))[0] for idx in range(int(num_blocks))
    ]
    cursor = hsize * (3 + int(num_blocks))
    chunks: list[bytes] = []
    for compressed_size in compressed_sizes:
        size = int(compressed_size)
        chunk = decoded[cursor : cursor + size]
        cursor += size
        chunks.append(zlib.decompress(chunk))

    payload = b"".join(chunks)
    expected_size = int(block_size) * max(0, int(num_blocks) - 1) + int(last_block_size)
    if len(payload) != expected_size:
        raise ValueError(
            "Decoded VTI payload size mismatch: "
            f"expected={expected_size}, actual={len(payload)}"
        )
    return payload


def _to_zyx_local(
    raw_array: np.ndarray,
    dims_xyz: tuple[int, int, int],
    location: ArrayLocation,
    *,
    flat_layout: FlatArrayLayout,
) -> np.ndarray:
    array = np.asarray(raw_array)
    nx, ny, nz = (int(v) for v in dims_xyz)
    point_shape_xyz = (nx, ny, nz)
    cell_shape_xyz = (max(nx - 1, 1), max(ny - 1, 1), max(nz - 1, 1))
    candidates = [point_shape_xyz] if location == "point" else [point_shape_xyz, cell_shape_xyz]
    if array.ndim == 1:
        for shape_xyz in candidates:
            if int(np.prod(shape_xyz)) == int(array.size):
                if flat_layout == "vtk_x_fastest":
                    shape_zyx = (shape_xyz[2], shape_xyz[1], shape_xyz[0])
                    return np.reshape(array, shape_zyx, order="C")
                return np.reshape(array, shape_xyz, order="C").transpose(2, 1, 0)
    elif array.ndim == 3:
        for shape_xyz in candidates:
            shape_zyx = (shape_xyz[2], shape_xyz[1], shape_xyz[0])
            if tuple(array.shape) == shape_xyz:
                return array.transpose(2, 1, 0)
            if tuple(array.shape) == shape_zyx:
                return array
            if int(np.prod(shape_xyz)) == int(array.size):
                return np.reshape(array, shape_xyz, order="C").transpose(2, 1, 0)
    raise ValueError(
        f"Material array shape does not match expected VTI layout. "
        f"shape={array.shape}, dims_xyz={dims_xyz}, location={location}"
    )


def _coerce_scalar_labels_local(labels: np.ndarray) -> np.ndarray:
    if np.issubdtype(labels.dtype, np.integer):
        return labels.astype(np.int64, copy=False)
    if np.issubdtype(labels.dtype, np.floating):
        rounded = np.rint(labels)
        if np.allclose(labels, rounded, atol=0.0):
            return rounded.astype(np.int64)
    raise ValueError(f"Material array must be integer-valued, got dtype={labels.dtype}")
