from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import wafergeo
from wafergeo.compare.loader_types import LabelLoader
from wafergeo.core.grid import GridSpec
from wafergeo.core.hashing import hash_config, sha256_file
from wafergeo.core.meta import Meta
from wafergeo.core.types import LabelVolume, MaterialSpec
from wafergeo.io.vti_reader import (
    extract_material_ids,
    read_vti_with_xml_fallback,
    resolve_material_array_name,
)
from wafergeo.label.normalize import convert_point_labels_to_cell_zyx


def _meta(*, profile_id: str, input_hash: str, extra: dict[str, str] | None = None) -> Meta:
    return Meta(
        schema_version="label/v1",
        profile_id=profile_id,
        config_hash=hash_config({"profile_id": profile_id}),
        generator_version=wafergeo.__version__,
        git_commit="unknown",
        input_hash=input_hash,
        created_at=datetime.now(UTC).isoformat(),
        extra={} if extra is None else dict(extra),
    )


def _material_from_labels(
    labels_zyx: np.ndarray,
    material_ids: list[int] | None = None,
    material_names: list[str] | None = None,
    void_id: int | None = None,
) -> MaterialSpec:
    label_ids = sorted(int(v) for v in np.unique(labels_zyx).tolist())
    ids = list(material_ids) if material_ids is not None else label_ids
    if not ids:
        raise ValueError("material_ids must be non-empty")
    if len(set(ids)) != len(ids):
        raise ValueError("material_ids must be unique")
    unknown = sorted(set(label_ids).difference(ids))
    if unknown:
        raise ValueError(f"labels contain ids not listed in material_ids: {unknown}")
    if material_names is not None and len(material_names) == len(ids):
        names = [str(v) for v in material_names]
    else:
        names = [f"material_{mid}" for mid in ids]
    resolved_void_id = void_id
    if resolved_void_id is None:
        if 0 not in ids:
            raise ValueError("void_id must be specified when material id 0 is not present")
        resolved_void_id = 0
    if resolved_void_id not in ids:
        raise ValueError(f"void_id={resolved_void_id} must be listed in material_ids")
    return MaterialSpec(
        ids=ids,
        names=names,
        void_id=resolved_void_id,
        priority=list(range(len(ids))),
        ignore_in_exposure=[mid == resolved_void_id for mid in ids],
    )


def load_npz_label(path: str | Path, *, void_id: int | None = None) -> LabelVolume:
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as data:
        if "labels" not in data:
            raise ValueError("npz_label requires array 'labels' with shape [X,Y,Z]")
        labels_xyz = np.asarray(data["labels"])
        if labels_xyz.ndim != 3:
            raise ValueError(f"npz_label.labels must be 3D [X,Y,Z], got {labels_xyz.ndim}D")
        if not np.issubdtype(labels_xyz.dtype, np.integer):
            raise ValueError("npz_label.labels dtype must be integer")
        spacing_raw = data["spacing"] if "spacing" in data else [1.0, 1.0, 1.0]
        origin_raw = data["origin"] if "origin" in data else [0.0, 0.0, 0.0]
        spacing_xyz = np.asarray(spacing_raw, dtype=float)
        origin_xyz = np.asarray(origin_raw, dtype=float)
        if spacing_xyz.shape != (3,) or origin_xyz.shape != (3,):
            raise ValueError("npz_label.spacing and origin must be shape [3] in [X,Y,Z] order")
        material_names = None
        if "material_names" in data:
            material_names = [str(v) for v in data["material_names"].tolist()]
        material_ids = None
        if "material_ids" in data:
            material_ids = [int(v) for v in np.asarray(data["material_ids"]).tolist()]
        void_id_raw = void_id
        if void_id_raw is None and "void_id" in data:
            void_id_raw = int(np.asarray(data["void_id"]).reshape(()).item())

    labels_zyx = labels_xyz.transpose(2, 1, 0).astype(np.int64, copy=False)
    grid = GridSpec(
        dim=3,
        spacing=(float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0])),
        origin=(float(origin_xyz[2]), float(origin_xyz[1]), float(origin_xyz[0])),
        axis_order="ZYX",
        sample_location="cell_center",
        units="nm",
    )
    material = _material_from_labels(
        labels_zyx,
        material_ids=material_ids,
        material_names=material_names,
        void_id=void_id_raw,
    )
    return LabelVolume(
        grid=grid,
        material=material,
        material_id=labels_zyx,
        meta=_meta(
            profile_id="npz_label_simple_v1",
            input_hash=sha256_file(input_path),
            extra={"source_path": str(input_path), "source_axis_order": "XYZ"},
        ),
    )


def load_vti_label(path: str | Path, *, void_id: int | None = None) -> LabelVolume:
    input_path = Path(path)
    read_result = read_vti_with_xml_fallback(input_path)
    raw = read_result.raw
    array_name = resolve_material_array_name(raw)
    labels_zyx, location = extract_material_ids(raw, array_name)
    if location == "point":
        labels_zyx = convert_point_labels_to_cell_zyx(
            labels_zyx,
            policy="nearest",
        )
    labels_zyx = labels_zyx.astype(np.int64, copy=False)
    sx, sy, sz = (float(v) for v in raw.spacing_xyz)
    ox, oy, oz = (float(v) for v in raw.origin_xyz)
    grid = GridSpec(
        dim=3,
        spacing=(sz, sy, sx),
        origin=(oz + 0.5 * sz, oy + 0.5 * sy, ox + 0.5 * sx),
        axis_order="ZYX",
        sample_location="cell_center",
        units="nm",
    )
    material = _material_from_labels(labels_zyx, void_id=void_id)
    return LabelVolume(
        grid=grid,
        material=material,
        material_id=labels_zyx,
        meta=_meta(
            profile_id="vti_label_simple_v1",
            input_hash=sha256_file(input_path),
            extra={
                "source_path": str(input_path),
                "read_backend": read_result.backend_used,
                "source_array": array_name,
            },
        ),
    )


LABEL_LOADERS: dict[str, LabelLoader] = {
    "npz_label": load_npz_label,
    "vti_label": load_vti_label,
}


def is_label_input_kind(kind: str) -> bool:
    return kind in LABEL_LOADERS


def load_simulation_label(
    kind: str,
    path: str | Path,
    *,
    void_id: int | None = None,
) -> LabelVolume:
    if kind in LABEL_LOADERS:
        return LABEL_LOADERS[kind](path, void_id=void_id)
    raise ValueError(f"unsupported simulation kind: {kind}")
