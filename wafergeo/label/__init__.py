"""Ingest/Label normalization layer."""

from wafergeo.label.artifact import build_label_artifact_payload, write_label_artifact
from wafergeo.label.errors import (
    InvalidArrayShapeError,
    InvalidGridMetaError,
    MissingLabelArrayError,
    PointToCellConversionError,
    UnknownMaterialIdError,
)
from wafergeo.label.materials import (
    load_materials_yaml,
    material_id_to_index,
    material_index_to_id,
)
from wafergeo.label.normalize import (
    LabelNormalizeConfig,
    MaskMergePolicy,
    PointToCellPolicy,
    UnknownLabelPolicy,
    convert_point_labels_to_cell_zyx,
    normalize_raw_to_label,
)
from wafergeo.label.qa import LabelQA, compute_label_qa

__all__ = [
    "LabelNormalizeConfig",
    "UnknownLabelPolicy",
    "PointToCellPolicy",
    "MaskMergePolicy",
    "LabelQA",
    "convert_point_labels_to_cell_zyx",
    "normalize_raw_to_label",
    "compute_label_qa",
    "load_materials_yaml",
    "material_id_to_index",
    "material_index_to_id",
    "build_label_artifact_payload",
    "write_label_artifact",
    "MissingLabelArrayError",
    "InvalidArrayShapeError",
    "UnknownMaterialIdError",
    "PointToCellConversionError",
    "InvalidGridMetaError",
]
