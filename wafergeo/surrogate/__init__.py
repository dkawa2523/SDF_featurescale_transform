"""Surrogate data packaging layer (no training/inference execution)."""

from wafergeo.surrogate.builder import build_dataset_package
from wafergeo.surrogate.manifest_io import (
    SampleInputRow,
    load_input_manifest_json,
    read_dataset_manifest,
    write_dataset_manifest,
)
from wafergeo.surrogate.schema import DatasetManifest, SampleRecord, StorageMode, TaskKind
from wafergeo.surrogate.spec import (
    DatasetBuildSpecV1,
    DatasetQASpec,
    GroupSplitSpec,
    dataset_build_spec_hash,
    load_dataset_build_spec_yaml,
)
from wafergeo.surrogate.splits import make_group_split

__all__ = [
    "StorageMode",
    "TaskKind",
    "SampleRecord",
    "DatasetManifest",
    "SampleInputRow",
    "DatasetBuildSpecV1",
    "GroupSplitSpec",
    "DatasetQASpec",
    "load_dataset_build_spec_yaml",
    "dataset_build_spec_hash",
    "load_input_manifest_json",
    "write_dataset_manifest",
    "read_dataset_manifest",
    "make_group_split",
    "build_dataset_package",
]
