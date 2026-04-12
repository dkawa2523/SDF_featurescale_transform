"""Core contracts for wafergeo."""

from wafergeo.core.grid import AxisOrder, GridSpec, SampleLocation
from wafergeo.core.hashing import (
    canonical_json_dumps,
    hash_config,
    sha256_bytes,
    sha256_file,
)
from wafergeo.core.meta import Meta
from wafergeo.core.registry import (
    Registry,
    register_sdf_engine,
    sdf_backend_registry,
)
from wafergeo.core.types import (
    LabelVolume,
    MaterialSpec,
    MeshGeom,
    PointCloud,
    TSDFVolume,
)

__all__ = [
    "AxisOrder",
    "GridSpec",
    "SampleLocation",
    "Meta",
    "MaterialSpec",
    "LabelVolume",
    "TSDFVolume",
    "MeshGeom",
    "PointCloud",
    "Registry",
    "sdf_backend_registry",
    "register_sdf_engine",
    "sha256_bytes",
    "sha256_file",
    "canonical_json_dumps",
    "hash_config",
]
