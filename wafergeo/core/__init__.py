"""Core contracts for wafergeo."""

from wafergeo.core.geometry import nearest_neighbor_distances_numpy, vtk_polys_to_triangles
from wafergeo.core.grid import AxisOrder, GridSpec, SampleLocation
from wafergeo.core.hashing import (
    canonical_json_dumps,
    hash_config,
    make_artifact_id,
    sha256_bytes,
    sha256_file,
)
from wafergeo.core.meta import Meta
from wafergeo.core.registry import (
    Registry,
    metric_registry,
    observer_registry,
    register_sdf_engine,
    report_extractor_registry,
    report_plot_registry,
    sdf_backend_registry,
    sem_reader_registry,
)
from wafergeo.core.types import (
    ContourLoop,
    LabelVolume,
    MaterialSpec,
    MeshGeom,
    Obs2D,
    ObserverSpec,
    PointCloud,
    Status,
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
    "ContourLoop",
    "Obs2D",
    "ObserverSpec",
    "Status",
    "Registry",
    "nearest_neighbor_distances_numpy",
    "vtk_polys_to_triangles",
    "sdf_backend_registry",
    "register_sdf_engine",
    "observer_registry",
    "metric_registry",
    "report_plot_registry",
    "report_extractor_registry",
    "sem_reader_registry",
    "sha256_bytes",
    "sha256_file",
    "canonical_json_dumps",
    "hash_config",
    "make_artifact_id",
]
