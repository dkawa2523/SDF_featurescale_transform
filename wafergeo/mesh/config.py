from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MeshMode = Literal["interface_mesh", "material_shell"]
MeshBackend = Literal["vtk", "naive_interface"]
SampleSurfaceStrategy = Literal["area_uniform"]
PostprocessOnExceed = Literal["warn", "fail"]


@dataclass(frozen=True)
class MeshBuildConfig:
    mode: MeshMode = "interface_mesh"
    backend: MeshBackend = "naive_interface"
    iso_value: float = 0.0
    channel_material_ids: list[int] | None = None
    sample_points_n: int = 128
    sample_seed: int = 0
    sample_surface_strategy: SampleSurfaceStrategy = "area_uniform"
    vtk_viz_postprocess_enabled: bool = False
    vtk_smoothing_iterations: int = 20
    vtk_smoothing_pass_band: float = 0.05
    vtk_subdivision_levels: int = 1
    vtk_boundary_smoothing: bool = False
    vtk_feature_edge_smoothing: bool = False
    qa_degenerate_area_eps: float = 1e-12
    qa_max_bbox_shift_nm: float = 2.0
    qa_max_area_rel_error: float = 0.15
    qa_postprocess_on_exceed: PostprocessOnExceed = "warn"
    schema_version: str = "mesh/v1"
    profile_id: str = "mesh_build_v1"

    def to_hash_payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "backend": self.backend,
            "iso_value": self.iso_value,
            "channel_material_ids": self.channel_material_ids,
            "sample_points_n": self.sample_points_n,
            "sample_seed": self.sample_seed,
            "sample_surface_strategy": self.sample_surface_strategy,
            "vtk_viz_postprocess_enabled": self.vtk_viz_postprocess_enabled,
            "vtk_smoothing_iterations": self.vtk_smoothing_iterations,
            "vtk_smoothing_pass_band": self.vtk_smoothing_pass_band,
            "vtk_subdivision_levels": self.vtk_subdivision_levels,
            "vtk_boundary_smoothing": self.vtk_boundary_smoothing,
            "vtk_feature_edge_smoothing": self.vtk_feature_edge_smoothing,
            "qa_degenerate_area_eps": self.qa_degenerate_area_eps,
            "qa_max_bbox_shift_nm": self.qa_max_bbox_shift_nm,
            "qa_max_area_rel_error": self.qa_max_area_rel_error,
            "qa_postprocess_on_exceed": self.qa_postprocess_on_exceed,
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
        }
