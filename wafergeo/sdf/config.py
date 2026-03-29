from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EDTBackendName = str
TSDFStoreDType = Literal["float16", "float32"]


@dataclass(frozen=True)
class SDFBuildConfig:
    mu_nm: float
    backend: EDTBackendName = "scipy"
    include_void_channel: bool = True
    boundary_features: bool = True
    pair_code_outside_band: int = 255
    band_only_pair_code: bool = True
    roi_zyx: tuple[slice, slice, slice] | None = None
    roi_margin_nm: float | None = None
    tsdf_store_dtype: TSDFStoreDType = "float16"
    compute_present_mask: bool = True
    qa_grad_check: bool = True
    qa_grad_tolerance: float = 0.2
    schema_version: str = "sdf/v1"
    profile_id: str = "sdf_build_v1"

    def to_hash_payload(self) -> dict[str, object]:
        roi_payload: object | None
        if self.roi_zyx is None:
            roi_payload = None
        else:
            roi_payload = [
                {
                    "start": s.start,
                    "stop": s.stop,
                    "step": s.step,
                }
                for s in self.roi_zyx
            ]

        return {
            "mu_nm": self.mu_nm,
            "backend": self.backend,
            "include_void_channel": self.include_void_channel,
            "boundary_features": self.boundary_features,
            "pair_code_outside_band": self.pair_code_outside_band,
            "band_only_pair_code": self.band_only_pair_code,
            "roi_zyx": roi_payload,
            "roi_margin_nm": self.roi_margin_nm,
            "tsdf_store_dtype": self.tsdf_store_dtype,
            "compute_present_mask": self.compute_present_mask,
            "qa_grad_check": self.qa_grad_check,
            "qa_grad_tolerance": self.qa_grad_tolerance,
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
        }
