from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from wafergeo.core.grid import GridSpec
from wafergeo.core.meta import Meta
from wafergeo.core.registry import sdf_backend_registry
from wafergeo.core.types import LabelVolume, MaterialSpec
from wafergeo.sdf.engines import EngineCapabilities, MethodCard, register_sdf_engine


def register_bruteforce_backend(name: str = "brute") -> str:
    def distance(mask: np.ndarray, sampling_zyx: tuple[float, ...]) -> np.ndarray:
        binary = np.asarray(mask, dtype=bool)
        out = np.zeros(binary.shape, dtype=np.float32)
        true_coords = np.argwhere(binary)
        false_coords = np.argwhere(~binary)
        if true_coords.size == 0 or false_coords.size == 0:
            return out

        sampling = np.asarray(sampling_zyx, dtype=np.float32)
        for coord in true_coords:
            delta = (false_coords - coord) * sampling
            squared = np.sum(delta * delta, axis=1)
            out[tuple(coord)] = float(np.sqrt(np.min(squared)))
        return out

    sdf_backend_registry.register(name, distance, override=True)
    return name


def register_bruteforce_engine(name: str = "brute_engine") -> str:
    class _BruteEngine:
        def __init__(self, engine_name: str) -> None:
            self.name = engine_name
            self.version = "1.0.0"
            self.capabilities = EngineCapabilities(
                supported_inputs=("binary_mask",),
                supports_2d=True,
                supports_3d=True,
                exact=False,
                deterministic=True,
                supports_anisotropic_spacing=True,
                supports_roi_margin=True,
                gpu_accelerated=False,
            )
            self.method_card = MethodCard(
                summary="Brute-force distance test engine.",
                dependencies=(),
                limitations=("Slow implementation for tests.",),
                references=(),
                recommended_use_cases=("Unit tests",),
                install_hint="",
            )

        def distance(self, mask: np.ndarray, sampling_zyx: tuple[float, ...]) -> np.ndarray:
            binary = np.asarray(mask, dtype=bool)
            out = np.zeros(binary.shape, dtype=np.float32)
            true_coords = np.argwhere(binary)
            false_coords = np.argwhere(~binary)
            if true_coords.size == 0 or false_coords.size == 0:
                return out

            sampling = np.asarray(sampling_zyx, dtype=np.float32)
            for coord in true_coords:
                delta = (false_coords - coord) * sampling
                squared = np.sum(delta * delta, axis=1)
                out[tuple(coord)] = float(np.sqrt(np.min(squared)))
            return out

    register_sdf_engine(_BruteEngine(name))
    return name


def build_material_spec() -> MaterialSpec:
    return MaterialSpec(
        ids=[0, 1, 2],
        names=["void", "resist", "oxide"],
        void_id=0,
        priority=[0, 10, 20],
        ignore_in_exposure=[True, False, False],
    )


def build_label_volume(material_id: np.ndarray) -> LabelVolume:
    grid = GridSpec(
        dim=3,
        spacing=(10.0, 10.0, 10.0),
        origin=(0.0, 0.0, 0.0),
        axis_order="ZYX",
        sample_location="cell_center",
        units="nm",
    )
    meta = Meta(
        schema_version="label/v1",
        profile_id="ingest_label_v1",
        config_hash="cfg",
        generator_version="0.1.0",
        git_commit="deadbeef",
        input_hash="input",
        created_at=datetime.now(UTC).isoformat(),
        extra={"source": "synthetic"},
    )
    return LabelVolume(
        grid=grid,
        material=build_material_spec(),
        material_id=np.asarray(material_id, dtype=np.uint8),
        meta=meta,
    )
