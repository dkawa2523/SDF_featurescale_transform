from __future__ import annotations

from dataclasses import asdict
from typing import Any

from wafergeo.core.types import LabelVolume
from wafergeo.io.artifact_store import ArtifactStore
from wafergeo.label.qa import LabelQA


def build_label_artifact_payload(label: LabelVolume, qa: LabelQA) -> dict[str, object]:
    return {
        "material_id": label.material_id,
        "grid": asdict(label.grid),
        "materials": asdict(label.material),
        "meta": label.meta.to_dict(),
        "qa": asdict(qa),
    }


def write_label_artifact(store: ArtifactStore, label: LabelVolume, qa: LabelQA) -> str:
    payload: dict[str, Any] = build_label_artifact_payload(label, qa)
    return store.write("label", payload, label.meta)
