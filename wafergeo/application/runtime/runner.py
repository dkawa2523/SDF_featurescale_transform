from __future__ import annotations

from pathlib import Path

from wafergeo.compare import (
    run_batch_compare_from_config,
    run_compare_from_config,
    run_transform_from_config,
)

PUBLIC_PIPELINES = ("transform", "compare", "batch-compare")


def run_pipeline_from_config(pipeline: str, config_path: str | Path) -> dict[str, object]:
    """Run one of the simple public pipelines from a YAML config."""

    if pipeline == "transform":
        return run_transform_from_config(config_path)
    if pipeline == "compare":
        return run_compare_from_config(config_path)
    if pipeline == "batch-compare":
        return run_batch_compare_from_config(config_path)
    raise ValueError(
        "unsupported pipeline. Use one of: "
        + ", ".join(PUBLIC_PIPELINES)
    )
