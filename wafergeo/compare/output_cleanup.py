from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path


def remove_output_paths(output_dir: Path, names: Iterable[str]) -> None:
    for name in names:
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def clean_transform_output_dir(output_dir: Path) -> None:
    remove_output_paths(
        output_dir,
        (
            "features",
            "input_shape.png",
            "summary.json",
            "feature_summary.json",
            "label_summary.json",
        ),
    )


def clean_compare_output_dir(output_dir: Path) -> None:
    remove_output_paths(
        output_dir,
        (
            "features",
            "objective.json",
            "score.json",
            "metrics.csv",
            "metric_details.json",
            "per_material_sdf.csv",
            "material_confusion.csv",
            "difference.png",
            "difference_legend.json",
            "difference_summary.json",
            "simulation_label_summary.json",
            "target_label_summary.json",
            "cd_profile.csv",
            "cd_profile.png",
            "cd_profile_summary.json",
            "profile.csv",
            "profile_summary.json",
            "summary.json",
        ),
    )


def clean_batch_transform_output_dir(output_dir: Path) -> None:
    remove_output_paths(
        output_dir,
        (
            "cases",
            "dataset_index.csv",
            "features_summary.csv",
            "summary.json",
        ),
    )


def clean_batch_compare_output_dir(output_dir: Path) -> None:
    remove_output_paths(
        output_dir,
        (
            "cases",
            "differences",
            "shared_targets",
            "objectives.csv",
            "ranking.csv",
            "ranking_top.png",
            "metrics.csv",
            "metric_summary.csv",
            "score_summary.json",
            "difference_summary.csv",
            "per_material_sdf.csv",
            "material_confusion.csv",
        ),
    )
