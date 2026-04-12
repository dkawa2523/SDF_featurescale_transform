from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from wafergeo.compare.features import (
    ViewFeature,
    write_view_features,
)
from wafergeo.compare.transform_features import write_label_sdf_feature, write_mesh_feature
from wafergeo.core.types import LabelVolume

TransformWriter = Callable[[LabelVolume, ViewFeature, Path], dict[str, str]]


def _write_transform_sdf(
    _label: LabelVolume,
    view_feature: ViewFeature,
    output_dir: Path,
) -> dict[str, str]:
    written = write_view_features(
        feature=view_feature,
        output_dir=output_dir,
        prefix="simulation",
        include_sdf=True,
        include_contour=False,
    )
    return {"sdf": written["sdf"]}


def _write_transform_contour(
    _label: LabelVolume,
    view_feature: ViewFeature,
    output_dir: Path,
) -> dict[str, str]:
    written = write_view_features(
        feature=view_feature,
        output_dir=output_dir,
        prefix="simulation",
        include_sdf=False,
        include_contour=True,
    )
    return {"contour": written["contour"]}


def _write_transform_slice(
    _label: LabelVolume,
    view_feature: ViewFeature,
    output_dir: Path,
) -> dict[str, str]:
    written = write_view_features(
        feature=view_feature,
        output_dir=output_dir,
        prefix="simulation",
        include_sdf=False,
        include_contour=False,
        include_slice=True,
    )
    return {"slice": written["slice"]}


def _write_transform_sdf3d(
    label: LabelVolume,
    _view_feature: ViewFeature,
    output_dir: Path,
) -> dict[str, str]:
    return {"sdf3d": write_label_sdf_feature(label, output_dir)}


def _write_transform_mesh(
    label: LabelVolume,
    _view_feature: ViewFeature,
    output_dir: Path,
) -> dict[str, str]:
    return write_mesh_feature(label, output_dir)


TRANSFORM_FEATURE_WRITERS: dict[str, TransformWriter] = {
    "sdf": _write_transform_sdf,
    "contour": _write_transform_contour,
    "slice": _write_transform_slice,
    "sdf3d": _write_transform_sdf3d,
    "mesh": _write_transform_mesh,
}


def write_transform_feature_outputs(
    *,
    label: LabelVolume,
    view_feature: ViewFeature,
    feature_names: Iterable[str],
    output_dir: Path,
) -> dict[str, str]:
    written: dict[str, str] = {}
    names = list(feature_names)
    for name in names:
        writer = TRANSFORM_FEATURE_WRITERS[name]
        written.update(writer(label, view_feature, output_dir))
    return written


def write_compare_feature_outputs(
    *,
    sim_feature: ViewFeature,
    target_feature: ViewFeature | None,
    feature_names: Iterable[str],
    output_dir: Path,
) -> None:
    names = set(feature_names)
    features_dir = output_dir / "features"
    write_view_features(
        feature=sim_feature,
        output_dir=features_dir,
        prefix="simulation",
        include_sdf="sdf" in names,
        include_contour="contour" in names,
    )
    if target_feature is not None:
        write_target_feature_outputs(
            target_feature=target_feature,
            feature_names=names,
            output_dir=features_dir,
        )


def write_target_feature_outputs(
    *,
    target_feature: ViewFeature,
    feature_names: Iterable[str],
    output_dir: Path,
) -> None:
    names = set(feature_names)
    write_view_features(
        feature=target_feature,
        output_dir=output_dir,
        prefix="target",
        include_sdf="sdf" in names,
        include_contour="contour" in names,
    )
