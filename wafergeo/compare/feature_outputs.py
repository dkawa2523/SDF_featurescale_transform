from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from wafergeo.compare.features import (
    ViewFeature,
    write_view_features,
)
from wafergeo.compare.transform_features import (
    write_label_sdf_feature,
    write_label_sdf_raw_feature,
    write_label_udf_feature,
    write_material_profile_feature,
    write_material_sdf_feature,
    write_mesh_feature,
    write_process_delta_profile_feature,
    write_process_delta_sdf_feature,
    write_transform_feature_summary,
    write_tsdf_views_feature,
)
from wafergeo.core.types import LabelVolume

TransformWriter = Callable[[LabelVolume, ViewFeature, Path, LabelVolume | None], dict[str, str]]


def _write_transform_sdf(
    _label: LabelVolume,
    view_feature: ViewFeature,
    output_dir: Path,
    _reference_label: LabelVolume | None,
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
    _reference_label: LabelVolume | None,
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
    _reference_label: LabelVolume | None,
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
    _reference_label: LabelVolume | None,
) -> dict[str, str]:
    return {"sdf3d": write_label_sdf_feature(label, output_dir)}


def _write_transform_sdf_raw(
    label: LabelVolume,
    _view_feature: ViewFeature,
    output_dir: Path,
    _reference_label: LabelVolume | None,
) -> dict[str, str]:
    return {"sdf_raw": write_label_sdf_raw_feature(label, output_dir)}


def _write_transform_tsdf_views(
    label: LabelVolume,
    _view_feature: ViewFeature,
    output_dir: Path,
    _reference_label: LabelVolume | None,
) -> dict[str, str]:
    return {"tsdf_views": write_tsdf_views_feature(label, output_dir)}


def _write_transform_udf(
    label: LabelVolume,
    _view_feature: ViewFeature,
    output_dir: Path,
    _reference_label: LabelVolume | None,
) -> dict[str, str]:
    return {"udf": write_label_udf_feature(label, output_dir)}


def _write_transform_material_sdf(
    label: LabelVolume,
    _view_feature: ViewFeature,
    output_dir: Path,
    _reference_label: LabelVolume | None,
) -> dict[str, str]:
    return {"material_sdf": write_material_sdf_feature(label, output_dir)}


def _write_transform_material_profile(
    label: LabelVolume,
    _view_feature: ViewFeature,
    output_dir: Path,
    _reference_label: LabelVolume | None,
) -> dict[str, str]:
    return write_material_profile_feature(label, output_dir)


def _write_transform_process_delta_profile(
    label: LabelVolume,
    _view_feature: ViewFeature,
    output_dir: Path,
    reference_label: LabelVolume | None,
) -> dict[str, str]:
    if reference_label is None:
        raise ValueError("process_delta_profile requires input.reference")
    return write_process_delta_profile_feature(
        reference_label=reference_label,
        final_label=label,
        output_dir=output_dir,
    )


def _write_transform_process_delta_sdf(
    label: LabelVolume,
    _view_feature: ViewFeature,
    output_dir: Path,
    reference_label: LabelVolume | None,
) -> dict[str, str]:
    if reference_label is None:
        raise ValueError("process_delta_sdf requires input.reference")
    return write_process_delta_sdf_feature(
        reference_label=reference_label,
        final_label=label,
        output_dir=output_dir,
    )


def _write_transform_mesh(
    label: LabelVolume,
    _view_feature: ViewFeature,
    output_dir: Path,
    _reference_label: LabelVolume | None,
) -> dict[str, str]:
    return write_mesh_feature(label, output_dir)


TRANSFORM_FEATURE_WRITERS: dict[str, TransformWriter] = {
    "sdf": _write_transform_sdf,
    "contour": _write_transform_contour,
    "slice": _write_transform_slice,
    "sdf_raw": _write_transform_sdf_raw,
    "sdf3d": _write_transform_sdf3d,
    "tsdf_views": _write_transform_tsdf_views,
    "udf": _write_transform_udf,
    "material_sdf": _write_transform_material_sdf,
    "material_profile": _write_transform_material_profile,
    "process_delta_profile": _write_transform_process_delta_profile,
    "process_delta_sdf": _write_transform_process_delta_sdf,
    "mesh": _write_transform_mesh,
}


def write_transform_feature_outputs(
    *,
    label: LabelVolume,
    view_feature: ViewFeature,
    feature_names: Iterable[str],
    output_dir: Path,
    reference_label: LabelVolume | None = None,
) -> dict[str, str]:
    written: dict[str, str] = {}
    names = list(feature_names)
    for name in names:
        writer = TRANSFORM_FEATURE_WRITERS[name]
        written.update(writer(label, view_feature, output_dir, reference_label))
    write_transform_feature_summary(
        label=label,
        output_dir=output_dir,
        written=written,
    )
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
