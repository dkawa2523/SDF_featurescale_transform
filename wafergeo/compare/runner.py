from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from wafergeo.compare.feature_outputs import (
    write_compare_feature_outputs,
    write_transform_feature_outputs,
)
from wafergeo.compare.features import (
    ViewFeature,
    contour_feature_on_grid,
    extract_view_feature,
)
from wafergeo.compare.loader import (
    CONTOUR_LOADERS,
    is_label_input_kind,
    load_simulation_label,
)
from wafergeo.compare.output_artifacts import (
    write_cd_profile_png,
    write_label_preview_png,
    write_material_confusion_outputs,
    write_per_material_sdf_csv,
)
from wafergeo.compare.render import (
    difference_summary,
    write_difference_legend_json,
    write_difference_png,
    write_difference_summary_json,
)
from wafergeo.compare.runtime_io import resolve_path, write_json, write_run_info
from wafergeo.compare.schema import (
    CompareSpec,
    TransformSpec,
    load_compare_spec_yaml,
    load_transform_spec_yaml,
)
from wafergeo.compare.scoring import ScoreResult, score_features, write_score_outputs
from wafergeo.compare.summaries import (
    summarize_label_volume,
    summarize_view_feature,
    write_json_summary,
)
from wafergeo.core.types import LabelVolume


@dataclass(frozen=True)
class PreparedTarget:
    label: LabelVolume | None
    feature: ViewFeature
    summary_payload: dict[str, object]


def run_transform_from_config(config_path: str | Path) -> dict[str, object]:
    config_file = Path(config_path).resolve()
    spec = load_transform_spec_yaml(config_file)
    base_dir = config_file.parent
    out_dir = resolve_path(spec.output.dir, base_dir=base_dir)
    return run_transform_spec(
        spec=spec,
        config_file=config_file,
        base_dir=base_dir,
        output_dir=out_dir,
        write_run_metadata=True,
    )


def run_transform_spec(
    *,
    spec: TransformSpec,
    config_file: Path,
    base_dir: Path,
    output_dir: Path,
    write_run_metadata: bool,
) -> dict[str, object]:
    out_dir = output_dir
    features_dir = out_dir / "features"
    sim_path = resolve_path(spec.simulation.path, base_dir=base_dir)
    label = load_simulation_label(spec.simulation.kind, sim_path, void_id=spec.simulation.void_id)
    reference_path: Path | None = None
    reference_label: LabelVolume | None = None
    if spec.process.enabled:
        if spec.reference is None:
            raise ValueError("process.enabled requires input.reference")
        reference_path = resolve_path(spec.reference.path, base_dir=base_dir)
        reference_label = load_simulation_label(
            spec.reference.kind,
            reference_path,
            void_id=spec.reference.void_id,
        )
        _require_compatible_label_volume(reference_label, label)
    view_feature = extract_view_feature(
        label,
        axes=spec.view.axes,
        depth_axis=spec.view.depth_axis,
    )
    written = write_transform_feature_outputs(
        label=label,
        view_feature=view_feature,
        feature_names=spec.features.use,
        output_dir=features_dir,
        reference_label=reference_label,
    )
    summary: dict[str, object] = {
        "task": "transform",
        "status": "OK",
        "input": {"kind": spec.simulation.kind, "path": str(sim_path)},
        "process": {"enabled": spec.process.enabled},
        "view": asdict(spec.view),
        "features": written,
        "feature_summary": "feature_summary.json",
        "label_summary": "label_summary.json",
        "output_dir": str(out_dir),
    }
    if reference_path is not None and spec.reference is not None:
        summary["reference"] = {
            "kind": spec.reference.kind,
            "path": str(reference_path),
            "void_id": spec.reference.void_id,
        }
    label_summary_payload: dict[str, object] = {
        "label_volume": summarize_label_volume(label),
        "view": summarize_view_feature(view_feature),
    }
    if reference_label is not None:
        label_summary_payload["reference_label_volume"] = summarize_label_volume(reference_label)
    write_json_summary(
        out_dir / "label_summary.json",
        label_summary_payload,
    )
    write_label_preview_png(
        out_dir / "preview.png",
        view_feature.label2d,
        void_id=view_feature.void_id,
    )
    write_json(out_dir / "summary.json", summary)
    if write_run_metadata:
        inputs = {"simulation": str(sim_path)}
        if reference_path is not None:
            inputs["reference"] = str(reference_path)
        write_run_info(
            config_path=config_file,
            output_dir=out_dir,
            task="transform",
            inputs=inputs,
        )
    return summary


def _require_compatible_label_volume(reference: LabelVolume, label: LabelVolume) -> None:
    if reference.material_id.shape != label.material_id.shape:
        raise ValueError(
            "input.reference and input.simulation shapes differ: "
            f"{reference.material_id.shape} != {label.material_id.shape}"
        )
    if reference.grid.axis_order != label.grid.axis_order:
        raise ValueError(
            "input.reference and input.simulation axis_order differ: "
            f"{reference.grid.axis_order} != {label.grid.axis_order}"
        )
    if not np.allclose(reference.grid.spacing, label.grid.spacing):
        raise ValueError(
            "input.reference and input.simulation spacing differ: "
            f"{reference.grid.spacing} != {label.grid.spacing}"
        )
    if not np.allclose(reference.grid.origin, label.grid.origin):
        raise ValueError(
            "input.reference and input.simulation origin differ: "
            f"{reference.grid.origin} != {label.grid.origin}"
        )
    if int(reference.material.void_id) != int(label.material.void_id):
        raise ValueError(
            "input.reference and input.simulation void_id differ: "
            f"{reference.material.void_id} != {label.material.void_id}"
        )


def _simulation_label_and_feature(
    spec: CompareSpec,
    sim_path: Path,
) -> tuple[LabelVolume, ViewFeature]:
    label = load_simulation_label(spec.simulation.kind, sim_path, void_id=spec.simulation.void_id)
    feature = extract_view_feature(
        label,
        axes=spec.view.axes,
        depth_axis=spec.view.depth_axis,
        contour_mode="outer" if spec.target.kind == "contour_json" else "material",
    )
    return label, feature


def _target_label_and_feature(
    spec: CompareSpec,
    *,
    target_path: Path,
    sim_feature: ViewFeature,
) -> tuple[LabelVolume | None, ViewFeature]:
    if spec.target.kind in CONTOUR_LOADERS:
        target_contours = CONTOUR_LOADERS[spec.target.kind](
            target_path,
            units_override=spec.target.units,
            view_axes=spec.view.axes,
        )
        return (
            None,
            contour_feature_on_grid(
                target_contours,
                sim_feature.grid2d,
                sim_feature.mask.shape,
                axes=spec.view.axes,
            ),
        )

    if not is_label_input_kind(spec.target.kind):
        raise ValueError(f"unsupported target kind: {spec.target.kind}")

    target_label = load_simulation_label(
        spec.target.kind,
        target_path,
        void_id=spec.target.void_id,
    )
    return (
        target_label,
        extract_view_feature(
            target_label,
            axes=spec.view.axes,
            depth_axis=spec.view.depth_axis,
            contour_mode="material",
        ),
    )


def prepare_label_target_for_compare(
    spec: CompareSpec,
    target_path: Path,
) -> PreparedTarget:
    if not is_label_input_kind(spec.target.kind):
        raise ValueError(f"unsupported reusable target kind: {spec.target.kind}")
    target_label = load_simulation_label(
        spec.target.kind,
        target_path,
        void_id=spec.target.void_id,
    )
    target_feature = extract_view_feature(
        target_label,
        axes=spec.view.axes,
        depth_axis=spec.view.depth_axis,
        contour_mode="material",
    )
    return PreparedTarget(
        label=target_label,
        feature=target_feature,
        summary_payload=build_input_summary_payload(feature=target_feature, label=target_label),
    )


def build_input_summary_payload(
    *,
    feature: ViewFeature,
    label: LabelVolume | None,
) -> dict[str, object]:
    payload: dict[str, object] = {"view": summarize_view_feature(feature)}
    if label is not None:
        payload["label_volume"] = summarize_label_volume(label)
    return payload


def _require_compatible_view_grid(sim_feature: ViewFeature, target_feature: ViewFeature) -> None:
    if sim_feature.axes != target_feature.axes:
        raise ValueError(
            "simulation and target view axes differ: "
            f"{sim_feature.axes} != {target_feature.axes}"
        )
    if sim_feature.label2d.shape != target_feature.label2d.shape:
        raise ValueError(
            "simulation and target projected view shapes differ: "
            f"{sim_feature.label2d.shape} != {target_feature.label2d.shape}. "
            "Resampling/alignment is not implicit in simple compare v1."
        )
    if not np.allclose(sim_feature.grid2d.spacing, target_feature.grid2d.spacing):
        raise ValueError(
            "simulation and target projected view spacing differ: "
            f"{sim_feature.grid2d.spacing} != {target_feature.grid2d.spacing}. "
            "Resampling/alignment is not implicit in simple compare v1."
        )
    if not np.allclose(sim_feature.grid2d.origin, target_feature.grid2d.origin):
        raise ValueError(
            "simulation and target projected view origin differ: "
            f"{sim_feature.grid2d.origin} != {target_feature.grid2d.origin}. "
            "Resampling/alignment is not implicit in simple compare v1."
        )


def _write_input_summary(
    *,
    output_dir: Path,
    prefix: str,
    feature: ViewFeature,
    label: LabelVolume | None,
    payload: dict[str, object] | None = None,
) -> str:
    filename = f"{prefix}_label_summary.json"
    write_json_summary(
        output_dir / filename,
        build_input_summary_payload(feature=feature, label=label) if payload is None else payload,
    )
    return filename


def _write_compare_features(
    *,
    sim_feature: ViewFeature,
    target_feature: ViewFeature,
    feature_names: set[str],
    output_dir: Path,
    write_target_features: bool = True,
) -> None:
    write_compare_feature_outputs(
        sim_feature=sim_feature,
        target_feature=target_feature if write_target_features else None,
        feature_names=feature_names,
        output_dir=output_dir,
    )


def _write_compare_difference(
    *,
    sim_feature: ViewFeature,
    target_feature: ViewFeature,
    output_dir: Path,
) -> dict[str, int | str]:
    sim_diff = sim_feature.label2d
    target_diff = target_feature.label2d
    sim_diff_mask = sim_feature.mask
    target_diff_mask = target_feature.mask
    legend_mode = "label"
    if (
        sim_feature.source == "label_volume"
        and target_feature.source == "label_volume"
        and sim_feature.boundary_mask is not None
        and target_feature.boundary_mask is not None
    ):
        sim_diff = sim_feature.boundary_mask
        target_diff = target_feature.boundary_mask
        sim_diff_mask = sim_feature.boundary_mask
        target_diff_mask = target_feature.boundary_mask
        legend_mode = "boundary"
    display_scale = write_difference_png(
        output_dir / "difference.png",
        sim_diff,
        target_diff,
        sim_mask=sim_diff_mask,
        target_mask=target_diff_mask,
    )
    write_difference_legend_json(
        output_dir / "difference_legend.json",
        mode=legend_mode,
        display_scale=display_scale,
    )
    write_difference_summary_json(
        output_dir / "difference_summary.json",
        sim_diff,
        target_diff,
        sim_mask=sim_diff_mask,
        target_mask=target_diff_mask,
        mode=legend_mode,
    )
    return difference_summary(
        sim_diff,
        target_diff,
        sim_mask=sim_diff_mask,
        target_mask=target_diff_mask,
        mode=legend_mode,
    )


def run_compare_spec(
    *,
    spec: CompareSpec,
    config_file: Path,
    base_dir: Path,
    output_dir: Path,
    write_run_metadata: bool,
    prepared_target: PreparedTarget | None = None,
    write_target_features: bool = True,
    write_case_outputs: bool = True,
) -> tuple[ScoreResult, dict[str, object]]:
    sim_path = resolve_path(spec.simulation.path, base_dir=base_dir)
    target_path = resolve_path(spec.target.path, base_dir=base_dir)
    sim_label, sim_feature = _simulation_label_and_feature(spec, sim_path)
    target_summary_payload: dict[str, object] | None = None
    if prepared_target is None:
        target_label, target_feature = _target_label_and_feature(
            spec,
            target_path=target_path,
            sim_feature=sim_feature,
        )
    else:
        target_label = prepared_target.label
        target_feature = prepared_target.feature
        target_summary_payload = prepared_target.summary_payload
    _require_compatible_view_grid(sim_feature, target_feature)
    feature_names = set(spec.features.use)
    score = score_features(sim_feature, target_feature, spec.metrics)
    sim_summary_file = ""
    target_summary_file = ""
    diff_summary: dict[str, int | str] | None = None
    if write_case_outputs:
        _write_compare_features(
            sim_feature=sim_feature,
            target_feature=target_feature,
            feature_names=feature_names,
            output_dir=output_dir,
            write_target_features=write_target_features,
        )
        write_score_outputs(score, output_dir)
        write_per_material_sdf_csv(output_dir / "per_material_sdf.csv", score.metric_details)
        write_material_confusion_outputs(
            output_dir,
            sim_feature=sim_feature,
            target_feature=target_feature,
        )
        write_cd_profile_png(output_dir / "cd_profile.png", score.cd_profile)
        sim_summary_file = _write_input_summary(
            output_dir=output_dir,
            prefix="simulation",
            feature=sim_feature,
            label=sim_label,
        )
        target_summary_file = _write_input_summary(
            output_dir=output_dir,
            prefix="target",
            feature=target_feature,
            label=target_label,
            payload=target_summary_payload,
        )
        if spec.output.difference_image:
            diff_summary = _write_compare_difference(
                sim_feature=sim_feature,
                target_feature=target_feature,
                output_dir=output_dir,
            )
    summary: dict[str, object] = {
        "task": "compare",
        "status": "OK",
        "total_score": score.total_score,
        "simulation": {"kind": spec.simulation.kind, "path": str(sim_path)},
        "target": {
            "kind": spec.target.kind,
            "path": str(target_path),
            "units": spec.target.units if spec.target.kind == "contour_json" else None,
            "void_id": spec.target.void_id,
        },
        "view": asdict(spec.view),
        "label_summaries": {
            "simulation": sim_summary_file,
            "target": target_summary_file,
        },
        "difference_summary": diff_summary,
        "output_dir": str(output_dir),
    }
    if write_run_metadata:
        write_run_info(
            config_path=config_file,
            output_dir=output_dir,
            task="compare",
            inputs={"simulation": str(sim_path), "target": str(target_path)},
        )
    return score, summary


def run_compare_from_config(config_path: str | Path) -> dict[str, object]:
    config_file = Path(config_path).resolve()
    spec = load_compare_spec_yaml(config_file)
    base_dir = config_file.parent
    out_dir = resolve_path(spec.output.dir, base_dir=base_dir)
    score, summary = run_compare_spec(
        spec=spec,
        config_file=config_file,
        base_dir=base_dir,
        output_dir=out_dir,
        write_run_metadata=True,
    )
    _ = score
    return summary
