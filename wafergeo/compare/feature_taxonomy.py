from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureTaxonomy:
    code_name: str
    method: str
    target_shape: str
    output_kind: str
    role: str


FEATURE_TAXONOMY: dict[str, FeatureTaxonomy] = {
    "sdf_raw": FeatureTaxonomy(
        code_name="sdf_raw",
        method="sdf",
        target_shape="full_shape",
        output_kind="spatial_field_3d",
        role="sdf_like_feature",
    ),
    "tsdf_views": FeatureTaxonomy(
        code_name="tsdf_views",
        method="multi_scale_tsdf",
        target_shape="full_shape",
        output_kind="spatial_field_3d",
        role="sdf_like_feature",
    ),
    "udf": FeatureTaxonomy(
        code_name="udf",
        method="udf",
        target_shape="full_shape",
        output_kind="spatial_field_3d",
        role="sdf_like_feature",
    ),
    "material_sdf": FeatureTaxonomy(
        code_name="material_sdf",
        method="sdf",
        target_shape="material_shape",
        output_kind="spatial_field_3d",
        role="sdf_like_feature",
    ),
    "material_tsdf_views": FeatureTaxonomy(
        code_name="material_tsdf_views",
        method="multi_scale_tsdf",
        target_shape="material_shape",
        output_kind="spatial_field_3d",
        role="sdf_like_feature",
    ),
    "material_udf": FeatureTaxonomy(
        code_name="material_udf",
        method="udf",
        target_shape="material_shape",
        output_kind="spatial_field_3d",
        role="sdf_like_feature",
    ),
    "material_interface_relation": FeatureTaxonomy(
        code_name="material_interface_relation",
        method="interface_relation",
        target_shape="material_shape",
        output_kind="spatial_field_3d",
        role="derived_relation",
    ),
    "process_delta_sdf": FeatureTaxonomy(
        code_name="process_delta_sdf",
        method="sdf",
        target_shape="process_delta_shape",
        output_kind="spatial_field_3d",
        role="sdf_like_feature",
    ),
    "process_delta_tsdf_views": FeatureTaxonomy(
        code_name="process_delta_tsdf_views",
        method="multi_scale_tsdf",
        target_shape="process_delta_shape",
        output_kind="spatial_field_3d",
        role="sdf_like_feature",
    ),
    "process_delta_udf": FeatureTaxonomy(
        code_name="process_delta_udf",
        method="udf",
        target_shape="process_delta_shape",
        output_kind="spatial_field_3d",
        role="sdf_like_feature",
    ),
    "process_transition_relation": FeatureTaxonomy(
        code_name="process_transition_relation",
        method="transition_relation",
        target_shape="process_delta_shape",
        output_kind="spatial_field_3d",
        role="derived_relation",
    ),
}

TRANSFORM_EVAL_FEATURES: dict[tuple[str, str], str] = {
    ("full_shape", "sdf"): "sdf_raw",
    ("full_shape", "multi_scale_tsdf"): "tsdf_views",
    ("full_shape", "udf"): "udf",
    ("material_shape", "sdf"): "material_sdf",
    ("material_shape", "multi_scale_tsdf"): "material_tsdf_views",
    ("material_shape", "udf"): "material_udf",
    ("process_delta_shape", "sdf"): "process_delta_sdf",
    ("process_delta_shape", "multi_scale_tsdf"): "process_delta_tsdf_views",
    ("process_delta_shape", "udf"): "process_delta_udf",
}


def classify_feature(code_name: str) -> FeatureTaxonomy:
    try:
        return FEATURE_TAXONOMY[code_name]
    except KeyError as exc:
        raise ValueError(f"unclassified transform feature: {code_name}") from exc


def resolve_transform_eval_code_name(target_shape: str, method: str) -> str:
    key = (target_shape, method)
    try:
        return TRANSFORM_EVAL_FEATURES[key]
    except KeyError as exc:
        allowed = ", ".join(
            f"target_shape={shape}, method={name}"
            for shape, name in sorted(TRANSFORM_EVAL_FEATURES)
        )
        raise ValueError(
            "unsupported transform-eval feature combination: "
            f"target_shape={target_shape!r}, method={method!r}. "
            f"Supported combinations: {allowed}"
        ) from exc
