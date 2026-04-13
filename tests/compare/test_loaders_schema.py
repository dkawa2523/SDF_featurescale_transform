from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.compare.helpers import write_contour, write_npz, write_open_contour
from wafergeo.compare import (
    CONTOUR_LOADERS,
    LABEL_LOADERS,
    METRIC_DEFINITIONS,
    load_contour_json,
    load_simulation_label,
)
from wafergeo.compare.features import extract_view_feature
from wafergeo.compare.schema import (
    load_batch_compare_spec_yaml,
    load_compare_eval_spec_yaml,
    load_compare_spec_yaml,
    load_transform_spec_yaml,
)


def test_npz_label_loader_converts_xyz_to_internal_zyx(tmp_path: Path) -> None:
    npz = write_npz(tmp_path / "sim.npz")

    label = load_simulation_label("npz_label", npz)

    assert label.material_id.shape == (2, 8, 8)
    assert label.grid.axis_order == "ZYX"
    assert label.grid.spacing == (1.0, 1.0, 1.0)
    assert int(label.material_id[0, 2, 2]) == 1
    assert label.material.ids == [0, 1]


def test_npz_label_loader_respects_material_ids(tmp_path: Path) -> None:
    labels = np.full((4, 4, 1), 7, dtype=np.uint8)
    labels[:2, :, :] = 0
    npz = tmp_path / "sim_materials.npz"
    np.savez(
        npz,
        labels=labels,
        spacing=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        material_ids=np.array([0, 7], dtype=np.int32),
        material_names=np.array(["void", "film"]),
    )

    label = load_simulation_label("npz_label", npz)

    assert label.material.ids == [0, 7]
    assert label.material.names == ["void", "film"]


def test_npz_label_loader_rejects_unknown_material_id(tmp_path: Path) -> None:
    labels = np.array([[[0], [9]]], dtype=np.uint8)
    npz = tmp_path / "bad_materials.npz"
    np.savez(npz, labels=labels, material_ids=np.array([0, 1], dtype=np.int32))

    with pytest.raises(ValueError, match="not listed in material_ids"):
        load_simulation_label("npz_label", npz)


def test_npz_label_loader_requires_void_id_without_zero(tmp_path: Path) -> None:
    labels = np.ones((4, 4, 1), dtype=np.uint8)
    npz = tmp_path / "no_zero.npz"
    np.savez(npz, labels=labels, material_ids=np.array([1], dtype=np.int32))

    with pytest.raises(ValueError, match="void_id must be specified"):
        load_simulation_label("npz_label", npz)

    label = load_simulation_label("npz_label", npz, void_id=1)
    assert label.material.void_id == 1


def test_view_projection_uses_topmost_non_void_label(tmp_path: Path) -> None:
    labels = np.zeros((4, 4, 3), dtype=np.uint8)
    labels[1, 1, 0] = 1
    labels[1, 1, 2] = 2
    npz = tmp_path / "stacked.npz"
    np.savez(
        npz,
        labels=labels,
        spacing=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        material_ids=np.array([0, 1, 2], dtype=np.int32),
    )

    feature = extract_view_feature(load_simulation_label("npz_label", npz))

    assert int(feature.label2d[1, 1]) == 2


def test_contour_json_loader_projects_xyz_and_units_override(tmp_path: Path) -> None:
    contour = write_contour(tmp_path / "target.json", units="um")

    data = load_contour_json(contour, units_override="nm", view_axes=("x", "y"))

    assert data.units == "nm"
    assert data.contours[0].points_xy_nm.shape == (4, 2)
    np.testing.assert_allclose(data.contours[0].points_xy_nm[0], [1.5, 1.5])


def test_contour_json_loader_preserves_open_contour_flag(tmp_path: Path) -> None:
    contour = write_open_contour(tmp_path / "open_target.json")

    data = load_contour_json(contour, view_axes=("x", "y"))

    assert data.contours[0].closed is False
    assert data.contours[0].points_xy_nm.shape == (2, 2)


def test_contour_json_loader_rejects_bad_points(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "contour/v1",
                "contours": [{"id": "bad", "points": [[0.0], [1.0]]}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_contour_json(path)


def test_public_registries_include_initial_methods() -> None:
    assert set(LABEL_LOADERS) == {"npz_label", "vti_label"}
    assert set(CONTOUR_LOADERS) == {"contour_json"}
    assert set(METRIC_DEFINITIONS) == {
        "cd",
        "chamfer",
        "corner",
        "profile",
        "sdf",
        "sdf_material",
        "sdf_band",
        "iou",
        "topology",
    }
    assert METRIC_DEFINITIONS["cd"].required_features == frozenset({"contour"})
    assert METRIC_DEFINITIONS["chamfer"].required_features == frozenset({"contour"})
    assert METRIC_DEFINITIONS["corner"].required_features == frozenset({"contour"})
    assert METRIC_DEFINITIONS["profile"].required_features == frozenset({"contour"})
    assert METRIC_DEFINITIONS["sdf"].required_features == frozenset({"sdf"})
    assert METRIC_DEFINITIONS["sdf_material"].required_features == frozenset({"sdf"})
    assert METRIC_DEFINITIONS["sdf_band"].required_features == frozenset({"sdf"})
    assert METRIC_DEFINITIONS["iou"].required_features == frozenset()
    assert METRIC_DEFINITIONS["topology"].required_features == frozenset()
    assert METRIC_DEFINITIONS["cd"].loss_scale == 10.0
    assert METRIC_DEFINITIONS["corner"].loss_scale == 10.0
    assert METRIC_DEFINITIONS["profile"].loss_scale == 10.0
    assert METRIC_DEFINITIONS["iou"].loss_scale == 1.0
    assert METRIC_DEFINITIONS["topology"].loss_scale == 1.0


def test_simple_yaml_rejects_wrong_task_and_unknown_feature(tmp_path: Path) -> None:
    wrong_task = tmp_path / "wrong_task.yaml"
    wrong_task.write_text(
        """
task: transform
input:
  simulation:
    kind: npz_label
    path: sim.npz
  target:
    kind: contour_json
    path: target.json
output:
  dir: out
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="task must be 'compare'"):
        load_compare_spec_yaml(wrong_task)

    unknown_feature = tmp_path / "unknown_feature.yaml"
    unknown_feature.write_text(
        """
task: compare
input:
  simulation:
    kind: npz_label
    path: sim.npz
  target:
    kind: contour_json
    path: target.json
features:
  use: [sdf, magic]
output:
  dir: out
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported features"):
        load_compare_spec_yaml(unknown_feature)

    unknown_metric = tmp_path / "unknown_metric.yaml"
    unknown_metric.write_text(
        """
task: compare
input:
  simulation:
    kind: npz_label
    path: sim.npz
  target:
    kind: contour_json
    path: target.json
metrics:
  use: [not_a_metric]
output:
  dir: out
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported metrics"):
        load_compare_spec_yaml(unknown_metric)


def test_compare_yaml_rejects_transform_only_features(tmp_path: Path) -> None:
    config = tmp_path / "compare_mesh.yaml"
    config.write_text(
        """
task: compare
input:
  simulation:
    kind: npz_label
    path: sim.npz
  target:
    kind: contour_json
    path: target.json
features:
  use: [sdf, mesh]
output:
  dir: out
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="compare does not support"):
        load_compare_spec_yaml(config)

    tsdf_views_config = tmp_path / "compare_tsdf_views.yaml"
    tsdf_views_config.write_text(
        """
task: compare
input:
  simulation:
    kind: npz_label
    path: sim.npz
  target:
    kind: contour_json
    path: target.json
features:
  use: [tsdf_views]
output:
  dir: out
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="compare does not support"):
        load_compare_spec_yaml(tsdf_views_config)

    sdf_raw_config = tmp_path / "compare_sdf_raw.yaml"
    sdf_raw_config.write_text(
        """
task: compare
input:
  simulation:
    kind: npz_label
    path: sim.npz
  target:
    kind: contour_json
    path: target.json
features:
  use: [sdf_raw]
output:
  dir: out
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="compare does not support"):
        load_compare_spec_yaml(sdf_raw_config)

    udf_config = tmp_path / "compare_udf.yaml"
    udf_config.write_text(
        """
task: compare
input:
  simulation:
    kind: npz_label
    path: sim.npz
  target:
    kind: contour_json
    path: target.json
features:
  use: [udf]
output:
  dir: out
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="compare does not support"):
        load_compare_spec_yaml(udf_config)

    material_sdf_config = tmp_path / "compare_material_sdf.yaml"
    material_sdf_config.write_text(
        """
task: compare
input:
  simulation:
    kind: npz_label
    path: sim.npz
  target:
    kind: contour_json
    path: target.json
features:
  use: [material_sdf]
output:
  dir: out
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="compare does not support"):
        load_compare_spec_yaml(material_sdf_config)


def test_compare_yaml_rejects_metric_without_required_feature(tmp_path: Path) -> None:
    compare_config = tmp_path / "compare_missing_contour.yaml"
    compare_config.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {tmp_path / "sim.npz"}
  target:
    kind: contour_json
    path: {tmp_path / "target.json"}
features:
  use: [sdf]
metrics:
  use: [cd]
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feature 'contour' is required"):
        load_compare_spec_yaml(compare_config)

    batch_config = tmp_path / "batch_missing_sdf.yaml"
    batch_config.write_text(
        f"""
task: batch-compare
input:
  index: {tmp_path / "pairs.csv"}
features:
  use: [contour]
metrics:
  use: [sdf]
output:
  dir: {tmp_path / "batch"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feature 'sdf' is required"):
        load_batch_compare_spec_yaml(batch_config)

    compare_missing_profile = tmp_path / "compare_missing_profile_contour.yaml"
    compare_missing_profile.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {tmp_path / "sim.npz"}
  target:
    kind: npz_label
    path: {tmp_path / "target.npz"}
features:
  use: [sdf]
metrics:
  use: [profile]
output:
  dir: {tmp_path / "out_profile"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feature 'contour' is required"):
        load_compare_spec_yaml(compare_missing_profile)

    compare_missing_corner = tmp_path / "compare_missing_corner_contour.yaml"
    compare_missing_corner.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {tmp_path / "sim.npz"}
  target:
    kind: npz_label
    path: {tmp_path / "target.npz"}
features:
  use: [sdf]
metrics:
  use: [corner]
output:
  dir: {tmp_path / "out_corner"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feature 'contour' is required"):
        load_compare_spec_yaml(compare_missing_corner)


def test_compare_eval_yaml_accepts_candidates_and_validates_dependencies(tmp_path: Path) -> None:
    config = tmp_path / "compare_eval.yaml"
    config.write_text(
        f"""
task: compare-eval
input:
  index: {tmp_path / "pairs.csv"}
view:
  axes: [x, z]
  depth_axis: y
eval:
  candidates:
    primary:
      features:
        use: [sdf, contour]
      metrics:
        use: [cd, sdf, iou]
    material:
      features:
        use: [sdf, contour]
      metrics:
        use: [cd, sdf, iou, sdf_material]
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    spec = load_compare_eval_spec_yaml(config)

    assert spec.task == "compare-eval"
    assert tuple(spec.candidates) == ("primary", "material")
    assert spec.candidates["primary"].metrics.use == ("cd", "sdf", "iou")
    assert spec.candidates["material"].metrics.use == ("cd", "sdf", "iou", "sdf_material")

    missing_feature = tmp_path / "compare_eval_missing_feature.yaml"
    missing_feature.write_text(
        f"""
task: compare-eval
input:
  index: {tmp_path / "pairs.csv"}
eval:
  candidates:
    bad:
      features:
        use: [sdf]
      metrics:
        use: [cd]
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feature 'contour' is required"):
        load_compare_eval_spec_yaml(missing_feature)

    transform_feature = tmp_path / "compare_eval_transform_feature.yaml"
    transform_feature.write_text(
        f"""
task: compare-eval
input:
  index: {tmp_path / "pairs.csv"}
eval:
  candidates:
    bad:
      features:
        use: [sdf_raw]
      metrics:
        use: [sdf]
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="compare-eval does not support"):
        load_compare_eval_spec_yaml(transform_feature)


def test_compare_yaml_defaults_to_primary_metrics(tmp_path: Path) -> None:
    config = tmp_path / "compare_default_metrics.yaml"
    config.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {tmp_path / "sim.npz"}
  target:
    kind: npz_label
    path: {tmp_path / "target.npz"}
features:
  use: [sdf, contour]
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    spec = load_compare_spec_yaml(config)

    assert spec.metrics.use == ("cd", "sdf", "iou")

    batch_config = tmp_path / "batch_default_metrics.yaml"
    batch_config.write_text(
        f"""
task: batch-compare
input:
  index: {tmp_path / "pairs.csv"}
features:
  use: [sdf, contour]
output:
  dir: {tmp_path / "batch_out"}
""",
        encoding="utf-8",
    )

    batch_spec = load_batch_compare_spec_yaml(batch_config)

    assert batch_spec.metrics.use == ("cd", "sdf", "iou")


def test_compare_yaml_accepts_cd_gauge_and_rejects_view_mismatch(tmp_path: Path) -> None:
    config = tmp_path / "compare_cd_gauge.yaml"
    config.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {tmp_path / "sim.npz"}
  target:
    kind: npz_label
    path: {tmp_path / "target.npz"}
view:
  axes: [x, z]
  depth_axis: y
features:
  use: [contour]
metrics:
  use: [cd]
  cd:
    material_ids: [2]
    gauge:
      axis: x
      height_axis: z
      center: 4.0
      height_range: [2.0, 6.0]
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    spec = load_compare_spec_yaml(config)

    assert spec.metrics.cd_material_ids == (2,)
    assert spec.metrics.cd_gauge is not None
    assert spec.metrics.cd_gauge.axis == "x"
    assert spec.metrics.cd_gauge.height_axis == "z"
    assert spec.metrics.cd_gauge.center == 4.0
    assert spec.metrics.cd_gauge.height_range == (2.0, 6.0)

    mismatch = tmp_path / "compare_cd_gauge_bad_view.yaml"
    mismatch.write_text(
        f"""
task: compare
input:
  simulation:
    kind: npz_label
    path: {tmp_path / "sim.npz"}
  target:
    kind: npz_label
    path: {tmp_path / "target.npz"}
view:
  axes: [x, y]
  depth_axis: z
features:
  use: [contour]
metrics:
  use: [cd]
  cd:
    gauge:
      axis: x
      height_axis: z
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="height_axis must be included"):
        load_compare_spec_yaml(mismatch)


def test_transform_process_requires_reference(tmp_path: Path) -> None:
    missing_reference = tmp_path / "transform_process_missing_reference.yaml"
    missing_reference.write_text(
        f"""
task: transform
input:
  simulation:
    kind: npz_label
    path: {tmp_path / "final.npz"}
process:
  enabled: true
features:
  use: [sdf_raw]
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="process.enabled requires input.reference"):
        load_transform_spec_yaml(missing_reference)

    with_reference = tmp_path / "transform_process_reference.yaml"
    with_reference.write_text(
        f"""
task: transform
input:
  reference:
    kind: npz_label
    path: {tmp_path / "initial.npz"}
  simulation:
    kind: npz_label
    path: {tmp_path / "final.npz"}
process:
  enabled: true
features:
  use: [sdf_raw]
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    spec = load_transform_spec_yaml(with_reference)

    assert spec.process.enabled is True
    assert spec.reference is not None
    assert spec.reference.kind == "npz_label"
    assert spec.reference.path == str(tmp_path / "initial.npz")


def test_transform_process_feature_requires_process_mode(tmp_path: Path) -> None:
    config = tmp_path / "transform_process_feature_without_mode.yaml"
    config.write_text(
        f"""
task: transform
input:
  simulation:
    kind: npz_label
    path: {tmp_path / "final.npz"}
features:
  use: [process_delta_profile]
output:
  dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="process features require process.enabled"):
        load_transform_spec_yaml(config)
