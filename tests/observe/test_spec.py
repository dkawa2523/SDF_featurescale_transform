from __future__ import annotations

from pathlib import Path

from wafergeo.observe.spec import load_observer_spec_yaml, observer_spec_hash


def test_load_observer_spec_yaml_and_hash_is_deterministic(tmp_path: Path) -> None:
    yaml_text = """
schema_version: observer/v2
name: topdown_test
kind: topdown_exposed
target_grid_2d:
  dim: 2
  spacing: [10.0, 10.0]
  origin: [0.0, 0.0]
  axis_order: YX
  sample_location: cell_center
  units: nm
roi:
  x_min: 0.0
  x_max: 100.0
mask_definition:
  kind: exposed_union
  include_materials: [resist]
tsdf2d:
  mu_nm: 30.0
  engine: brute_observe_spec
contour:
  source: tsdf
  level: 0.0
  resample_points: 0
params: {}
debug: {}
qa: {}
"""
    spec_path = tmp_path / "observer.yaml"
    spec_path.write_text(yaml_text, encoding="utf-8")

    spec_a = load_observer_spec_yaml(spec_path)
    spec_b = load_observer_spec_yaml(spec_path)

    assert spec_a.kind == "topdown_exposed"
    assert spec_a.target_grid_2d.axis_order == "YX"
    assert observer_spec_hash(spec_a) == observer_spec_hash(spec_b)
