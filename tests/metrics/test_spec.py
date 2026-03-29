from __future__ import annotations

from pathlib import Path

from wafergeo.metrics.spec import (
    load_measurement_spec_yaml,
    load_metric_spec_yaml,
    measurement_spec_hash,
    metric_spec_hash,
)


def test_metric_spec_load_and_hash_deterministic(tmp_path: Path) -> None:
    metric_yaml = """
schema_version: metric/v2
metric_set_id: test_set
fail_penalty: 1000000.0
observer_weights:
  topdown: 2.0
metrics:
  - name: tsdf_band_robust_weight
    weight: 1.0
    observers: [topdown]
    params:
      band: obs_band
      robust: {type: huber, delta: 0.1}
  - name: cd_linescan
    weight: 0.5
    observers: [topdown]
    measurement_ref: cd_top
    params: {}
"""
    path = tmp_path / "metric.yaml"
    path.write_text(metric_yaml, encoding="utf-8")

    spec_a = load_metric_spec_yaml(path)
    spec_b = load_metric_spec_yaml(path)

    assert spec_a.metric_set_id == "test_set"
    assert len(spec_a.metrics) == 2
    assert metric_spec_hash(spec_a) == metric_spec_hash(spec_b)


def test_measurement_spec_load(tmp_path: Path) -> None:
    measurement_yaml = """
schema_version: measurement/v1
name: cd_top
lines:
  - id: l0
    axis: x
    coord_nm: 100.0
    range_nm: [50.0, 250.0]
    expected_edges: 2
    edge_pair: outer
    method: tsdf_zero_cross
"""
    path = tmp_path / "measurement.yaml"
    path.write_text(measurement_yaml, encoding="utf-8")

    spec = load_measurement_spec_yaml(path)

    assert spec.name == "cd_top"
    assert len(spec.lines) == 1
    assert measurement_spec_hash(spec)
