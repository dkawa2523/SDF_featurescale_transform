from __future__ import annotations

from pathlib import Path

from wafergeo.sem.spec import load_sem_prepare_spec_yaml, sem_prepare_spec_hash


def test_load_sem_prepare_spec_yaml_and_hash_deterministic(tmp_path: Path) -> None:
    yaml_text = """
schema_version: sem_prepare/v1
profile_id: sem_profile
target_shape_yx: [32, 48]
target_grid_2d:
  dim: 2
  spacing: [10.0, 10.0]
  origin: [0.0, 0.0]
  axis_order: YX
  sample_location: cell_center
  units: nm
input:
  contour_format: auto
  coord_system: pixel
  units: px
  pixel_size_nm: 2.5
  pixel_y_policy: flip_y
normalize:
  close_tol_nm: 3.0
  enforce_orientation: true
  resample_points_closed: 64
  resample_points_open: 32
tsdf:
  mode: auto
  mu_nm: 30.0
  open_tube_radius_nm: 12.0
  distance_backend: brute_sem_spec
weight:
  mode: uniform
  default_weight: 1.0
qa:
  min_mask_fraction: 0.01
  max_open_contours: 5
overlay:
  enable: true
  draw_contours: true
"""
    path = tmp_path / "sem_prepare.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    spec_a = load_sem_prepare_spec_yaml(path)
    spec_b = load_sem_prepare_spec_yaml(path)

    assert spec_a.profile_id == "sem_profile"
    assert spec_a.target_shape_yx == (32, 48)
    assert spec_a.input.pixel_y_policy == "flip_y"
    assert sem_prepare_spec_hash(spec_a) == sem_prepare_spec_hash(spec_b)
