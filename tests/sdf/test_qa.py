from __future__ import annotations

import numpy as np

from tests.sdf.helpers import build_label_volume, register_bruteforce_backend
from wafergeo.sdf.build import build_tsdf_volume
from wafergeo.sdf.config import SDFBuildConfig


def test_sdf_qa_fields() -> None:
    backend = register_bruteforce_backend()
    label_data = np.zeros((1, 3, 4), dtype=np.uint8)
    label_data[:, :, :2] = 1
    label = build_label_volume(label_data)

    tsdf_volume, qa = build_tsdf_volume(label, SDFBuildConfig(mu_nm=20.0, backend=backend))

    assert qa.nan_count == 0
    assert qa.inf_count == 0
    assert -1.0 <= qa.tsdf_min <= 1.0
    assert -1.0 <= qa.tsdf_max <= 1.0
    assert 0.0 <= qa.band_fraction <= 1.0
    assert set(qa.present_materials.keys()) == {0, 1, 2}
    assert isinstance(qa.grad_mag_mean, float)
    assert isinstance(qa.grad_mag_std, float)
    assert isinstance(qa.grad_unit_error_rate, float)
    assert tsdf_volume.tsdf.shape[1] == 1
