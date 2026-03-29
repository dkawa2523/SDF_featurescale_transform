from __future__ import annotations

import numpy as np

from tests.sdf.helpers import build_material_spec
from wafergeo.sdf.tsdf import from_tsdf, label_from_tsdf, to_tsdf


def test_to_tsdf_from_tsdf_roundtrip_band() -> None:
    phi = np.array([[-50.0, -10.0, 0.0, 10.0, 50.0]], dtype=np.float32)
    tsdf = to_tsdf(phi, mu_nm=20.0, out_dtype=np.float32)
    recovered = from_tsdf(tsdf, mu_nm=20.0)

    assert tsdf.dtype == np.float32
    assert float(tsdf.min()) >= -1.0
    assert float(tsdf.max()) <= 1.0
    np.testing.assert_allclose(recovered, np.clip(phi, -20.0, 20.0), atol=1e-6)


def test_label_from_tsdf_basic() -> None:
    material = build_material_spec()
    tsdf = np.ones((3, 1, 2, 2), dtype=np.float32)

    # Channel 1 (resist) dominates left column, channel 2 dominates right column.
    tsdf[1, 0, :, 0] = -0.1
    tsdf[2, 0, :, 1] = -0.2

    labels = label_from_tsdf(tsdf, material)

    assert labels.shape == (1, 2, 2)
    assert int(labels[0, 0, 0]) == 1
    assert int(labels[0, 0, 1]) == 2
    assert int(labels[0, 1, 0]) == 1
    assert int(labels[0, 1, 1]) == 2
