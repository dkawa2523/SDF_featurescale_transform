from __future__ import annotations

import numpy as np

from tests.sdf.helpers import build_label_volume, register_bruteforce_backend
from wafergeo.sdf.boundary_features import pair_codebook
from wafergeo.sdf.build import build_tsdf_volume
from wafergeo.sdf.config import SDFBuildConfig


def _plane_label() -> np.ndarray:
    label = np.empty((1, 3, 4), dtype=np.uint8)
    label[:, :, :2] = 1
    label[:, :, 2:] = 2
    return label


def test_build_tsdf_shape_and_order() -> None:
    backend = register_bruteforce_backend()
    label = build_label_volume(_plane_label())
    cfg = SDFBuildConfig(mu_nm=20.0, backend=backend, band_only_pair_code=False)

    tsdf_volume, qa = build_tsdf_volume(label, cfg)

    assert tsdf_volume.tsdf.shape == (3, 1, 3, 4)
    assert tsdf_volume.meta is not None
    assert tsdf_volume.meta.extra["selected_material_ids"] == "0,1,2"
    assert qa.nan_count == 0
    assert qa.inf_count == 0

    # Left side is resist (channel 1), right side is oxide (channel 2).
    assert float(tsdf_volume.tsdf[1, 0, 1, 0]) <= 0.0
    assert float(tsdf_volume.tsdf[2, 0, 1, 0]) >= 0.0
    assert float(tsdf_volume.tsdf[2, 0, 1, 3]) <= 0.0


def test_boundary_features_and_present_mask() -> None:
    backend = register_bruteforce_backend()
    label = build_label_volume(_plane_label())
    cfg = SDFBuildConfig(mu_nm=20.0, backend=backend, band_only_pair_code=False)

    tsdf_volume, _ = build_tsdf_volume(label, cfg)

    assert tsdf_volume.d_boundary is not None
    assert tsdf_volume.pair_code is not None
    assert tsdf_volume.present_mask is not None
    assert tsdf_volume.present_mask.shape == (3,)
    assert tsdf_volume.present_mask.tolist() == [False, True, True]

    codebook = pair_codebook(tsdf_volume.tsdf.shape[0])
    expected_code = codebook[(1, 2)]
    interface_codes = np.unique(tsdf_volume.pair_code[0, :, 1:3])
    assert expected_code in interface_codes.tolist()


def test_roi_fill_and_shape() -> None:
    backend = register_bruteforce_backend()
    label = build_label_volume(_plane_label())
    cfg = SDFBuildConfig(
        mu_nm=20.0,
        backend=backend,
        roi_zyx=(slice(0, 1), slice(0, 3), slice(1, 3)),
        roi_margin_nm=10.0,
    )

    tsdf_volume, _ = build_tsdf_volume(label, cfg)

    assert tsdf_volume.tsdf.shape == (3, 1, 3, 4)
    np.testing.assert_allclose(tsdf_volume.tsdf[:, :, :, 0], 1.0)
    np.testing.assert_allclose(tsdf_volume.tsdf[:, :, :, 3], 1.0)

    assert tsdf_volume.pair_code is not None
    assert tsdf_volume.d_boundary is not None
    assert np.all(tsdf_volume.pair_code[:, :, 0] == 255)
    np.testing.assert_allclose(tsdf_volume.d_boundary[:, :, 0], 1.0)


def test_meta_contains_engine_signature() -> None:
    backend = register_bruteforce_backend("brute_signature")
    label = build_label_volume(_plane_label())
    cfg = SDFBuildConfig(mu_nm=20.0, backend=backend)

    tsdf_volume, _ = build_tsdf_volume(label, cfg)

    assert tsdf_volume.meta is not None
    assert tsdf_volume.meta.extra["sdf_engine_name"] == backend
    assert tsdf_volume.meta.extra["sdf_engine_version"] == "legacy"
    assert "sdf_engine_exact" in tsdf_volume.meta.extra
    assert "sdf_engine_gpu" in tsdf_volume.meta.extra
