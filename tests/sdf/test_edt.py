from __future__ import annotations

import numpy as np
import pytest

from tests.sdf.helpers import register_bruteforce_backend
from wafergeo.sdf.edt import get_edt_backend, signed_distance_from_mask
from wafergeo.sdf.engines.registry import get_sdf_engine
from wafergeo.sdf.errors import EDTBackendUnavailableError


def test_signed_distance_sign_convention() -> None:
    backend = register_bruteforce_backend()
    mask = np.zeros((1, 5, 5), dtype=bool)
    mask[0, 2, 2] = True

    phi = signed_distance_from_mask(mask, (1.0, 1.0, 1.0), backend)

    assert phi.shape == mask.shape
    assert phi[0, 2, 2] < 0.0
    assert phi[0, 0, 0] > 0.0


def test_unknown_backend_raises() -> None:
    mask = np.zeros((1, 3, 3), dtype=bool)
    with pytest.raises(EDTBackendUnavailableError):
        signed_distance_from_mask(mask, (1.0, 1.0, 1.0), "does-not-exist")


def test_legacy_callable_backend_still_works() -> None:
    backend = register_bruteforce_backend("legacy_callable")
    engine = get_sdf_engine(backend)
    distance_fn = get_edt_backend(backend)

    assert engine.version == "legacy"
    mask = np.array([[True, False]], dtype=bool)
    out = distance_fn(mask, (1.0, 1.0))
    assert out.shape == (1, 2)
