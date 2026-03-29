from __future__ import annotations

import numpy as np
import pytest

from wafergeo.sdf.engines.registry import get_sdf_engine, register_default_sdf_engines
from wafergeo.sdf.errors import OptionalDependencyUnavailableError


def test_itk_backend_missing_dependency_error_is_helpful() -> None:
    register_default_sdf_engines()
    engine = get_sdf_engine("itk_maurer")

    with pytest.raises(OptionalDependencyUnavailableError) as exc_info:
        engine.distance(np.array([[True, False]], dtype=bool), (1.0, 1.0))

    message = str(exc_info.value)
    assert "itk" in message
    assert "wafergeo[itk]" in message


def test_cupy_backend_missing_dependency_error_is_helpful() -> None:
    register_default_sdf_engines()
    engine = get_sdf_engine("cupy_jfa")

    with pytest.raises(OptionalDependencyUnavailableError) as exc_info:
        engine.distance(np.array([[True, False]], dtype=bool), (1.0, 1.0))

    message = str(exc_info.value)
    assert "cupy" in message
    assert "wafergeo[cuda]" in message
