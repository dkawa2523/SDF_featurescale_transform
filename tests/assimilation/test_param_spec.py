from __future__ import annotations

import numpy as np

from wafergeo.assimilation.types import ParamAxis, ParamSpec


def _build_param_spec() -> ParamSpec:
    return ParamSpec(
        axes=[
            ParamAxis(
                name="temperature",
                kind="continuous",
                bounds=(200.0, 600.0),
                transform="identity",
                default=400.0,
            ),
            ParamAxis(
                name="steps",
                kind="int",
                bounds=(1.0, 20.0),
                transform="identity",
                default=5,
            ),
            ParamAxis(
                name="mode",
                kind="categorical",
                choices=["A", "B", "C"],
                default="B",
            ),
        ],
        vector_order=["temperature", "steps", "mode"],
    )


def test_param_spec_encode_decode_deterministic() -> None:
    spec = _build_param_spec()
    params = {"temperature": 350.0, "steps": 7, "mode": "C"}
    x = spec.encode(params)
    decoded, warnings, oob = spec.decode(x, out_of_bounds_policy="clamp")
    assert np.allclose(x, spec.encode(params))
    assert warnings == []
    assert oob is False
    assert decoded["temperature"] == 350.0
    assert decoded["steps"] == 7
    assert decoded["mode"] == "C"


def test_param_spec_out_of_bounds_clamp_penalty_fail_flags() -> None:
    spec = _build_param_spec()
    x = np.array([1000.0, -5.0, 99.0], dtype=np.float64)

    params_clamp, warnings_clamp, oob_clamp = spec.decode(x, out_of_bounds_policy="clamp")
    assert oob_clamp is True
    assert len(warnings_clamp) >= 2
    assert float(params_clamp["temperature"]) <= 600.0
    assert int(params_clamp["steps"]) >= 1
    assert params_clamp["mode"] in {"A", "B", "C"}

    _params_penalty, warnings_penalty, oob_penalty = spec.decode(
        x,
        out_of_bounds_policy="penalty",
    )
    assert oob_penalty is True
    assert len(warnings_penalty) >= 2

    _params_fail, warnings_fail, oob_fail = spec.decode(x, out_of_bounds_policy="fail")
    assert oob_fail is True
    assert len(warnings_fail) >= 2
