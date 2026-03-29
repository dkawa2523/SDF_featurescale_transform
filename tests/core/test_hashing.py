from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wafergeo.core.hashing import canonical_json_dumps, hash_config


@dataclass(frozen=True)
class DummyConfig:
    alpha: int
    beta: list[int]


def test_canonical_json_dumps_is_order_independent_for_dict() -> None:
    left = {"b": 2, "a": 1}
    right = {"a": 1, "b": 2}
    assert canonical_json_dumps(left) == canonical_json_dumps(right)


def test_hash_config_is_stable_for_dataclass_and_ndarray() -> None:
    cfg1 = {"conf": DummyConfig(alpha=1, beta=[3, 4]), "arr": np.array([[1, 2], [3, 4]])}
    cfg2 = {"arr": np.array([[1, 2], [3, 4]]), "conf": DummyConfig(alpha=1, beta=[3, 4])}
    assert hash_config(cfg1) == hash_config(cfg2)
