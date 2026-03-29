from __future__ import annotations

from wafergeo.core.hashing import make_artifact_id


def test_make_artifact_id_is_deterministic() -> None:
    first = make_artifact_id("input1", "profile", "cfgA", "0.1.0")
    second = make_artifact_id("input1", "profile", "cfgA", "0.1.0")
    assert first == second


def test_make_artifact_id_changes_when_component_changes() -> None:
    first = make_artifact_id("input1", "profile", "cfgA", "0.1.0")
    second = make_artifact_id("input1", "profile", "cfgB", "0.1.0")
    assert first != second
