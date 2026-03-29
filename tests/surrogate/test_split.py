from __future__ import annotations

from wafergeo.surrogate.manifest_io import SampleInputRow
from wafergeo.surrogate.spec import GroupSplitSpec
from wafergeo.surrogate.splits import make_group_split


def test_group_split_is_deterministic_and_no_group_leakage() -> None:
    rows = [
        SampleInputRow(sample_id="s0", group_id="g0"),
        SampleInputRow(sample_id="s1", group_id="g0"),
        SampleInputRow(sample_id="s2", group_id="g1"),
        SampleInputRow(sample_id="s3", group_id="g1"),
        SampleInputRow(sample_id="s4", group_id="g2"),
        SampleInputRow(sample_id="s5", group_id="g3"),
    ]
    split_spec = GroupSplitSpec(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=7)

    a = make_group_split(rows, split_spec)
    b = make_group_split(rows, split_spec)
    assert a == b

    sample_to_group = {row.sample_id: row.group_id for row in rows}
    groups_by_split = {
        key: {sample_to_group[sample_id] for sample_id in sample_ids}
        for key, sample_ids in a.items()
    }

    assert groups_by_split["train"].isdisjoint(groups_by_split["val"])
    assert groups_by_split["train"].isdisjoint(groups_by_split["test"])
    assert groups_by_split["val"].isdisjoint(groups_by_split["test"])
