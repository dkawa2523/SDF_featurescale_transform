from __future__ import annotations

import random

from wafergeo.surrogate.manifest_io import SampleInputRow
from wafergeo.surrogate.spec import GroupSplitSpec


def make_group_split(
    records: list[SampleInputRow],
    split_spec: GroupSplitSpec,
) -> dict[str, list[str]]:
    if not records:
        raise ValueError("records is empty")

    groups: dict[str, list[str]] = {}
    for row in records:
        groups.setdefault(row.group_id, []).append(row.sample_id)

    group_keys = sorted(groups.keys())
    if len(group_keys) < 2:
        raise ValueError("group split requires at least 2 groups")

    rng = random.Random(split_spec.seed)
    rng.shuffle(group_keys)

    n_groups = len(group_keys)
    n_train = int(round(n_groups * split_spec.train_ratio))
    n_val = int(round(n_groups * split_spec.val_ratio))
    n_test = n_groups - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(1, n_groups - n_train - n_val)
        while n_train + n_val + n_test > n_groups:
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

    train_groups = set(group_keys[:n_train])
    val_groups = set(group_keys[n_train : n_train + n_val])
    test_groups = set(group_keys[n_train + n_val : n_train + n_val + n_test])

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for group_id, sample_ids in groups.items():
        if group_id in train_groups:
            splits["train"].extend(sample_ids)
        elif group_id in val_groups:
            splits["val"].extend(sample_ids)
        elif group_id in test_groups:
            splits["test"].extend(sample_ids)

    for key in ("train", "val", "test"):
        splits[key] = sorted(splits[key])

    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError("split creation failed: one of train/val/test is empty")

    overlap_train_val = set(splits["train"]).intersection(splits["val"])
    overlap_train_test = set(splits["train"]).intersection(splits["test"])
    overlap_val_test = set(splits["val"]).intersection(splits["test"])
    if overlap_train_val or overlap_train_test or overlap_val_test:
        overlap = sorted(overlap_train_val | overlap_train_test | overlap_val_test)
        raise ValueError(f"sample leakage detected across splits: {overlap}")

    return splits
