from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

DEFAULT_GROUP_KEYS = ("scene_id", "geographic_group", "pair_id", "sample_id")


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def as_tuple(self) -> tuple[float, float, float]:
        total = self.train + self.validation + self.test
        if total <= 0:
            raise ValueError("Split ratios must sum to a positive value")
        return self.train / total, self.validation / total, self.test / total


DEFAULT_SPLIT_RATIOS = SplitRatios()


def choose_group_key(row: dict[str, str], group_keys: Sequence[str] = DEFAULT_GROUP_KEYS) -> str:
    for key in group_keys:
        value = row.get(key, "").strip()
        if value:
            return f"{key}:{value}"
    raise ValueError(f"Row has no usable group key: {row}")


def grouped_split(
    rows: Sequence[dict[str, str]],
    ratios: SplitRatios | None = None,
    seed: int = 42,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
) -> dict[str, list[dict[str, str]]]:
    if not rows:
        return {"train": [], "validation": [], "test": []}

    ratios = ratios or DEFAULT_SPLIT_RATIOS
    groups = connected_groups(rows, group_keys)

    shuffled_groups = list(groups.values())
    rng = random.Random(seed)
    rng.shuffle(shuffled_groups)

    train_ratio, validation_ratio, _ = ratios.as_tuple()
    total_rows = len(rows)
    train_target = total_rows * train_ratio
    validation_target = total_rows * validation_ratio

    splits: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}

    for group_rows in shuffled_groups:
        if len(splits["train"]) < train_target:
            split_name = "train"
        elif len(splits["validation"]) < validation_target:
            split_name = "validation"
        else:
            split_name = "test"

        for row in group_rows:
            copied = dict(row)
            copied["split"] = split_name
            splits[split_name].append(copied)

    return splits


def connected_groups(
    rows: Sequence[dict[str, str]],
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
) -> dict[int, list[dict[str, str]]]:
    """Group rows by any shared leakage key.

    A row can share a scene with one sample and a geographic group with another.
    Treating those keys independently can still leak information across splits,
    so we form connected components across all non-empty grouping tokens.
    """

    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    token_owner: dict[str, int] = {}
    for row_index, row in enumerate(rows):
        tokens = [
            f"{key}:{row.get(key, '').strip()}"
            for key in group_keys
            if row.get(key, "").strip()
        ]
        if not tokens:
            tokens = [f"row:{row_index}"]

        for token in tokens:
            if token in token_owner:
                union(row_index, token_owner[token])
            else:
                token_owner[token] = row_index

    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        groups[find(row_index)].append(row)

    return groups


def non_empty_values(rows: Iterable[dict[str, str]], key: str) -> set[str]:
    return {row[key] for row in rows if row.get(key)}


def assert_no_group_leakage(
    splits: dict[str, Sequence[dict[str, str]]],
    keys: Sequence[str] = ("scene_id", "geographic_group", "pair_id"),
) -> None:
    split_names = list(splits)
    for key in keys:
        values_by_split = {name: non_empty_values(rows, key) for name, rows in splits.items()}
        for index, left_name in enumerate(split_names):
            for right_name in split_names[index + 1 :]:
                overlap = values_by_split[left_name] & values_by_split[right_name]
                if overlap:
                    examples = ", ".join(sorted(overlap)[:5])
                    raise AssertionError(
                        f"Leakage detected for {key} between {left_name} "
                        f"and {right_name}: {examples}"
                    )
