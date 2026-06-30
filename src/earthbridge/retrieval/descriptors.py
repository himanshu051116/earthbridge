from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DescriptorStore:
    ids: list[str]
    descriptors: np.ndarray

    def __post_init__(self) -> None:
        if self.descriptors.ndim != 2:
            raise ValueError("descriptors must be a 2D array")
        if len(self.ids) != len(self.descriptors):
            raise ValueError("ids and descriptors must have the same number of rows")


def save_descriptor_store(
    ids: Sequence[str],
    descriptors: np.ndarray,
    descriptors_path: str | Path,
    ids_path: str | Path,
) -> None:
    store = DescriptorStore(list(ids), np.asarray(descriptors, dtype=np.float32))

    descriptor_output = Path(descriptors_path)
    ids_output = Path(ids_path)
    descriptor_output.parent.mkdir(parents=True, exist_ok=True)
    ids_output.parent.mkdir(parents=True, exist_ok=True)

    np.save(descriptor_output, store.descriptors)
    with ids_output.open("w", encoding="utf-8") as handle:
        json.dump(store.ids, handle, indent=2)


def load_descriptor_store(
    descriptors_path: str | Path,
    ids_path: str | Path,
) -> DescriptorStore:
    descriptors = np.load(Path(descriptors_path)).astype("float32")
    with Path(ids_path).open("r", encoding="utf-8") as handle:
        ids = json.load(handle)
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError("ids file must contain a JSON list of strings")
    return DescriptorStore(ids=ids, descriptors=descriptors)

