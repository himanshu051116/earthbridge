from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from earthbridge.data.dataset import infer_modality_channels
from earthbridge.data.image_io import load_image_tensor
from earthbridge.data.manifest import load_manifest, parse_label_string


def find_paired_rows(
    rows: list[dict[str, str]],
    left_modality: str,
    right_modality: str,
    pair_key: str = "pair_id",
) -> list[tuple[dict[str, str], dict[str, str]]]:
    by_pair: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pair_id = row.get(pair_key, "").strip()
        modality = row.get("modality", "").strip()
        if pair_id and modality:
            by_pair[pair_id][modality].append(row)

    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for modalities in by_pair.values():
        left_rows = modalities.get(left_modality, [])
        right_rows = modalities.get(right_modality, [])
        for left_row, right_row in zip(left_rows, right_rows, strict=False):
            pairs.append((left_row, right_row))

    return pairs


class PairedImageDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        left_modality: str,
        right_modality: str,
        root_dir: str | Path = ".",
        image_size: int = 224,
        pair_key: str = "pair_id",
        modality_channels: dict[str, int] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir)
        self.left_modality = left_modality
        self.right_modality = right_modality
        self.image_size = image_size

        rows = load_manifest(self.manifest_path)
        self.modality_channels = modality_channels or infer_modality_channels(rows)
        self.pairs = find_paired_rows(rows, left_modality, right_modality, pair_key=pair_key)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        left_row, right_row = self.pairs[index]
        left_pair_id = left_row.get("pair_id", "")
        right_pair_id = right_row.get("pair_id", "")
        if left_pair_id != right_pair_id:
            raise ValueError(
                f"Pair ID mismatch for paired sample: {left_pair_id} != {right_pair_id}"
            )
        labels = tuple(
            sorted(
                parse_label_string(left_row.get("labels", ""))
                | parse_label_string(right_row.get("labels", ""))
            )
        )
        left_image = load_image_tensor(
            self.root_dir / left_row["image_path"],
            image_size=self.image_size,
            expected_channels=self.modality_channels[self.left_modality],
        )
        right_image = load_image_tensor(
            self.root_dir / right_row["image_path"],
            image_size=self.image_size,
            expected_channels=self.modality_channels[self.right_modality],
        )

        return {
            "left_id": left_row["sample_id"],
            "right_id": right_row["sample_id"],
            "pair_id": left_pair_id,
            "left_pair_id": left_pair_id,
            "right_pair_id": right_pair_id,
            "labels": labels,
            "left_modality": self.left_modality,
            "right_modality": self.right_modality,
            "left_image": left_image,
            "right_image": right_image,
        }


def paired_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    left_modalities = {str(item["left_modality"]) for item in batch}
    right_modalities = {str(item["right_modality"]) for item in batch}
    if len(left_modalities) != 1 or len(right_modalities) != 1:
        raise ValueError("paired_collate expects one left and one right modality per batch")

    left_pair_ids = [str(item["left_pair_id"]) for item in batch]
    right_pair_ids = [str(item["right_pair_id"]) for item in batch]
    if left_pair_ids != right_pair_ids:
        raise ValueError("paired_collate received misaligned left/right pair IDs")

    return {
        "left_ids": [item["left_id"] for item in batch],
        "right_ids": [item["right_id"] for item in batch],
        "pair_ids": left_pair_ids,
        "left_pair_ids": left_pair_ids,
        "right_pair_ids": right_pair_ids,
        "labels": [tuple(item.get("labels", ())) for item in batch],
        "left_modality": left_modalities.pop(),
        "right_modality": right_modalities.pop(),
        "left_image": torch.stack([torch.as_tensor(item["left_image"]) for item in batch]),
        "right_image": torch.stack([torch.as_tensor(item["right_image"]) for item in batch]),
    }
