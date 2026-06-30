from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from earthbridge.data.image_io import load_image_tensor
from earthbridge.data.manifest import load_manifest

DEFAULT_CHANNELS = {
    "optical_rgb": 3,
    "sar": 2,
    "multispectral": 13,
}


def infer_modality_channels(rows: list[dict[str, str]]) -> dict[str, int]:
    channels: dict[str, int] = {}
    for row in rows:
        modality = row.get("modality", "")
        if not modality:
            continue

        raw_channels = row.get("channels", "").strip()
        if raw_channels.isdigit() and int(raw_channels) > 0:
            channels[modality] = max(channels.get(modality, 0), int(raw_channels))
        else:
            channels.setdefault(modality, DEFAULT_CHANNELS.get(modality, 3))

    return channels


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path = ".",
        image_size: int = 224,
        modality_channels: dict[str, int] | None = None,
        modality_filter: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir)
        rows = load_manifest(self.manifest_path)
        if modality_filter:
            rows = [row for row in rows if row.get("modality") == modality_filter]

        self.rows = rows
        self.image_size = image_size
        self.modality_channels = modality_channels or infer_modality_channels(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        modality = row["modality"]
        image_path = self.root_dir / row["image_path"]
        expected_channels = self.modality_channels[modality]
        image = load_image_tensor(image_path, self.image_size, expected_channels)

        return {
            "sample_id": row["sample_id"],
            "modality": modality,
            "image": image,
        }


def single_item_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    if len(batch) != 1:
        raise ValueError("single_item_collate expects batch_size=1")

    item = batch[0]
    return {
        "sample_id": item["sample_id"],
        "modality": item["modality"],
        "image": torch.as_tensor(item["image"]).unsqueeze(0),
    }

