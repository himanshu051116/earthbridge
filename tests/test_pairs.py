import csv

import numpy as np
import pytest
from PIL import Image

from earthbridge.data.pairs import PairedImageDataset, find_paired_rows, paired_collate


def test_find_paired_rows_matches_requested_modalities():
    rows = [
        {"sample_id": "O1", "modality": "optical_rgb", "pair_id": "P1"},
        {"sample_id": "S1", "modality": "sar", "pair_id": "P1"},
        {"sample_id": "O2", "modality": "optical_rgb", "pair_id": "P2"},
    ]

    pairs = find_paired_rows(rows, "optical_rgb", "sar")

    assert len(pairs) == 1
    assert pairs[0][0]["sample_id"] == "O1"
    assert pairs[0][1]["sample_id"] == "S1"


def test_paired_image_dataset_loads_pair(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.full((12, 12, 3), 128, dtype=np.uint8)).save(image_dir / "opt.png")
    Image.fromarray(np.full((12, 12), 64, dtype=np.uint8)).save(image_dir / "sar.png")

    manifest_path = tmp_path / "samples.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_path", "modality", "pair_id", "labels", "channels"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "O1",
                "image_path": "images/opt.png",
                "modality": "optical_rgb",
                "pair_id": "P1",
                "labels": "Forest|Water",
                "channels": "3",
            }
        )
        writer.writerow(
            {
                "sample_id": "S1",
                "image_path": "images/sar.png",
                "modality": "sar",
                "pair_id": "P1",
                "labels": "Water|Urban",
                "channels": "1",
            }
        )

    dataset = PairedImageDataset(
        manifest_path,
        left_modality="optical_rgb",
        right_modality="sar",
        root_dir=tmp_path,
        image_size=16,
    )
    item = dataset[0]

    assert len(dataset) == 1
    assert item["left_id"] == "O1"
    assert item["right_id"] == "S1"
    assert item["pair_id"] == "P1"
    assert item["left_pair_id"] == "P1"
    assert item["right_pair_id"] == "P1"
    assert item["labels"] == ("Forest", "Urban", "Water")
    assert item["left_image"].shape == (3, 16, 16)
    assert item["right_image"].shape == (1, 16, 16)


def test_paired_collate_rejects_misaligned_pair_ids():
    batch = [
        {
            "left_id": "O1",
            "right_id": "S1",
            "left_pair_id": "P1",
            "right_pair_id": "P2",
            "left_modality": "optical_rgb",
            "right_modality": "sar",
            "left_image": np.zeros((3, 8, 8), dtype=np.float32),
            "right_image": np.zeros((1, 8, 8), dtype=np.float32),
        }
    ]

    with pytest.raises(ValueError, match="misaligned"):
        paired_collate(batch)
