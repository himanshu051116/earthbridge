import csv

import numpy as np
import pytest
import rasterio
import torch
from PIL import Image

from earthbridge.training import TrainingConfig, train_paired_baseline
from earthbridge.training.trainer import assert_aligned_pair_ids


def write_training_raster(path, array: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=array.dtype,
    ) as dataset:
        dataset.write(array)


def shared_pair_pattern(pair_index: int, channels: int, size: int = 16) -> np.ndarray:
    block_size = size // 4
    rng = np.random.default_rng(20_000 + pair_index)
    code_a = rng.integers(0, 2, size=(4, 4), dtype=np.uint16) * 4095
    code_b = rng.integers(0, 2, size=(4, 4), dtype=np.uint16) * 4095
    code_a[0, 0] = np.uint16(pair_index % 2) * 4095
    code_b[0, 0] = np.uint16((pair_index // 2) % 2) * 4095
    base = np.repeat(np.repeat(code_a, block_size, axis=0), block_size, axis=1)
    flipped = np.repeat(np.repeat(code_b, block_size, axis=0), block_size, axis=1)
    bands = []
    for channel in range(channels):
        if channel == 0:
            band = base
        elif channel == 1:
            band = flipped
        else:
            band = np.roll(base if channel % 2 == 0 else flipped, shift=channel, axis=1)
        bands.append(band.astype(np.uint16))
    return np.stack(bands)


def test_train_paired_baseline_writes_checkpoint(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    manifest_path = tmp_path / "samples.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_path", "modality", "pair_id", "channels"],
        )
        writer.writeheader()
        for index in range(2):
            opt_path = image_dir / f"opt_{index}.png"
            sar_path = image_dir / f"sar_{index}.png"
            Image.fromarray(np.full((16, 16, 3), 80 + index, dtype=np.uint8)).save(opt_path)
            Image.fromarray(np.full((16, 16), 40 + index, dtype=np.uint8)).save(sar_path)
            writer.writerow(
                {
                    "sample_id": f"O{index}",
                    "image_path": f"images/opt_{index}.png",
                    "modality": "optical_rgb",
                    "pair_id": f"P{index}",
                    "channels": "3",
                }
            )
            writer.writerow(
                {
                    "sample_id": f"S{index}",
                    "image_path": f"images/sar_{index}.png",
                    "modality": "sar",
                    "pair_id": f"P{index}",
                    "channels": "1",
                }
            )

    checkpoint = tmp_path / "baseline.pt"
    result = train_paired_baseline(
        TrainingConfig(
            manifest_path=str(manifest_path),
            root_dir=str(tmp_path),
            image_size=16,
            embedding_dim=8,
            batch_size=2,
            epochs=1,
            output_checkpoint=str(checkpoint),
        )
    )

    payload = torch.load(checkpoint, map_location="cpu")

    assert result["pair_count"] == 2
    assert checkpoint.exists()
    assert "model_state_dict" in payload
    assert payload["metadata"]["pair_count"] == 2
    assert payload["metadata"]["best_validation"]["mean_recall_at_10"] >= 0.0


def test_assert_aligned_pair_ids_rejects_missing_or_misaligned_batch():
    with pytest.raises(ValueError, match="missing"):
        assert_aligned_pair_ids({})

    with pytest.raises(ValueError, match="misaligned"):
        assert_aligned_pair_ids(
            {
                "left_pair_ids": ["P1", "P2"],
                "right_pair_ids": ["P1", "P3"],
            }
        )


def test_tiny_cross_modal_training_overfits_exact_pairs(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    manifest_path = tmp_path / "samples.csv"
    pair_count = 128

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_path", "modality", "pair_id", "channels"],
        )
        writer.writeheader()
        for index in range(pair_count):
            pair_id = f"P{index:04d}"
            s1_path = image_dir / f"s1_{index:04d}.tif"
            s2_path = image_dir / f"s2_{index:04d}.tif"
            shared = shared_pair_pattern(index, channels=10)
            write_training_raster(s1_path, shared[:2])
            write_training_raster(s2_path, shared)
            writer.writerow(
                {
                    "sample_id": f"S2_{index:04d}",
                    "image_path": f"images/s2_{index:04d}.tif",
                    "modality": "multispectral",
                    "pair_id": pair_id,
                    "channels": "10",
                }
            )
            writer.writerow(
                {
                    "sample_id": f"S1_{index:04d}",
                    "image_path": f"images/s1_{index:04d}.tif",
                    "modality": "sar",
                    "pair_id": pair_id,
                    "channels": "2",
                }
            )

    checkpoint = tmp_path / "tiny_overfit.pt"
    result = train_paired_baseline(
        TrainingConfig(
            manifest_path=str(manifest_path),
            root_dir=str(tmp_path),
            left_modality="multispectral",
            right_modality="sar",
            validation_manifest_path=str(manifest_path),
            image_size=16,
            embedding_dim=128,
            projection_dropout=0.0,
            batch_size=128,
            epochs=100,
            learning_rate=0.001,
            weight_decay=0.0,
            temperature=0.02,
            semantic_loss_weight=0.0,
            hard_negative_loss_weight=1.0,
            hard_negative_margin=0.3,
            seed=42,
            output_checkpoint=str(checkpoint),
        )
    )

    validation = result["best_validation"]
    assert validation["multispectral_to_sar"]["recall_at_1"] >= 0.90
    assert validation["sar_to_multispectral"]["recall_at_1"] >= 0.90
    assert validation["multispectral_to_sar"]["recall_at_10"] >= 0.99
    assert validation["sar_to_multispectral"]["recall_at_10"] >= 0.99
