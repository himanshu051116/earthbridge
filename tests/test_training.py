import csv

import numpy as np
import torch
from PIL import Image

from earthbridge.training import TrainingConfig, train_paired_baseline


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

