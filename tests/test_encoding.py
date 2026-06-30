import csv

import numpy as np
from PIL import Image

from earthbridge.retrieval.encoding import encode_manifest


def test_encode_manifest_generates_descriptors(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.png"
    Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(image_path)

    manifest_path = tmp_path / "samples.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_path", "modality", "channels"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "OPT_0001",
                "image_path": "images/sample.png",
                "modality": "optical_rgb",
                "channels": "3",
            }
        )

    ids, descriptors = encode_manifest(
        manifest_path=manifest_path,
        root_dir=tmp_path,
        image_size=16,
        embedding_dim=8,
        model_type="baseline",
    )

    assert ids == ["OPT_0001"]
    assert descriptors.shape == (1, 8)
    assert np.isfinite(descriptors).all()

