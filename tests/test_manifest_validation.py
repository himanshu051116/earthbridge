import csv

import numpy as np
from PIL import Image

from earthbridge.data.validation import validate_manifest


def test_validate_manifest_accepts_paired_labeled_rows(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(image_dir / "s2.png")
    Image.fromarray(np.full((8, 8), 64, dtype=np.uint8)).save(image_dir / "s1.png")

    manifest_path = tmp_path / "train.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_path", "modality", "pair_id", "labels"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "S2_A",
                "image_path": "images/s2.png",
                "modality": "multispectral",
                "pair_id": "P1",
                "labels": "water|urban",
            }
        )
        writer.writerow(
            {
                "sample_id": "S1_A",
                "image_path": "images/s1.png",
                "modality": "sar",
                "pair_id": "P1",
                "labels": "water|urban",
            }
        )

    report = validate_manifest(
        manifest_path,
        root_dir=tmp_path,
        left_modality="multispectral",
        right_modality="sar",
        min_pairs=1,
        require_labels=True,
    )

    assert report["ok"]
    assert report["pair_count"] == 1
    assert report["missing_files_examples"] == []


def test_validate_manifest_rejects_missing_pairs_and_files(tmp_path):
    manifest_path = tmp_path / "train.csv"
    manifest_path.write_text(
        "sample_id,image_path,modality,pair_id,labels\n"
        "S2_A,images/missing.png,multispectral,,\n",
        encoding="utf-8",
    )

    report = validate_manifest(
        manifest_path,
        root_dir=tmp_path,
        left_modality="multispectral",
        right_modality="sar",
        min_pairs=1,
        require_labels=True,
    )

    assert not report["ok"]
    assert report["pair_count"] == 0
    assert report["missing_files_examples"] == ["images/missing.png"]
    assert any("No labels found" in issue for issue in report["issues"])
