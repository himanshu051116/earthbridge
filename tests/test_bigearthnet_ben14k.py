from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from earthbridge.data.bigearthnet import metadata_record_from_row
from earthbridge.data.inspection import inspect_dataset


def write_tiff(path: Path, bands: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.arange(bands * 8 * 8, dtype=np.uint16).reshape(bands, 8, 8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=bands,
        dtype=data.dtype,
        transform=from_origin(0, 0, 1, 1),
    ) as dataset:
        dataset.write(data)


def load_build_manifest_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    module_path = scripts_dir / "build_manifest.py"
    spec = importlib.util.spec_from_file_location("build_manifest_for_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_inspect_ben14k_multiband_tiffs_with_rasterio(tmp_path):
    root = tmp_path / "BEN_14k"
    write_tiff(root / "train" / "BigEarthNet-S1" / "S1_PATCH_A.tif", bands=2)
    write_tiff(root / "train" / "BigEarthNet-S2" / "S2_PATCH_A.tif", bands=10)

    inspections = inspect_dataset(root)
    by_modality = {item.modality: item for item in inspections}

    assert len(inspections) == 2
    assert by_modality["sar"].channels == 2
    assert by_modality["sar"].width == 8
    assert by_modality["sar"].height == 8
    assert by_modality["sar"].split == "train"
    assert by_modality["multispectral"].channels == 10
    assert by_modality["multispectral"].readable


def test_ben14k_manifest_pairs_s1_s2_from_metadata(tmp_path):
    root = tmp_path / "BEN_14k"
    write_tiff(root / "validation" / "BigEarthNet-S1" / "S1_PATCH_A.tif", bands=2)
    write_tiff(root / "validation" / "BigEarthNet-S2" / "S2_PATCH_A.tif", bands=10)

    metadata_path = root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patch_id", "s1_name", "labels", "split"])
        writer.writeheader()
        writer.writerow(
            {
                "patch_id": "S2_PATCH_A",
                "s1_name": "S1_PATCH_A",
                "labels": "Agriculture|Water",
                "split": "validation",
            }
        )

    module = load_build_manifest_module()
    rows = module.build_rows(root)

    assert len(rows) == 2
    assert {row["modality"] for row in rows} == {"sar", "multispectral"}
    assert {row["pair_id"] for row in rows} == {"S2_PATCH_A"}
    assert {row["split"] for row in rows} == {"validation"}
    assert {row["labels"] for row in rows} == {"Agriculture|Water"}
    assert {row["channels"] for row in rows} == {"2", "10"}


def test_metadata_record_uses_patch_id_and_s1_name():
    record = metadata_record_from_row(
        {
            "patch_id": "S2_A.tif",
            "s1_name": "S1_A.tif",
            "labels": "Urban;Water",
            "split": "test",
        }
    )

    assert record is not None
    assert record.patch_id == "S2_A"
    assert record.s1_name == "S1_A"
    assert record.labels == ("Urban", "Water")
    assert record.split == "test"

