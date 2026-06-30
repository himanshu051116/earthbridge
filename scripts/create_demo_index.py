from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import torch
from PIL import Image

from earthbridge.data.image_io import load_image_tensor
from earthbridge.models import BaselineRetriever
from earthbridge.retrieval.descriptors import save_descriptor_store
from earthbridge.retrieval.faiss_index import ExactFaissIndex
from earthbridge.training.checkpointing import save_checkpoint

MANIFEST_FIELDS = ["sample_id", "image_path", "modality", "pair_id", "split", "labels", "channels"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create API-ready synthetic artifacts for local smoke demos."
    )
    parser.add_argument("--output-dir", default="artifacts/demo", help="Demo artifact directory.")
    parser.add_argument("--pairs", type=int, default=4, help="Number of synthetic scene pairs.")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def synthetic_bands(pair_index: int, channels: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + pair_index * 97 + channels)
    y, x = np.indices((size, size))
    base = (x * (pair_index + 3) + y * (pair_index + 5)) % 4096
    bands = []
    for channel in range(channels):
        noise = rng.integers(0, 96, size=(size, size))
        band = (base + channel * 277 + pair_index * 193 + noise) % 4096
        bands.append(band.astype(np.uint16))
    return np.stack(bands, axis=0)


def write_raster(path: Path, array: np.ndarray) -> None:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[2],
        height=array.shape[1],
        count=array.shape[0],
        dtype=array.dtype,
    ) as dataset:
        dataset.write(array)


def write_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.transpose(array[:3], (1, 2, 0))
    rgb = ((rgb.astype(np.float32) / max(float(rgb.max()), 1.0)) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def create_synthetic_images(
    output_dir: Path,
    pairs: int,
    image_size: int,
    seed: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    labels = ["water|urban", "forest|agriculture", "river|settlement", "crop|bare_soil"]
    image_dir = output_dir / "images"

    for pair_index in range(pairs):
        pair_id = f"DEMO_PAIR_{pair_index:04d}"
        label = labels[pair_index % len(labels)]

        ms_path = image_dir / f"pair_{pair_index:04d}_s2.tif"
        sar_path = image_dir / f"pair_{pair_index:04d}_s1.tif"
        rgb_path = image_dir / f"pair_{pair_index:04d}_rgb.png"

        multispectral = synthetic_bands(pair_index, channels=10, size=image_size, seed=seed)
        sar = synthetic_bands(pair_index, channels=2, size=image_size, seed=seed + 1000)

        write_raster(ms_path, multispectral)
        write_raster(sar_path, sar)
        write_rgb(rgb_path, multispectral)

        rows.extend(
            [
                {
                    "sample_id": f"DEMO_S2_{pair_index:04d}",
                    "image_path": str(ms_path.relative_to(output_dir)).replace("\\", "/"),
                    "modality": "multispectral",
                    "pair_id": pair_id,
                    "split": "test",
                    "labels": label,
                    "channels": "10",
                },
                {
                    "sample_id": f"DEMO_S1_{pair_index:04d}",
                    "image_path": str(sar_path.relative_to(output_dir)).replace("\\", "/"),
                    "modality": "sar",
                    "pair_id": pair_id,
                    "split": "test",
                    "labels": label,
                    "channels": "2",
                },
                {
                    "sample_id": f"DEMO_RGB_{pair_index:04d}",
                    "image_path": str(rgb_path.relative_to(output_dir)).replace("\\", "/"),
                    "modality": "optical_rgb",
                    "pair_id": pair_id,
                    "split": "test",
                    "labels": label,
                    "channels": "3",
                },
            ]
        )

    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def encode_rows(
    model: BaselineRetriever,
    rows: list[dict[str, str]],
    output_dir: Path,
    image_size: int,
    modality_channels: dict[str, int],
) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    descriptors: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for row in rows:
            modality = row["modality"]
            image = load_image_tensor(
                output_dir / row["image_path"],
                image_size=image_size,
                expected_channels=modality_channels[modality],
            ).unsqueeze(0)
            descriptor = model.encode(image, modality).cpu().numpy()[0]
            ids.append(row["sample_id"])
            descriptors.append(descriptor)

    return ids, np.vstack(descriptors).astype(np.float32)


def write_smoke_reports(output_dir: Path, rows: list[dict[str, str]]) -> None:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "note": "Synthetic smoke-demo artifacts only; not final hackathon metrics.",
        "sample_count": len(rows),
        "modalities": sorted({row["modality"] for row in rows}),
    }
    (reports_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    with (reports_dir / "direction_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_modality", "target_modality", "query_count", "f1_at_5", "f1_at_10"],
        )
        writer.writeheader()
        for query_modality in summary["modalities"]:
            for target_modality in summary["modalities"]:
                writer.writerow(
                    {
                        "query_modality": query_modality,
                        "target_modality": target_modality,
                        "query_count": len(
                            [row for row in rows if row["modality"] == query_modality]
                        ),
                        "f1_at_5": "",
                        "f1_at_10": "",
                    }
                )

    latency = {
        "note": "Synthetic smoke-demo latency placeholder; benchmark trained artifacts separately.",
        "queries": 0,
        "top_k": 10,
        "average_latency_ms": None,
    }
    (reports_dir / "latency_summary.json").write_text(
        json.dumps(latency, indent=2),
        encoding="utf-8",
    )


def create_demo_artifacts(
    output_dir: str | Path = "artifacts/demo",
    pairs: int = 4,
    image_size: int = 64,
    embedding_dim: int = 256,
    seed: int = 42,
) -> dict[str, str]:
    output_path = Path(output_dir)
    torch.manual_seed(seed)
    np.random.seed(seed)

    modality_channels = {"multispectral": 10, "sar": 2, "optical_rgb": 3}
    rows = create_synthetic_images(output_path, pairs=pairs, image_size=image_size, seed=seed)
    manifest_path = output_path / "manifests" / "test.csv"
    checkpoint_path = output_path / "checkpoints" / "baseline_pair.pt"
    descriptors_path = output_path / "descriptors" / "gallery.npy"
    descriptor_ids_path = output_path / "descriptors" / "gallery_ids.json"
    index_path = output_path / "indexes" / "gallery.index"
    index_ids_path = output_path / "indexes" / "gallery_ids.json"

    write_manifest(manifest_path, rows)

    model = BaselineRetriever(
        modality_channels=modality_channels,
        backbone_name="small_cnn",
        embedding_dim=embedding_dim,
    )
    save_checkpoint(
        checkpoint_path,
        model,
        metadata={
            "config": {
                "model_type": "baseline",
                "image_size": image_size,
                "embedding_dim": embedding_dim,
                "backbone": "small_cnn",
            },
            "history": [],
            "modality_channels": modality_channels,
            "pair_count": pairs,
            "smoke_demo": True,
        },
    )

    ids, descriptors = encode_rows(model, rows, output_path, image_size, modality_channels)
    save_descriptor_store(ids, descriptors, descriptors_path, descriptor_ids_path)
    ExactFaissIndex.build(ids, descriptors).save(index_path, index_ids_path)

    # Backward-compatible paths from the first smoke-test script.
    save_descriptor_store(
        ids,
        descriptors,
        output_path / "descriptors.npy",
        output_path / "descriptor_ids.json",
    )
    ExactFaissIndex.build(ids, descriptors).save(
        output_path / "demo.index",
        output_path / "demo_ids.json",
    )

    write_smoke_reports(output_path, rows)

    return {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "descriptors": str(descriptors_path),
        "descriptor_ids": str(descriptor_ids_path),
        "index": str(index_path),
        "index_ids": str(index_ids_path),
        "gallery_root": str(output_path),
    }


def main() -> None:
    args = parse_args()
    artifacts = create_demo_artifacts(
        output_dir=args.output_dir,
        pairs=args.pairs,
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )

    print(json.dumps(artifacts, indent=2))
    print("\nPowerShell API smoke-demo environment:")
    print(f"$env:EARTHBRIDGE_CHECKPOINT_PATH='{artifacts['checkpoint']}'")
    print(f"$env:EARTHBRIDGE_INDEX_PATH='{artifacts['index']}'")
    print(f"$env:EARTHBRIDGE_IDS_PATH='{artifacts['index_ids']}'")
    print(f"$env:EARTHBRIDGE_GALLERY_MANIFEST='{artifacts['manifest']}'")
    print(f"$env:EARTHBRIDGE_GALLERY_ROOT='{artifacts['gallery_root']}'")


if __name__ == "__main__":
    main()
