from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_IMAGE_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".jp2",
    ".bmp",
}


@dataclass(frozen=True)
class ImageInspection:
    path: str
    modality: str
    extension: str
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    mode: str = ""
    readable: bool = True
    error: str = ""


def iter_image_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root_path}")

    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def infer_modality(root: str | Path, image_path: str | Path) -> str:
    root_path = Path(root)
    path = Path(image_path)
    try:
        relative = path.relative_to(root_path)
    except ValueError:
        return path.parent.name

    return relative.parts[0] if len(relative.parts) > 1 else "unknown"


def inspect_image(root: str | Path, image_path: str | Path) -> ImageInspection:
    path = Path(image_path)
    modality = infer_modality(root, path)
    extension = path.suffix.lower()

    try:
        from PIL import Image
    except ImportError:
        return ImageInspection(
            path=str(path),
            modality=modality,
            extension=extension,
            readable=True,
            error="Pillow not installed; dimensions not inspected",
        )

    try:
        with Image.open(path) as image:
            channels = len(image.getbands()) if image.getbands() else None
            return ImageInspection(
                path=str(path),
                modality=modality,
                extension=extension,
                width=image.width,
                height=image.height,
                channels=channels,
                mode=image.mode,
            )
    except Exception as exc:
        return ImageInspection(
            path=str(path),
            modality=modality,
            extension=extension,
            readable=False,
            error=str(exc),
        )


def inspect_dataset(root: str | Path) -> list[ImageInspection]:
    return [inspect_image(root, path) for path in iter_image_files(root)]


def summarize_inspections(inspections: list[ImageInspection]) -> dict[str, Any]:
    modality_counts = Counter(item.modality for item in inspections)
    extension_counts = Counter(item.extension for item in inspections)
    shape_counts = Counter(
        f"{item.width}x{item.height}x{item.channels}"
        for item in inspections
        if item.width and item.height and item.channels
    )
    unreadable = [item for item in inspections if not item.readable]

    examples_by_modality: dict[str, list[str]] = defaultdict(list)
    for item in inspections:
        if len(examples_by_modality[item.modality]) < 5:
            examples_by_modality[item.modality].append(item.path)

    return {
        "total_images": len(inspections),
        "modalities": dict(sorted(modality_counts.items())),
        "extensions": dict(sorted(extension_counts.items())),
        "image_shapes": dict(sorted(shape_counts.items())),
        "unreadable_count": len(unreadable),
        "unreadable_examples": [
            {"path": item.path, "error": item.error} for item in unreadable[:20]
        ],
        "examples_by_modality": dict(sorted(examples_by_modality.items())),
    }


def write_dataset_report(output_dir: str | Path, inspections: list[ImageInspection]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = summarize_inspections(inspections)
    with (output_path / "dataset_report.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (output_path / "modality_distribution.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["modality", "count"])
        writer.writeheader()
        for modality, count in summary["modalities"].items():
            writer.writerow({"modality": modality, "count": count})

    with (output_path / "image_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "modality",
                "extension",
                "width",
                "height",
                "channels",
                "mode",
                "readable",
                "error",
            ],
        )
        writer.writeheader()
        for item in inspections:
            writer.writerow(
                {
                    "path": item.path,
                    "modality": item.modality,
                    "extension": item.extension,
                    "width": item.width or "",
                    "height": item.height or "",
                    "channels": item.channels or "",
                    "mode": item.mode,
                    "readable": item.readable,
                    "error": item.error,
                }
            )
