from __future__ import annotations

import argparse
from pathlib import Path

import _path  # noqa: F401

from earthbridge.data.bigearthnet import (
    BigEarthNetMetadataRecord,
    geographic_patch_key,
    has_bigearthnet_folders,
    load_bigearthnet_metadata,
    normalize_patch_key,
)
from earthbridge.data.inspection import ImageInspection, inspect_dataset, normalize_modality_name
from earthbridge.data.manifest import write_manifest

FIELDNAMES = [
    "sample_id",
    "image_path",
    "modality",
    "pair_id",
    "scene_id",
    "geographic_group",
    "labels",
    "country",
    "contains_seasonal_snow",
    "contains_cloud_or_shadow",
    "channels",
    "split",
]


def normalize_modality(raw: str) -> str:
    return normalize_modality_name(raw)


def image_path_for_manifest(root: Path, image_path: str | Path) -> str:
    return str(Path(image_path).relative_to(root.parent))


def sample_id(prefix: str, value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in value
    )
    return f"{prefix}_{safe}"


def inspections_by_modality_and_key(
    inspections: list[ImageInspection],
    key_mode: str = "patch",
) -> dict[str, dict[str, list[ImageInspection]]]:
    grouped: dict[str, dict[str, list[ImageInspection]]] = {}
    for item in inspections:
        if not item.readable:
            continue
        modality = normalize_modality(item.modality)
        if key_mode == "geographic":
            key = geographic_patch_key(Path(item.path).stem)
        else:
            key = normalize_patch_key(Path(item.path).stem)
        if not key:
            continue
        grouped.setdefault(modality, {}).setdefault(key, []).append(item)
    return grouped


def choose_image(
    images: list[ImageInspection],
    preferred_split: str = "",
) -> ImageInspection:
    if preferred_split:
        for image in images:
            if image.split == preferred_split:
                return image
    return images[0]


def labels_text(record: BigEarthNetMetadataRecord) -> str:
    return "|".join(record.labels)


def build_bigearthnet_rows(
    root: Path,
    inspections: list[ImageInspection],
    metadata: list[BigEarthNetMetadataRecord],
) -> list[dict[str, str]]:
    grouped = inspections_by_modality_and_key(inspections)
    s1_images = grouped.get("sar", {})
    s2_images = grouped.get("multispectral", {})
    rows: list[dict[str, str]] = []

    for record in metadata:
        if record.patch_id not in s2_images or record.s1_name not in s1_images:
            continue

        s2 = choose_image(s2_images[record.patch_id], preferred_split=record.split)
        split = record.split or s2.split
        s1 = choose_image(s1_images[record.s1_name], preferred_split=split)
        split = split or s1.split

        pair_id = record.patch_id
        labels = labels_text(record)
        rows.extend(
            [
                {
                    "sample_id": sample_id("S1", record.s1_name),
                    "image_path": image_path_for_manifest(root, s1.path),
                    "modality": "sar",
                    "pair_id": pair_id,
                    "scene_id": pair_id,
                    "geographic_group": pair_id,
                    "labels": labels,
                    "country": record.country,
                    "contains_seasonal_snow": str(record.contains_seasonal_snow),
                    "contains_cloud_or_shadow": str(record.contains_cloud_or_shadow),
                    "channels": str(s1.channels or ""),
                    "split": split,
                },
                {
                    "sample_id": sample_id("S2", record.patch_id),
                    "image_path": image_path_for_manifest(root, s2.path),
                    "modality": "multispectral",
                    "pair_id": pair_id,
                    "scene_id": pair_id,
                    "geographic_group": pair_id,
                    "labels": labels,
                    "country": record.country,
                    "contains_seasonal_snow": str(record.contains_seasonal_snow),
                    "contains_cloud_or_shadow": str(record.contains_cloud_or_shadow),
                    "channels": str(s2.channels or ""),
                    "split": split,
                },
            ]
        )

    return rows


def build_bigearthnet_geographic_rows(
    root: Path,
    inspections: list[ImageInspection],
) -> list[dict[str, str]]:
    grouped = inspections_by_modality_and_key(inspections, key_mode="geographic")
    s1_images = grouped.get("sar", {})
    s2_images = grouped.get("multispectral", {})
    rows: list[dict[str, str]] = []

    for key in sorted(set(s1_images) & set(s2_images)):
        known_splits = {
            image.split
            for image in [*s1_images[key], *s2_images[key]]
            if image.split in {"train", "validation", "test"}
        }
        candidate_splits = ("train", "validation", "test") if known_splits else ("",)
        for split in candidate_splits:
            s1_candidates = [
                image for image in s1_images[key] if not split or image.split == split
            ]
            s2_candidates = [
                image for image in s2_images[key] if not split or image.split == split
            ]
            if not s1_candidates or not s2_candidates:
                continue

            s1 = s1_candidates[0]
            s2 = s2_candidates[0]
            resolved_split = split or s2.split or s1.split
            pair_id = f"BEN_GEO_{key}"
            rows.extend(
                [
                    {
                        "sample_id": sample_id("S1", pair_id),
                        "image_path": image_path_for_manifest(root, s1.path),
                        "modality": "sar",
                        "pair_id": pair_id,
                        "scene_id": key,
                        "geographic_group": key,
                        "labels": "",
                        "country": "",
                        "contains_seasonal_snow": "",
                        "contains_cloud_or_shadow": "",
                        "channels": str(s1.channels or ""),
                        "split": resolved_split,
                    },
                    {
                        "sample_id": sample_id("S2", pair_id),
                        "image_path": image_path_for_manifest(root, s2.path),
                        "modality": "multispectral",
                        "pair_id": pair_id,
                        "scene_id": key,
                        "geographic_group": key,
                        "labels": "",
                        "country": "",
                        "contains_seasonal_snow": "",
                        "contains_cloud_or_shadow": "",
                        "channels": str(s2.channels or ""),
                        "split": resolved_split,
                    },
                ]
            )
            break

    return rows


def build_rows(input_root: str | Path) -> list[dict[str, str]]:
    root = Path(input_root)
    inspections = inspect_dataset(root)
    metadata = load_bigearthnet_metadata(root)
    if metadata:
        return build_bigearthnet_rows(root, inspections, metadata)

    bigearthnet_dataset = has_bigearthnet_folders(root)
    if bigearthnet_dataset:
        return build_bigearthnet_geographic_rows(root, inspections)

    rows: list[dict[str, str]] = []

    counters: dict[str, int] = {}
    for item in inspections:
        if not item.readable:
            continue

        modality = normalize_modality(item.modality)
        counters[modality] = counters.get(modality, 0) + 1
        stem = Path(item.path).stem
        sample_prefix = modality.upper().replace("_RGB", "").replace("_", "")
        sample_id = f"{sample_prefix}_{counters[modality]:06d}"
        pair_id = stem

        rows.append(
            {
                "sample_id": sample_id,
                "image_path": image_path_for_manifest(root, item.path),
                "modality": modality,
                "pair_id": pair_id,
                "scene_id": pair_id,
                "geographic_group": pair_id,
                "labels": "",
                "country": "",
                "contains_seasonal_snow": "",
                "contains_cloud_or_shadow": "",
                "channels": str(item.channels or ""),
                "split": item.split,
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a first canonical image manifest.")
    parser.add_argument("--input", default="data/raw", help="Raw dataset root.")
    parser.add_argument(
        "--output",
        default="data/manifests/samples.csv",
        help="Output manifest CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.input)
    write_manifest(args.output, rows, FIELDNAMES)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
