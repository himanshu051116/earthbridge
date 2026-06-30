from __future__ import annotations

import argparse
from pathlib import Path

import _path  # noqa: F401

from earthbridge.data.inspection import inspect_dataset
from earthbridge.data.manifest import write_manifest

FIELDNAMES = [
    "sample_id",
    "image_path",
    "modality",
    "pair_id",
    "scene_id",
    "geographic_group",
    "labels",
    "channels",
]


def normalize_modality(raw: str) -> str:
    value = raw.lower().strip()
    if value in {"optical", "rgb", "sentinel2_rgb", "s2_rgb"}:
        return "optical_rgb"
    if value in {"sar", "sentinel1", "s1"}:
        return "sar"
    if value in {"multispectral", "sentinel2", "s2"}:
        return "multispectral"
    return value or "unknown"


def build_rows(input_root: str | Path) -> list[dict[str, str]]:
    root = Path(input_root)
    inspections = inspect_dataset(root)
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

        rows.append(
            {
                "sample_id": sample_id,
                "image_path": str(Path(item.path).relative_to(root.parent)),
                "modality": modality,
                "pair_id": stem,
                "scene_id": stem,
                "geographic_group": "",
                "labels": "",
                "channels": str(item.channels or ""),
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
