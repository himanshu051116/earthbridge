from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

REQUIRED_COLUMNS = {"sample_id", "image_path", "modality"}


def parse_label_string(value: str | None) -> frozenset[str]:
    if not value:
        return frozenset()

    normalized = value.replace(",", "|").replace(";", "|")
    labels = {part.strip() for part in normalized.split("|") if part.strip()}
    return frozenset(labels)


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        validate_manifest_columns(reader.fieldnames)
        return [{key: value or "" for key, value in row.items()} for row in reader]


def write_manifest(path: str | Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def validate_manifest_columns(columns: Iterable[str]) -> None:
    available = set(columns)
    missing = REQUIRED_COLUMNS - available
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Manifest missing required columns: {missing_text}")


def sample_ids(rows: Iterable[dict[str, str]]) -> set[str]:
    return {row["sample_id"] for row in rows if row.get("sample_id")}

