from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from earthbridge.data.inspection import SPLIT_ALIASES, normalize_folder_name

METADATA_EXTENSIONS = {".csv", ".json", ".jsonl", ".parquet"}
METADATA_NAME_HINTS = {
    "metadata",
    "meta",
    "labels",
    "label",
    "patches",
    "pairs",
    "pair",
    "s1_s2",
    "s2_s1",
    "ben",
}

PATCH_ID_COLUMNS = (
    "patch_id",
    "patchid",
    "patch",
    "id",
    "s2",
    "s2_id",
    "s2_patch_id",
    "s2_patchid",
    "s2_name",
    "s2name",
    "sentinel2_patch_id",
    "sentinel_2_patch_id",
    "name",
)
S1_NAME_COLUMNS = (
    "s1",
    "s1_id",
    "s1_name",
    "s1name",
    "s1_patch_id",
    "s1_patchid",
    "sentinel1_patch_id",
    "sentinel_1_patch_id",
)
LABEL_COLUMNS = (
    "labels",
    "label",
    "land_cover_labels",
    "land_cover",
    "classes",
    "class_names",
    "clc_labels",
)
SPLIT_COLUMNS = ("split", "fold", "partition", "subset")


@dataclass(frozen=True)
class BigEarthNetMetadataRecord:
    patch_id: str
    s1_name: str
    labels: tuple[str, ...] = ()
    split: str = ""


def normalize_patch_key(value: str | None) -> str:
    if not value:
        return ""
    return Path(str(value).strip()).stem


def geographic_patch_key(value: str | None) -> str:
    """Return a sensor-independent patch key from a BigEarthNet-style filename."""
    stem = normalize_patch_key(value).upper()
    if not stem:
        return ""

    tokens = [token for token in re.split(r"[^A-Z0-9]+", stem) if token]
    tile_index = next(
        (index for index, token in enumerate(tokens) if re.fullmatch(r"\d{2}[A-Z]{3}", token)),
        None,
    )
    if tile_index is not None:
        suffix = tokens[tile_index:]
        numeric_tail = []
        for token in reversed(suffix[1:]):
            if token.isdigit():
                numeric_tail.append(token)
            else:
                break
        if numeric_tail:
            return "_".join([suffix[0], *reversed(numeric_tail)])
        return "_".join(suffix)

    stripped = [
        token
        for token in tokens
        if token
        not in {
            "S1",
            "S1A",
            "S1B",
            "S2",
            "S2A",
            "S2B",
            "SENTINEL1",
            "SENTINEL2",
            "MSIL1C",
            "MSIL2A",
            "IW",
            "GRDH",
            "1SDV",
            "VV",
            "VH",
        }
    ]
    return "_".join(stripped or tokens)


def normalize_column_name(value: str) -> str:
    return value.lower().strip().replace("-", "_").replace(" ", "_")


def normalize_row_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {normalize_column_name(str(key)): value for key, value in row.items()}


def first_value(row: dict[str, Any], columns: tuple[str, ...]) -> Any:
    for column in columns:
        if column in row and row[column] not in {None, ""}:
            return row[column]
    return None


def parse_labels(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, list | tuple | set):
        return tuple(str(item).strip() for item in value if str(item).strip())

    text = str(value).strip()
    if not text:
        return ()

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return tuple(str(item).strip() for item in parsed if str(item).strip())
        except json.JSONDecodeError:
            pass

    normalized = text.replace(";", "|").replace(",", "|")
    return tuple(part.strip() for part in normalized.split("|") if part.strip())


def normalize_split(value: Any) -> str:
    if value is None:
        return ""
    return SPLIT_ALIASES.get(normalize_folder_name(str(value)), "")


def metadata_record_from_row(row: dict[str, Any]) -> BigEarthNetMetadataRecord | None:
    normalized = normalize_row_keys(row)
    patch_id = normalize_patch_key(first_value(normalized, PATCH_ID_COLUMNS))
    s1_name = normalize_patch_key(first_value(normalized, S1_NAME_COLUMNS))
    if not patch_id or not s1_name:
        return None

    labels = parse_labels(first_value(normalized, LABEL_COLUMNS))
    split = normalize_split(first_value(normalized, SPLIT_COLUMNS))
    return BigEarthNetMetadataRecord(patch_id=patch_id, s1_name=s1_name, labels=labels, split=split)


def looks_like_metadata(path: Path) -> bool:
    if path.suffix.lower() not in METADATA_EXTENSIONS:
        return False
    normalized_name = normalize_folder_name(path.stem)
    return any(hint in normalized_name for hint in METADATA_NAME_HINTS)


def find_metadata_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    return sorted(
        path for path in root_path.rglob("*") if path.is_file() and looks_like_metadata(path)
    )


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    return pd.read_parquet(path).to_dict("records")


def load_metadata_file(path: Path) -> list[BigEarthNetMetadataRecord]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = load_csv_rows(path)
    elif suffix == ".json":
        rows = load_json_rows(path)
    elif suffix == ".jsonl":
        rows = load_jsonl_rows(path)
    elif suffix == ".parquet":
        rows = load_parquet_rows(path)
    else:
        rows = []

    records = [metadata_record_from_row(row) for row in rows]
    return [record for record in records if record is not None]


def load_bigearthnet_metadata(root: str | Path) -> list[BigEarthNetMetadataRecord]:
    records: list[BigEarthNetMetadataRecord] = []
    for path in find_metadata_files(root):
        try:
            records.extend(load_metadata_file(path))
        except Exception:
            continue

    deduped: dict[tuple[str, str], BigEarthNetMetadataRecord] = {}
    for record in records:
        deduped[(record.patch_id, record.s1_name)] = record
    return list(deduped.values())


def has_bigearthnet_folders(root: str | Path) -> bool:
    root_path = Path(root)
    folder_names = {
        normalize_folder_name(path.name) for path in root_path.rglob("*") if path.is_dir()
    }
    return "bigearthnet_s1" in folder_names or "bigearthnet-s1" in folder_names
