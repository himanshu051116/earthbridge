from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from earthbridge.data.manifest import REQUIRED_COLUMNS, load_manifest
from earthbridge.data.pairs import find_paired_rows


def duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(value for value in values if value)
    return sorted(value for value, count in counts.items() if count > 1)


def validate_manifest(
    manifest_path: str | Path,
    root_dir: str | Path = ".",
    left_modality: str = "",
    right_modality: str = "",
    min_rows: int = 1,
    min_pairs: int = 0,
    require_labels: bool = False,
    check_files: bool = True,
) -> dict[str, Any]:
    path = Path(manifest_path)
    root = Path(root_dir)
    rows = load_manifest(path)
    issues: list[str] = []

    columns = set(rows[0].keys()) if rows else set()
    missing_required = sorted(REQUIRED_COLUMNS - columns)
    if missing_required:
        issues.append(f"Missing required columns: {', '.join(missing_required)}")

    if len(rows) < min_rows:
        issues.append(f"Expected at least {min_rows} rows, found {len(rows)}")

    sample_ids = [row.get("sample_id", "").strip() for row in rows]
    duplicate_sample_ids = duplicate_values(sample_ids)
    if duplicate_sample_ids:
        issues.append(f"Found duplicate sample_id values: {', '.join(duplicate_sample_ids[:10])}")

    modality_counts = Counter(row.get("modality", "").strip() or "unknown" for row in rows)
    split_counts = Counter(row.get("split", "").strip() or "unknown" for row in rows)
    label_rows = sum(1 for row in rows if row.get("labels", "").strip())
    if require_labels and rows and label_rows == 0:
        issues.append("No labels found, but labels are required")

    missing_files: list[str] = []
    if check_files:
        for row in rows:
            image_path = row.get("image_path", "").strip()
            if not image_path:
                missing_files.append("<empty image_path>")
                continue
            if not (root / image_path).exists():
                missing_files.append(image_path)
                if len(missing_files) >= 20:
                    break
        if missing_files:
            issues.append(f"Missing image files: {len(missing_files)} example(s)")

    pair_count = 0
    if left_modality and right_modality:
        pair_count = len(find_paired_rows(rows, left_modality, right_modality))
        if pair_count < min_pairs:
            issues.append(
                f"Expected at least {min_pairs} {left_modality}->{right_modality} pairs, "
                f"found {pair_count}"
            )

    return {
        "manifest_path": str(path),
        "root_dir": str(root),
        "row_count": len(rows),
        "columns": sorted(columns),
        "missing_required_columns": missing_required,
        "duplicate_sample_ids": duplicate_sample_ids,
        "modality_counts": dict(sorted(modality_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "label_rows": label_rows,
        "unlabeled_rows": len(rows) - label_rows,
        "left_modality": left_modality,
        "right_modality": right_modality,
        "pair_count": pair_count,
        "missing_files_examples": missing_files,
        "issues": issues,
        "ok": not issues,
    }
