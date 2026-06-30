from __future__ import annotations

import argparse
from pathlib import Path

import _path  # noqa: F401

from earthbridge.data.manifest import load_manifest, write_manifest
from earthbridge.data.splitting import SplitRatios, assert_no_group_leakage, grouped_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create leakage-free train/validation/test splits."
    )
    parser.add_argument("--manifest", required=True, help="Input canonical manifest CSV.")
    parser.add_argument(
        "--output-dir",
        default="data/manifests",
        help="Directory for split CSV files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    ratios = SplitRatios(args.train_ratio, args.validation_ratio, args.test_ratio)
    splits = grouped_split(rows, ratios=ratios, seed=args.seed)
    assert_no_group_leakage(splits)

    input_fields = list(rows[0].keys()) if rows else ["sample_id", "image_path", "modality"]
    fieldnames = input_fields if "split" in input_fields else [*input_fields, "split"]
    output_dir = Path(args.output_dir)

    for split_name, split_rows in splits.items():
        write_manifest(output_dir / f"{split_name}.csv", split_rows, fieldnames)

    combined_rows = [row for split_rows in splits.values() for row in split_rows]
    write_manifest(output_dir / "samples_with_splits.csv", combined_rows, fieldnames)

    print(
        "Created splits: "
        f"train={len(splits['train'])}, "
        f"validation={len(splits['validation'])}, "
        f"test={len(splits['test'])}"
    )


if __name__ == "__main__":
    main()
