from __future__ import annotations

import argparse

import _path  # noqa: F401

from earthbridge.data.inspection import inspect_dataset, summarize_inspections, write_dataset_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a raw multi-sensor satellite dataset.")
    parser.add_argument("--input", default="data/raw", help="Raw dataset root.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/reports",
        help="Report output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspections = inspect_dataset(args.input)
    write_dataset_report(args.output_dir, inspections)
    summary = summarize_inspections(inspections)

    print(f"Images: {summary['total_images']}")
    print(f"Modalities: {summary['modalities']}")
    print(f"Unreadable: {summary['unreadable_count']}")
    print(f"Wrote reports to {args.output_dir}")


if __name__ == "__main__":
    main()
