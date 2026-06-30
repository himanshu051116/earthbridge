from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _path  # noqa: F401

from earthbridge.data.validation import validate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an EarthBridge manifest before training."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--left-modality", default="")
    parser.add_argument("--right-modality", default="")
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-pairs", type=int, default=0)
    parser.add_argument("--require-labels", action="store_true")
    parser.add_argument("--skip-file-check", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--allow-fail", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_manifest(
        manifest_path=args.manifest,
        root_dir=args.root_dir,
        left_modality=args.left_modality,
        right_modality=args.right_modality,
        min_rows=args.min_rows,
        min_pairs=args.min_pairs,
        require_labels=args.require_labels,
        check_files=not args.skip_file_check,
    )

    text = json.dumps(report, indent=2)
    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")

    if not report["ok"] and not args.allow_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
