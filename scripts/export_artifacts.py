from __future__ import annotations

import argparse
import json

import _path  # noqa: F401

from earthbridge.artifacts import export_artifact_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export final EarthBridge demo artifacts to a zip."
    )
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--output", default="artifacts/earthbridge_export.zip")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Export whatever exists instead of failing on missing artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_artifact_bundle(
        artifact_root=args.artifact_root,
        output_zip=args.output,
        allow_missing=args.allow_missing,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
