from __future__ import annotations

import argparse
import json
import sys

import _path  # noqa: F401

from earthbridge.artifacts import verify_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify EarthBridge final demo artifacts.")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--shallow", action="store_true", help="Only check presence and file size.")
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Return exit code 0 even if invalid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_artifacts(args.artifact_root, deep=not args.shallow)
    print(json.dumps(report, indent=2))
    if not report["ok"] and not args.allow_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
