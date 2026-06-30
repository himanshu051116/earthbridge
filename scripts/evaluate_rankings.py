from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _path  # noqa: F401

from earthbridge.data.manifest import load_manifest
from earthbridge.evaluation.evaluator import evaluate_rankings
from earthbridge.evaluation.relevance import RelevanceMode, SampleRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval rankings against a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--rankings",
        required=True,
        help="JSON rankings keyed by query/modality direction.",
    )
    parser.add_argument("--output", default="artifacts/reports/direction_metrics.csv")
    parser.add_argument(
        "--relevance-mode",
        choices=[mode.value for mode in RelevanceMode],
        default=RelevanceMode.SEMANTIC.value,
    )
    parser.add_argument("--semantic-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_rankings(path: str | Path) -> dict[tuple[str, str, str], list[str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    rankings: dict[tuple[str, str, str], list[str]] = {}
    for key, values in raw.items():
        query_id, query_modality, target_modality = key.split("|")
        rankings[(query_id, query_modality, target_modality)] = list(values)
    return rankings


def main() -> None:
    args = parse_args()
    records = [SampleRecord.from_row(row) for row in load_manifest(args.manifest)]
    rankings = load_rankings(args.rankings)
    scores = evaluate_rankings(
        records,
        rankings,
        RelevanceMode(args.relevance_mode),
        semantic_threshold=args.semantic_threshold,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_modality", "target_modality", "query_count", "f1_at_5", "f1_at_10"],
        )
        writer.writeheader()
        for score in scores:
            writer.writerow(
                {
                    "query_modality": score.query_modality,
                    "target_modality": score.target_modality,
                    "query_count": score.query_count,
                    "f1_at_5": score.f1_by_k.get(5, 0.0),
                    "f1_at_10": score.f1_by_k.get(10, 0.0),
                }
            )

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
