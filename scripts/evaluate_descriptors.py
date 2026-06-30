from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _path  # noqa: F401

from earthbridge.data.manifest import load_manifest
from earthbridge.evaluation.descriptor_eval import evaluate_descriptors
from earthbridge.evaluation.relevance import RelevanceMode, SampleRecord
from earthbridge.retrieval.descriptors import load_descriptor_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate descriptor retrieval across directions.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--descriptors", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument(
        "--relevance-mode",
        choices=[mode.value for mode in RelevanceMode],
        default=RelevanceMode.SEMANTIC.value,
    )
    parser.add_argument("--semantic-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="artifacts/reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    records = [SampleRecord.from_row(row) for row in rows]
    store = load_descriptor_store(args.descriptors, args.ids)
    result = evaluate_descriptors(
        records=records,
        ids=store.ids,
        descriptors=store.descriptors,
        relevance_mode=RelevanceMode(args.relevance_mode),
        semantic_threshold=args.semantic_threshold,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    direction_path = output_dir / "direction_metrics.csv"
    with direction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_modality", "target_modality", "query_count", "f1_at_5", "f1_at_10"],
        )
        writer.writeheader()
        for score in result.direction_scores:
            writer.writerow(
                {
                    "query_modality": score.query_modality,
                    "target_modality": score.target_modality,
                    "query_count": score.query_count,
                    "f1_at_5": score.f1_by_k.get(5, 0.0),
                    "f1_at_10": score.f1_by_k.get(10, 0.0),
                }
            )

    summary = {
        "query_count": result.query_count,
        "mean_search_latency_ms": result.mean_search_latency_ms,
        "directions": [
            {
                "query_modality": score.query_modality,
                "target_modality": score.target_modality,
                "query_count": score.query_count,
                "f1_by_k": score.f1_by_k,
            }
            for score in result.direction_scores
        ],
    }
    summary_path = output_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote {direction_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

