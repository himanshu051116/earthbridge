from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

import _path  # noqa: F401

from earthbridge.retrieval.descriptors import load_descriptor_store
from earthbridge.retrieval.faiss_index import ExactFaissIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark exact FAISS search latency.")
    parser.add_argument("--descriptors", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default="artifacts/reports/latency_summary.json")
    return parser.parse_args()


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percent / 100) * (len(ordered) - 1)))
    return ordered[index]


def main() -> None:
    args = parse_args()
    store = load_descriptor_store(args.descriptors, args.ids)
    index = ExactFaissIndex.build(store.ids, store.descriptors)

    latencies: list[float] = []
    query_count = min(args.queries, len(store.ids))
    for query_index in range(query_count):
        response = index.search(store.descriptors[query_index], top_k=args.top_k)
        latencies.append(response.search_time_ms)

    summary = {
        "query_count": query_count,
        "gallery_size": len(store.ids),
        "descriptor_dim": int(store.descriptors.shape[1]) if len(store.descriptors) else 0,
        "top_k": args.top_k,
        "mean_ms": mean(latencies) if latencies else 0.0,
        "median_ms": median(latencies) if latencies else 0.0,
        "p95_ms": percentile(latencies, 95),
        "index_type": "faiss.IndexFlatIP",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

