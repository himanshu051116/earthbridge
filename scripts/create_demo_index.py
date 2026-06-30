from __future__ import annotations

import argparse
from pathlib import Path

import _path  # noqa: F401
import numpy as np

from earthbridge.retrieval.descriptors import save_descriptor_store
from earthbridge.retrieval.faiss_index import ExactFaissIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a tiny synthetic index for API smoke tests."
    )
    parser.add_argument("--output-dir", default="artifacts/demo", help="Demo artifact directory.")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)
    descriptors = rng.normal(size=(args.count, args.dim)).astype("float32")
    ids = [f"DEMO_{index:04d}" for index in range(args.count)]

    descriptors_path = output_dir / "descriptors.npy"
    descriptor_ids_path = output_dir / "descriptor_ids.json"
    index_path = output_dir / "demo.index"
    index_ids_path = output_dir / "demo_ids.json"

    save_descriptor_store(ids, descriptors, descriptors_path, descriptor_ids_path)
    ExactFaissIndex.build(ids, descriptors).save(index_path, index_ids_path)

    print(f"Wrote demo descriptors to {descriptors_path}")
    print(f"Wrote demo index to {index_path}")


if __name__ == "__main__":
    main()
