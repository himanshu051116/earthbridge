from __future__ import annotations

import argparse
from pathlib import Path

import _path  # noqa: F401

from earthbridge.retrieval.descriptors import load_descriptor_store
from earthbridge.retrieval.faiss_index import ExactFaissIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an exact FAISS index from descriptors.")
    parser.add_argument("--descriptors", required=True, help="Input .npy descriptor matrix.")
    parser.add_argument("--ids", required=True, help="Input JSON ID list aligned with descriptors.")
    parser.add_argument("--output-index", required=True, help="Output FAISS .index path.")
    parser.add_argument(
        "--output-ids",
        default="",
        help="Output ID list path. Defaults to <output-index stem>_ids.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = load_descriptor_store(args.descriptors, args.ids)
    index = ExactFaissIndex.build(store.ids, store.descriptors)

    output_index = Path(args.output_index)
    output_ids = Path(args.output_ids) if args.output_ids else output_index.with_name(
        f"{output_index.stem}_ids.json"
    )
    index.save(output_index, output_ids)
    print(f"Wrote {output_index} and {output_ids}")


if __name__ == "__main__":
    main()
