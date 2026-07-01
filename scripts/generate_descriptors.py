from __future__ import annotations

import argparse
from pathlib import Path

import _path  # noqa: F401

from earthbridge.retrieval.descriptors import save_descriptor_store
from earthbridge.retrieval.encoding import encode_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate descriptors from a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--output-dir", default="artifacts/descriptors")
    parser.add_argument("--name", default="gallery")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model-type", choices=["baseline", "dual_head"], default="baseline")
    parser.add_argument("--head", choices=["cross", "same"], default="cross")
    parser.add_argument("--backbone", default="small_cnn")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--projection-dropout", type=float, default=0.1)
    parser.add_argument("--shared-backbone", action="store_true")
    parser.add_argument("--modality", default="")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids, descriptors = encode_manifest(
        manifest_path=args.manifest,
        root_dir=args.root_dir,
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
        backbone=args.backbone,
        model_type=args.model_type,
        head=args.head,
        checkpoint=args.checkpoint or None,
        modality_filter=args.modality or None,
        device=args.device,
        projection_dropout=args.projection_dropout,
        shared_backbone=args.shared_backbone,
    )

    output_dir = Path(args.output_dir)
    descriptor_path = output_dir / f"{args.name}.npy"
    ids_path = output_dir / f"{args.name}_ids.json"
    save_descriptor_store(ids, descriptors, descriptor_path, ids_path)
    print(f"Wrote {len(ids)} descriptors to {descriptor_path}")


if __name__ == "__main__":
    main()
