from __future__ import annotations

import argparse
import json

import _path  # noqa: F401

from earthbridge.training import TrainingConfig, train_paired_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the paired contrastive baseline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--left-modality", default="optical_rgb")
    parser.add_argument("--right-modality", default="sar")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--backbone", default="small_cnn")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-checkpoint",
        default="artifacts/checkpoints/baseline_pair.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        manifest_path=args.manifest,
        root_dir=args.root_dir,
        left_modality=args.left_modality,
        right_modality=args.right_modality,
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
        backbone=args.backbone,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        device=args.device,
        output_checkpoint=args.output_checkpoint,
    )
    result = train_paired_baseline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

