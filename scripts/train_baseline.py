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
    parser.add_argument("--validation-manifest", default="")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--backbone", default="small_cnn")
    parser.add_argument("--projection-dropout", type=float, default=0.0)
    parser.add_argument("--shared-backbone", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learnable-temperature", action="store_true")
    parser.add_argument("--semantic-loss-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-loss-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.2)
    parser.add_argument("--stop-recall-at-1", type=float, default=0.0)
    parser.add_argument("--stop-recall-at-10", type=float, default=0.0)
    parser.add_argument("--require-validation-pair-alignment", action="store_true")
    parser.add_argument("--diagnostic-sample-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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
        validation_manifest_path=args.validation_manifest,
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
        backbone=args.backbone,
        projection_dropout=args.projection_dropout,
        shared_backbone=args.shared_backbone,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        learnable_temperature=args.learnable_temperature,
        semantic_loss_weight=args.semantic_loss_weight,
        hard_negative_loss_weight=args.hard_negative_loss_weight,
        hard_negative_margin=args.hard_negative_margin,
        stop_recall_at_1=args.stop_recall_at_1,
        stop_recall_at_10=args.stop_recall_at_10,
        require_validation_pair_alignment=args.require_validation_pair_alignment,
        diagnostic_sample_count=args.diagnostic_sample_count,
        seed=args.seed,
        device=args.device,
        output_checkpoint=args.output_checkpoint,
    )
    result = train_paired_baseline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
