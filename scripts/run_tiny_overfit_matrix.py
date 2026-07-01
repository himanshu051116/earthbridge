from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import _path  # noqa: F401

from earthbridge.data.manifest import load_manifest, write_manifest
from earthbridge.data.pairs import find_paired_rows
from earthbridge.training import TrainingConfig, train_paired_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled exact-pair overfit matrix on a tiny real-data subset."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--output-dir", default="artifacts/tiny_overfit")
    parser.add_argument("--left-modality", default="multispectral")
    parser.add_argument("--right-modality", default="sar")
    parser.add_argument("--pair-count", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--backbone", default="small_cnn")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.05, 0.07])
    parser.add_argument("--learnable-temperature-start", type=float, default=0.07)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--projection-dropout", type=float, default=0.0)
    parser.add_argument("--stop-recall-at-1", type=float, default=0.90)
    parser.add_argument("--stop-recall-at-10", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def create_subset_manifest(
    source_manifest: str | Path,
    output_manifest: str | Path,
    left_modality: str,
    right_modality: str,
    pair_count: int,
) -> int:
    rows = load_manifest(source_manifest)
    pairs = find_paired_rows(rows, left_modality, right_modality)
    selected_pairs = pairs[:pair_count]
    if len(selected_pairs) < pair_count:
        raise ValueError(f"Requested {pair_count} pairs, found only {len(selected_pairs)}")

    selected_rows = []
    for left_row, right_row in selected_pairs:
        selected_rows.extend([left_row, right_row])

    fieldnames = list(dict.fromkeys(field for row in selected_rows for field in row))
    write_manifest(output_manifest, selected_rows, fieldnames=fieldnames)
    return len(selected_pairs)


def gate_passed(
    result: dict[str, object],
    left_modality: str,
    right_modality: str,
    recall_at_1: float,
    recall_at_10: float,
) -> bool:
    validation = result["best_validation"]
    left_to_right = validation[f"{left_modality}_to_{right_modality}"]
    right_to_left = validation[f"{right_modality}_to_{left_modality}"]
    return (
        left_to_right["recall_at_1"] >= recall_at_1
        and right_to_left["recall_at_1"] >= recall_at_1
        and left_to_right["recall_at_10"] >= recall_at_10
        and right_to_left["recall_at_10"] >= recall_at_10
    )


def matrix_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for learning_rate in args.learning_rates:
        for temperature in args.temperatures:
            configs.append(
                {
                    "learning_rate": learning_rate,
                    "temperature": temperature,
                    "learnable_temperature": False,
                }
            )
        configs.append(
            {
                "learning_rate": learning_rate,
                "temperature": args.learnable_temperature_start,
                "learnable_temperature": True,
            }
        )
    return configs


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "overfit_128_pairs.csv"
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    selected_count = create_subset_manifest(
        source_manifest=args.manifest,
        output_manifest=manifest_path,
        left_modality=args.left_modality,
        right_modality=args.right_modality,
        pair_count=args.pair_count,
    )

    results: list[dict[str, object]] = []
    best_result: dict[str, object] | None = None
    for run_index, run_config in enumerate(matrix_configs(args), start=1):
        checkpoint_path = checkpoint_dir / f"run_{run_index:02d}.pt"
        config = TrainingConfig(
            manifest_path=str(manifest_path),
            root_dir=args.root_dir,
            left_modality=args.left_modality,
            right_modality=args.right_modality,
            validation_manifest_path=str(manifest_path),
            image_size=args.image_size,
            embedding_dim=args.embedding_dim,
            backbone=args.backbone,
            projection_dropout=args.projection_dropout,
            shared_backbone=False,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=float(run_config["learning_rate"]),
            weight_decay=args.weight_decay,
            temperature=float(run_config["temperature"]),
            learnable_temperature=bool(run_config["learnable_temperature"]),
            semantic_loss_weight=0.0,
            hard_negative_loss_weight=0.0,
            stop_recall_at_1=args.stop_recall_at_1,
            stop_recall_at_10=args.stop_recall_at_10,
            require_validation_pair_alignment=True,
            diagnostic_sample_count=args.pair_count,
            seed=args.seed,
            device=args.device,
            output_checkpoint=str(checkpoint_path),
        )
        result = train_paired_baseline(config)
        passed = gate_passed(
            result,
            args.left_modality,
            args.right_modality,
            recall_at_1=args.stop_recall_at_1,
            recall_at_10=args.stop_recall_at_10,
        )
        record = {
            "run_index": run_index,
            "config": asdict(config),
            "passed_gate": passed,
            "result": result,
        }
        results.append(record)

        current_score = float(result["best_mean_recall_at_1"])
        best_score = (
            float(best_result["result"]["best_mean_recall_at_1"])
            if best_result is not None
            else -1.0
        )
        if best_result is None or current_score > best_score:
            best_result = record

        report = {
            "selected_pair_count": selected_count,
            "gate": {
                "recall_at_1": args.stop_recall_at_1,
                "recall_at_10": args.stop_recall_at_10,
            },
            "best_result": best_result,
            "runs": results,
        }
        (output_dir / "tiny_overfit_matrix_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        if passed:
            (output_dir / "best_tiny_overfit_config.json").write_text(
                json.dumps(record, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(report, indent=2))
            return

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
