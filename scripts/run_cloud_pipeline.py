from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full EarthBridge cloud training and export pipeline."
    )
    parser.add_argument(
        "--data-raw",
        default="/kaggle/input/datasets/narendraaironi/bigearthnet-14k/BEN_14k",
    )
    parser.add_argument("--image-root", default="")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--left-modality", default="multispectral")
    parser.add_argument("--right-modality", default="sar")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--backbone", default="small_cnn")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--projection-dropout", type=float, default=0.0)
    parser.add_argument("--semantic-loss-weight", type=float, default=0.1)
    parser.add_argument("--hard-negative-loss-weight", type=float, default=0.2)
    parser.add_argument("--hard-negative-margin", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--latency-queries", type=int, default=100)
    parser.add_argument("--relevance-mode", default="semantic")
    parser.add_argument("--export-zip", default="")
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Skip label-required checks for quick unlabeled smoke subsets.",
    )
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def script_command(script_name: str, *args: str) -> list[str]:
    return [sys.executable, str(Path("scripts") / script_name), *args]


def label_check_args(args: argparse.Namespace) -> list[str]:
    return [] if args.allow_missing_labels else ["--require-labels"]


def build_steps(args: argparse.Namespace) -> list[PipelineStep]:
    data_raw = Path(args.data_raw)
    image_root = Path(args.image_root) if args.image_root else data_raw.parent
    artifact_root = Path(args.artifact_root)
    manifest_dir = Path(args.manifest_dir)

    reports_dir = artifact_root / "reports"
    descriptors_dir = artifact_root / "descriptors"
    indexes_dir = artifact_root / "indexes"
    checkpoints_dir = artifact_root / "checkpoints"

    samples_manifest = manifest_dir / "samples.csv"
    train_manifest = manifest_dir / "train.csv"
    test_manifest = manifest_dir / "test.csv"
    checkpoint = checkpoints_dir / "baseline_pair.pt"
    descriptors = descriptors_dir / "gallery.npy"
    descriptor_ids = descriptors_dir / "gallery_ids.json"
    index = indexes_dir / "gallery.index"
    index_ids = indexes_dir / "gallery_ids.json"

    steps = [
        PipelineStep(
            "inspect dataset",
            script_command(
                "inspect_data.py",
                "--input",
                str(data_raw),
                "--output-dir",
                str(reports_dir),
            ),
        ),
        PipelineStep(
            "build manifest",
            script_command(
                "build_manifest.py",
                "--input",
                str(data_raw),
                "--output",
                str(samples_manifest),
            ),
        ),
        PipelineStep(
            "create splits",
            script_command(
                "create_splits.py",
                "--manifest",
                str(samples_manifest),
                "--output-dir",
                str(manifest_dir),
            ),
        ),
        PipelineStep(
            "check full manifest",
            script_command(
                "check_manifest.py",
                "--manifest",
                str(samples_manifest),
                "--root-dir",
                str(image_root),
                *label_check_args(args),
                "--output",
                str(reports_dir / "manifest_samples_check.json"),
            ),
        ),
        PipelineStep(
            "check train pairs",
            script_command(
                "check_manifest.py",
                "--manifest",
                str(train_manifest),
                "--root-dir",
                str(image_root),
                "--left-modality",
                args.left_modality,
                "--right-modality",
                args.right_modality,
                "--min-pairs",
                "1",
                *label_check_args(args),
                "--output",
                str(reports_dir / "manifest_train_check.json"),
            ),
        ),
        PipelineStep(
            "check test pairs",
            script_command(
                "check_manifest.py",
                "--manifest",
                str(test_manifest),
                "--root-dir",
                str(image_root),
                "--left-modality",
                args.left_modality,
                "--right-modality",
                args.right_modality,
                "--min-pairs",
                "1",
                *label_check_args(args),
                "--output",
                str(reports_dir / "manifest_test_check.json"),
            ),
        ),
    ]

    if not args.skip_train:
        steps.append(
            PipelineStep(
                "train baseline",
                script_command(
                    "train_baseline.py",
                    "--manifest",
                    str(train_manifest),
                    "--root-dir",
                    str(image_root),
                    "--left-modality",
                    args.left_modality,
                    "--right-modality",
                    args.right_modality,
                    "--validation-manifest",
                    str(manifest_dir / "validation.csv"),
                    "--image-size",
                    str(args.image_size),
                    "--embedding-dim",
                    str(args.embedding_dim),
                    "--backbone",
                    args.backbone,
                    "--projection-dropout",
                    str(args.projection_dropout),
                    "--batch-size",
                    str(args.batch_size),
                    "--epochs",
                    str(args.epochs),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--weight-decay",
                    str(args.weight_decay),
                    "--temperature",
                    str(args.temperature),
                    "--semantic-loss-weight",
                    str(args.semantic_loss_weight),
                    "--hard-negative-loss-weight",
                    str(args.hard_negative_loss_weight),
                    "--hard-negative-margin",
                    str(args.hard_negative_margin),
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                    "--output-checkpoint",
                    str(checkpoint),
                ),
            )
        )

    steps.extend(
        [
            PipelineStep(
                "generate descriptors",
                script_command(
                    "generate_descriptors.py",
                    "--manifest",
                    str(test_manifest),
                    "--root-dir",
                    str(image_root),
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(descriptors_dir),
                    "--name",
                    "gallery",
                    "--image-size",
                    str(args.image_size),
                    "--embedding-dim",
                    str(args.embedding_dim),
                    "--backbone",
                    args.backbone,
                    "--projection-dropout",
                    str(args.projection_dropout),
                    "--device",
                    args.device,
                ),
            ),
            PipelineStep(
                "build FAISS index",
                script_command(
                    "build_indexes.py",
                    "--descriptors",
                    str(descriptors),
                    "--ids",
                    str(descriptor_ids),
                    "--output-index",
                    str(index),
                    "--output-ids",
                    str(index_ids),
                ),
            ),
            PipelineStep(
                "evaluate descriptors",
                script_command(
                    "evaluate_descriptors.py",
                    "--manifest",
                    str(test_manifest),
                    "--descriptors",
                    str(descriptors),
                    "--ids",
                    str(descriptor_ids),
                    "--relevance-mode",
                    args.relevance_mode,
                    "--output-dir",
                    str(reports_dir),
                ),
            ),
            PipelineStep(
                "benchmark latency",
                script_command(
                    "benchmark_latency.py",
                    "--descriptors",
                    str(descriptors),
                    "--ids",
                    str(descriptor_ids),
                    "--queries",
                    str(args.latency_queries),
                    "--top-k",
                    str(args.top_k),
                    "--output",
                    str(reports_dir / "latency_summary.json"),
                ),
            ),
            PipelineStep(
                "verify artifacts",
                script_command("verify_artifacts.py", "--artifact-root", str(artifact_root)),
            ),
            PipelineStep(
                "export artifacts",
                script_command(
                    "export_artifacts.py",
                    "--artifact-root",
                    str(artifact_root),
                    "--output",
                    args.export_zip or str(artifact_root / "earthbridge_export.zip"),
                ),
            ),
        ]
    )
    return steps


def copy_test_manifest(args: argparse.Namespace) -> None:
    manifest_dir = Path(args.manifest_dir)
    artifact_root = Path(args.artifact_root)
    output = artifact_root / "manifests" / "test.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_dir / "test.csv", output)


def run_step(step: PipelineStep) -> None:
    print(f"\n=== {step.name} ===")
    print(" ".join(step.command))
    subprocess.run(step.command, check=True)


def main() -> None:
    args = parse_args()
    steps = build_steps(args)
    print(json.dumps({"steps": [asdict(step) for step in steps]}, indent=2))

    for step in steps:
        if step.name == "verify artifacts":
            copy_test_manifest(args)
        run_step(step)


if __name__ == "__main__":
    main()
