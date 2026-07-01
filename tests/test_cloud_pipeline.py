import argparse
import importlib.util
import sys
from pathlib import Path


def load_pipeline_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_cloud_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_cloud_pipeline_for_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_cloud_pipeline_for_test"] = module
    spec.loader.exec_module(module)
    return module


def args(**overrides):
    values = {
        "data_raw": "/kaggle/input/datasets/narendraaironi/bigearthnet-14k/BEN_14k",
        "image_root": "",
        "artifact_root": "artifacts",
        "manifest_dir": "data/manifests",
        "left_modality": "multispectral",
        "right_modality": "sar",
        "image_size": 128,
        "embedding_dim": 256,
        "backbone": "small_cnn",
        "projection_dropout": 0.0,
        "batch_size": 128,
        "epochs": 5,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "temperature": 0.07,
        "device": "cuda",
        "learnable_temperature": False,
        "semantic_loss_weight": 0.0,
        "hard_negative_loss_weight": 0.0,
        "hard_negative_margin": 0.2,
        "diagnostic_sample_count": 128,
        "seed": 42,
        "top_k": 10,
        "latency_queries": 100,
        "relevance_mode": "semantic",
        "export_zip": "",
        "allow_missing_labels": False,
        "skip_train": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_steps_runs_full_cloud_pipeline_with_label_checks():
    module = load_pipeline_module()

    steps = module.build_steps(args())
    names = [step.name for step in steps]
    commands = [" ".join(step.command) for step in steps]

    assert names == [
        "inspect dataset",
        "build manifest",
        "create splits",
        "check full manifest",
        "check train pairs",
        "check test pairs",
        "train baseline",
        "generate descriptors",
        "build FAISS index",
        "evaluate descriptors",
        "benchmark latency",
        "verify artifacts",
        "export artifacts",
    ]
    assert any("--require-labels" in command for command in commands)
    assert any(
        "--left-modality multispectral --right-modality sar" in command
        for command in commands
    )
    normalized_commands = [command.replace("\\", "/") for command in commands]
    assert any(
        "--validation-manifest data/manifests/validation.csv" in command
        for command in normalized_commands
    )
    assert any("--projection-dropout 0.0" in command for command in commands)
    assert any("--batch-size 128" in command for command in commands)
    assert any("--semantic-loss-weight 0.0" in command for command in commands)
    assert any("--hard-negative-loss-weight 0.0" in command for command in commands)
    assert any("--hard-negative-margin 0.2" in command for command in commands)
    assert any("--diagnostic-sample-count 128" in command for command in commands)
    assert any("--seed 42" in command for command in commands)


def test_build_steps_can_allow_missing_labels_for_debug_subsets():
    module = load_pipeline_module()

    steps = module.build_steps(args(allow_missing_labels=True))
    commands = [" ".join(step.command) for step in steps]

    assert not any("--require-labels" in command for command in commands)
