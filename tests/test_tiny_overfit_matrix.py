import argparse
import csv
import importlib.util
import sys
from pathlib import Path

from earthbridge.data.manifest import load_manifest


def load_matrix_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    module_path = scripts_dir / "run_tiny_overfit_matrix.py"
    sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("run_tiny_overfit_matrix_for_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_tiny_overfit_matrix_for_test"] = module
    spec.loader.exec_module(module)
    return module


def test_create_subset_manifest_preserves_paired_rows(tmp_path):
    module = load_matrix_module()
    source = tmp_path / "train.csv"
    output = tmp_path / "overfit.csv"

    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_path", "modality", "pair_id", "channels"],
        )
        writer.writeheader()
        for index in range(3):
            writer.writerow(
                {
                    "sample_id": f"S2_{index}",
                    "image_path": f"s2_{index}.tif",
                    "modality": "multispectral",
                    "pair_id": f"P{index}",
                    "channels": "10",
                }
            )
            writer.writerow(
                {
                    "sample_id": f"S1_{index}",
                    "image_path": f"s1_{index}.tif",
                    "modality": "sar",
                    "pair_id": f"P{index}",
                    "channels": "2",
                }
            )

    selected_count = module.create_subset_manifest(
        source_manifest=source,
        output_manifest=output,
        left_modality="multispectral",
        right_modality="sar",
        pair_count=2,
    )
    rows = load_manifest(output)

    assert selected_count == 2
    assert [row["pair_id"] for row in rows] == ["P0", "P0", "P1", "P1"]


def test_matrix_configs_include_fixed_and_learnable_temperature():
    module = load_matrix_module()
    args = argparse.Namespace(
        learning_rates=[1e-4, 3e-4, 1e-3],
        temperatures=[0.05, 0.07],
        learnable_temperature_start=0.07,
    )

    configs = module.matrix_configs(args)

    assert len(configs) == 9
    assert {"learning_rate": 1e-4, "temperature": 0.05, "learnable_temperature": False} in configs
    assert {"learning_rate": 1e-3, "temperature": 0.07, "learnable_temperature": True} in configs
