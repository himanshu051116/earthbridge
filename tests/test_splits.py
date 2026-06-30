import importlib.util
import sys
from pathlib import Path

from earthbridge.data.splitting import SplitRatios, assert_no_group_leakage, grouped_split


def load_create_splits_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    module_path = scripts_dir / "create_splits.py"
    spec = importlib.util.spec_from_file_location("create_splits_for_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_rows():
    rows = []
    for index in range(12):
        rows.append(
            {
                "sample_id": f"OPT_{index}",
                "image_path": f"optical/{index}.tif",
                "modality": "optical_rgb",
                "pair_id": f"P{index}",
                "scene_id": f"S{index // 2}",
                "geographic_group": f"G{index // 3}",
                "labels": "water" if index % 2 else "urban",
            }
        )
        rows.append(
            {
                "sample_id": f"SAR_{index}",
                "image_path": f"sar/{index}.tif",
                "modality": "sar",
                "pair_id": f"P{index}",
                "scene_id": f"S{index // 2}",
                "geographic_group": f"G{index // 3}",
                "labels": "water" if index % 2 else "urban",
            }
        )
    return rows


def test_grouped_split_keeps_groups_together():
    splits = grouped_split(make_rows(), ratios=SplitRatios(0.5, 0.25, 0.25), seed=7)

    assert_no_group_leakage(splits)
    assert sum(len(rows) for rows in splits.values()) == len(make_rows())


def test_grouped_split_adds_split_column():
    splits = grouped_split(make_rows(), ratios=SplitRatios(0.5, 0.25, 0.25), seed=7)

    for split_name, rows in splits.items():
        for row in rows:
            assert row["split"] == split_name


def test_existing_manifest_splits_are_detected_and_preserved():
    module = load_create_splits_module()
    rows = [
        {"sample_id": "A", "image_path": "a.tif", "modality": "sar", "split": "train"},
        {"sample_id": "B", "image_path": "b.tif", "modality": "sar", "split": "test"},
    ]

    assert module.rows_have_existing_splits(rows)
    assert [row["sample_id"] for row in module.existing_splits(rows)["train"]] == ["A"]
    assert [row["sample_id"] for row in module.existing_splits(rows)["test"]] == ["B"]
