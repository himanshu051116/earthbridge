from earthbridge.data.splitting import SplitRatios, assert_no_group_leakage, grouped_split


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

