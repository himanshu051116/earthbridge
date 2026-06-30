import numpy as np

from earthbridge.evaluation.descriptor_eval import evaluate_descriptors
from earthbridge.evaluation.relevance import RelevanceMode, SampleRecord


def test_evaluate_descriptors_scores_cross_modal_pairs():
    records = [
        SampleRecord(
            sample_id="O1",
            modality="optical_rgb",
            pair_id="P1",
            labels=frozenset({"water"}),
        ),
        SampleRecord(sample_id="S1", modality="sar", pair_id="P1", labels=frozenset({"water"})),
        SampleRecord(
            sample_id="O2",
            modality="optical_rgb",
            pair_id="P2",
            labels=frozenset({"urban"}),
        ),
        SampleRecord(sample_id="S2", modality="sar", pair_id="P2", labels=frozenset({"urban"})),
    ]
    ids = ["O1", "S1", "O2", "S2"]
    descriptors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )

    result = evaluate_descriptors(
        records,
        ids,
        descriptors,
        relevance_mode=RelevanceMode.SEMANTIC,
        k_values=(1,),
    )

    optical_to_sar = next(
        score
        for score in result.direction_scores
        if score.query_modality == "optical_rgb" and score.target_modality == "sar"
    )

    assert optical_to_sar.f1_by_k[1] == 1.0
    assert result.query_count > 0

