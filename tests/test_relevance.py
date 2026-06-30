from earthbridge.evaluation.relevance import (
    RelevanceMode,
    SampleRecord,
    is_relevant,
    jaccard_similarity,
    relevant_ids,
)


def test_geographic_relevance_uses_pair_scene_or_group():
    query = SampleRecord(sample_id="Q", modality="optical_rgb", pair_id="P1")
    paired = SampleRecord(sample_id="G1", modality="sar", pair_id="P1")
    different = SampleRecord(sample_id="G2", modality="sar", pair_id="P2")

    assert is_relevant(query, paired, RelevanceMode.GEOGRAPHIC)
    assert not is_relevant(query, different, RelevanceMode.GEOGRAPHIC)


def test_semantic_relevance_uses_jaccard_threshold():
    query = SampleRecord(
        sample_id="Q",
        modality="optical_rgb",
        labels=frozenset({"water", "urban"}),
    )
    related = SampleRecord(sample_id="G1", modality="sar", labels=frozenset({"water", "urban"}))
    weak = SampleRecord(sample_id="G2", modality="sar", labels=frozenset({"water", "forest"}))

    assert jaccard_similarity(query.labels, weak.labels) == 1 / 3
    assert is_relevant(query, related, RelevanceMode.SEMANTIC, semantic_threshold=0.5)
    assert not is_relevant(query, weak, RelevanceMode.SEMANTIC, semantic_threshold=0.5)


def test_predefined_relevance_overrides_inference():
    query = SampleRecord(sample_id="Q", modality="optical_rgb", pair_id="P1")
    candidate = SampleRecord(sample_id="G1", modality="sar", pair_id="P1")

    assert not is_relevant(
        query,
        candidate,
        RelevanceMode.PREDEFINED,
        predefined={"Q": {"SOME_OTHER_ID"}},
    )


def test_relevant_ids_excludes_self_for_same_modal_search():
    query = SampleRecord(sample_id="Q", modality="optical_rgb", labels=frozenset({"water"}))
    gallery = [
        query,
        SampleRecord(sample_id="G1", modality="optical_rgb", labels=frozenset({"water"})),
    ]

    assert relevant_ids(query, gallery, RelevanceMode.SEMANTIC) == {"G1"}
