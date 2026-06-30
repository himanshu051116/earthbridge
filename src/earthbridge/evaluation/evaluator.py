from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean

from earthbridge.evaluation.metrics import f1_at_k
from earthbridge.evaluation.relevance import RelevanceMode, SampleRecord, relevant_ids

DIRECTIONS = [
    ("optical_rgb", "optical_rgb"),
    ("sar", "sar"),
    ("multispectral", "multispectral"),
    ("optical_rgb", "sar"),
    ("sar", "optical_rgb"),
    ("optical_rgb", "multispectral"),
    ("multispectral", "optical_rgb"),
    ("sar", "multispectral"),
    ("multispectral", "sar"),
]


@dataclass(frozen=True)
class DirectionScore:
    query_modality: str
    target_modality: str
    query_count: int
    f1_by_k: dict[int, float]


def supported_directions(records: Sequence[SampleRecord]) -> list[tuple[str, str]]:
    modalities = {record.modality for record in records}
    canonical = [direction for direction in DIRECTIONS if set(direction) <= modalities]

    if canonical:
        return canonical

    return sorted((left, right) for left in modalities for right in modalities)


def evaluate_rankings(
    records: Sequence[SampleRecord],
    rankings: Mapping[tuple[str, str, str], Sequence[str]],
    mode: RelevanceMode,
    k_values: Sequence[int] = (5, 10),
    predefined: Mapping[str, set[str]] | None = None,
    semantic_threshold: float = 0.5,
) -> list[DirectionScore]:
    by_modality: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        by_modality[record.modality].append(record)

    scores: list[DirectionScore] = []
    for query_modality, target_modality in supported_directions(records):
        query_records = by_modality[query_modality]
        gallery_records = by_modality[target_modality]
        per_k_scores: dict[int, list[float]] = {k: [] for k in k_values}

        for query in query_records:
            key = (query.sample_id, query_modality, target_modality)
            retrieved = rankings.get(key, [])
            relevant = relevant_ids(
                query,
                gallery_records,
                mode,
                predefined=predefined,
                semantic_threshold=semantic_threshold,
                exclude_self=query_modality == target_modality,
            )

            for k in k_values:
                per_k_scores[k].append(f1_at_k(retrieved, relevant, k))

        scores.append(
            DirectionScore(
                query_modality=query_modality,
                target_modality=target_modality,
                query_count=len(query_records),
                f1_by_k={
                    k: mean(values) if values else 0.0 for k, values in per_k_scores.items()
                },
            )
        )

    return scores

