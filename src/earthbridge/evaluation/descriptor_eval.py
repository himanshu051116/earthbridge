from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

import numpy as np

from earthbridge.evaluation.evaluator import DirectionScore, supported_directions
from earthbridge.evaluation.metrics import f1_at_k
from earthbridge.evaluation.relevance import RelevanceMode, SampleRecord, relevant_ids
from earthbridge.retrieval.faiss_index import ExactFaissIndex


@dataclass(frozen=True)
class DescriptorEvaluationResult:
    direction_scores: list[DirectionScore]
    mean_search_latency_ms: float
    query_count: int


def evaluate_descriptors(
    records: list[SampleRecord],
    ids: list[str],
    descriptors: np.ndarray,
    relevance_mode: RelevanceMode,
    k_values: tuple[int, ...] = (5, 10),
    semantic_threshold: float = 0.5,
) -> DescriptorEvaluationResult:
    if len(ids) != len(descriptors):
        raise ValueError("ids and descriptors must have the same number of rows")

    record_by_id = {record.sample_id: record for record in records}
    descriptor_by_id = {
        sample_id: descriptors[index]
        for index, sample_id in enumerate(ids)
        if sample_id in record_by_id
    }
    by_modality: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        if record.sample_id in descriptor_by_id:
            by_modality[record.modality].append(record)

    direction_scores: list[DirectionScore] = []
    latencies: list[float] = []
    total_queries = 0

    for query_modality, target_modality in supported_directions(records):
        query_records = by_modality.get(query_modality, [])
        gallery_records = by_modality.get(target_modality, [])
        if not query_records or not gallery_records:
            continue

        gallery_ids = [record.sample_id for record in gallery_records]
        gallery_descriptors = np.stack([descriptor_by_id[sample_id] for sample_id in gallery_ids])
        index = ExactFaissIndex.build(gallery_ids, gallery_descriptors)
        per_k_scores: dict[int, list[float]] = {k: [] for k in k_values}

        for query in query_records:
            query_descriptor = descriptor_by_id[query.sample_id]
            response = index.search(
                query_descriptor,
                top_k=max(k_values),
                exclude_ids={query.sample_id} if query_modality == target_modality else set(),
            )
            retrieved = [result.sample_id for result in response.results]
            relevant = relevant_ids(
                query,
                gallery_records,
                relevance_mode,
                semantic_threshold=semantic_threshold,
                exclude_self=query_modality == target_modality,
            )

            for k in k_values:
                per_k_scores[k].append(f1_at_k(retrieved, relevant, k))
            latencies.append(response.search_time_ms)
            total_queries += 1

        direction_scores.append(
            DirectionScore(
                query_modality=query_modality,
                target_modality=target_modality,
                query_count=len(query_records),
                f1_by_k={
                    k: mean(values) if values else 0.0 for k, values in per_k_scores.items()
                },
            )
        )

    return DescriptorEvaluationResult(
        direction_scores=direction_scores,
        mean_search_latency_ms=mean(latencies) if latencies else 0.0,
        query_count=total_queries,
    )
