from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class RetrievalMetrics:
    k: int
    precision: float
    recall: float
    f1: float


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be a positive integer")


def true_positives_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> int:
    _validate_k(k)
    return len(set(retrieved_ids[:k]) & set(relevant_ids))


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    _validate_k(k)
    return true_positives_at_k(retrieved_ids, relevant_ids, k) / k


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    _validate_k(k)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return true_positives_at_k(retrieved_ids, relevant, k) / len(relevant)


def f1_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    _validate_k(k)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    precision = precision_at_k(retrieved_ids, relevant, k)
    recall = recall_at_k(retrieved_ids, relevant, k)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def metrics_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> RetrievalMetrics:
    relevant = set(relevant_ids)
    precision = precision_at_k(retrieved_ids, relevant, k)
    recall = recall_at_k(retrieved_ids, relevant, k)
    f1 = f1_at_k(retrieved_ids, relevant, k)
    return RetrievalMetrics(k=k, precision=precision, recall=recall, f1=f1)


def mean_f1_at_k(results: Iterable[tuple[Sequence[str], set[str]]], k: int) -> float:
    scores = [f1_at_k(retrieved, relevant, k) for retrieved, relevant in results]
    return mean(scores) if scores else 0.0
