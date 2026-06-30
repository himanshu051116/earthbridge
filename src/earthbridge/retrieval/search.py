from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np


def l2_normalize(vectors: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, epsilon)


@dataclass(frozen=True)
class SearchResult:
    sample_id: str
    score: float


@dataclass(frozen=True)
class SearchResponse:
    results: list[SearchResult]
    search_time_ms: float


class NumpyCosineIndex:
    """Small exact cosine-search fallback for tests and demos before FAISS is wired in."""

    def __init__(self, ids: Sequence[str], descriptors: np.ndarray) -> None:
        if len(ids) != len(descriptors):
            raise ValueError("ids and descriptors must have the same length")
        if descriptors.ndim != 2:
            raise ValueError("descriptors must be a 2D array")

        self.ids = list(ids)
        self.descriptors = l2_normalize(descriptors)

    def search(
        self,
        query_descriptor: np.ndarray,
        top_k: int = 10,
        exclude_ids: set[str] | None = None,
    ) -> SearchResponse:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query = np.asarray(query_descriptor, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[0] != 1:
            raise ValueError("query_descriptor must represent one query")
        if query.shape[1] != self.descriptors.shape[1]:
            raise ValueError("query descriptor dimension does not match index dimension")

        start = perf_counter()
        normalized_query = l2_normalize(query)
        scores = self.descriptors @ normalized_query[0]
        order = np.argsort(-scores)

        excluded = exclude_ids or set()
        results: list[SearchResult] = []
        for position in order:
            sample_id = self.ids[int(position)]
            if sample_id in excluded:
                continue
            results.append(SearchResult(sample_id=sample_id, score=float(scores[int(position)])))
            if len(results) == top_k:
                break

        elapsed_ms = (perf_counter() - start) * 1000
        return SearchResponse(results=results, search_time_ms=elapsed_ms)

