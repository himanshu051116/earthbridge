from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from earthbridge.retrieval.search import SearchResponse, SearchResult


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required for FAISS indexing") from exc
    return faiss


def normalize_descriptors(descriptors: np.ndarray) -> np.ndarray:
    array = np.asarray(descriptors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


@dataclass
class ExactFaissIndex:
    ids: list[str]
    index: object

    @classmethod
    def build(cls, ids: Sequence[str], descriptors: np.ndarray) -> ExactFaissIndex:
        faiss = _import_faiss()
        descriptors = normalize_descriptors(descriptors)
        if descriptors.ndim != 2:
            raise ValueError("descriptors must be a 2D array")
        if len(ids) != descriptors.shape[0]:
            raise ValueError("ids and descriptors must have the same number of rows")

        index = faiss.IndexFlatIP(descriptors.shape[1])
        index.add(descriptors)
        return cls(ids=list(ids), index=index)

    @classmethod
    def load(cls, index_path: str | Path, ids_path: str | Path) -> ExactFaissIndex:
        faiss = _import_faiss()
        index = faiss.read_index(str(index_path))
        with Path(ids_path).open("r", encoding="utf-8") as handle:
            ids = json.load(handle)
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise ValueError("ids file must contain a JSON list of strings")
        return cls(ids=ids, index=index)

    def save(self, index_path: str | Path, ids_path: str | Path) -> None:
        faiss = _import_faiss()
        index_output = Path(index_path)
        ids_output = Path(ids_path)
        index_output.parent.mkdir(parents=True, exist_ok=True)
        ids_output.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_output))
        with ids_output.open("w", encoding="utf-8") as handle:
            json.dump(self.ids, handle, indent=2)

    def search(
        self,
        query_descriptor: np.ndarray,
        top_k: int = 10,
        exclude_ids: set[str] | None = None,
        overfetch: int = 10,
    ) -> SearchResponse:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query = np.asarray(query_descriptor, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[0] != 1:
            raise ValueError("query_descriptor must represent one query")

        query = normalize_descriptors(query)
        requested = min(len(self.ids), top_k + max(overfetch, 0) + len(exclude_ids or set()))

        start = perf_counter()
        scores, positions = self.index.search(query, requested)
        excluded = exclude_ids or set()

        results: list[SearchResult] = []
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:
                continue
            sample_id = self.ids[int(position)]
            if sample_id in excluded:
                continue
            results.append(SearchResult(sample_id=sample_id, score=float(score)))
            if len(results) == top_k:
                break

        elapsed_ms = (perf_counter() - start) * 1000
        return SearchResponse(results=results, search_time_ms=elapsed_ms)

