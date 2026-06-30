from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from earthbridge.retrieval.faiss_index import ExactFaissIndex


@dataclass
class RetrievalService:
    index: ExactFaissIndex | None
    name: str = "earthbridge-baseline"

    @classmethod
    def from_paths(cls, index_path: str | Path, ids_path: str | Path) -> RetrievalService:
        index_file = Path(index_path)
        ids_file = Path(ids_path)
        if not index_file.exists() or not ids_file.exists():
            return cls(index=None)
        return cls(index=ExactFaissIndex.load(index_file, ids_file))

    @classmethod
    def from_environment(cls) -> RetrievalService:
        index_path = os.getenv("EARTHBRIDGE_INDEX_PATH", "artifacts/demo/demo.index")
        ids_path = os.getenv("EARTHBRIDGE_IDS_PATH", "artifacts/demo/demo_ids.json")
        return cls.from_paths(index_path, ids_path)

    @property
    def index_loaded(self) -> bool:
        return self.index is not None

    def model_info(self) -> dict[str, object]:
        if self.index is None:
            return {
                "name": self.name,
                "embedding_dim": None,
                "index_size": 0,
                "index_type": "not_loaded",
            }

        return {
            "name": self.name,
            "embedding_dim": int(self.index.index.d),
            "index_size": len(self.index.ids),
            "index_type": "faiss.IndexFlatIP",
        }

    def retrieve_descriptor(
        self,
        descriptor: list[float],
        top_k: int = 10,
        exclude_ids: list[str] | None = None,
    ) -> dict[str, object]:
        if self.index is None:
            raise RuntimeError(
                "No index loaded. Create one with scripts/create_demo_index.py "
                "or scripts/build_indexes.py"
            )

        response = self.index.search(
            np.asarray(descriptor, dtype=np.float32),
            top_k=top_k,
            exclude_ids=set(exclude_ids or []),
        )
        return {
            "retrieval_type": "descriptor",
            "retrieval_time_ms": response.search_time_ms,
            "results": [
                {
                    "rank": rank,
                    "gallery_id": result.sample_id,
                    "similarity": result.score,
                }
                for rank, result in enumerate(response.results, start=1)
            ],
        }
