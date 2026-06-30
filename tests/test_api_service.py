import numpy as np

from earthbridge.api.service import RetrievalService
from earthbridge.retrieval.faiss_index import ExactFaissIndex


def test_retrieval_service_loads_index_and_retrieves(tmp_path):
    index_path = tmp_path / "demo.index"
    ids_path = tmp_path / "demo_ids.json"
    ExactFaissIndex.build(
        ["A", "B"],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    ).save(index_path, ids_path)

    service = RetrievalService.from_paths(index_path, ids_path)
    response = service.retrieve_descriptor([1.0, 0.0], top_k=1)

    assert service.index_loaded
    assert response["results"][0]["gallery_id"] == "A"


def test_retrieval_service_reports_not_loaded_for_missing_index(tmp_path):
    service = RetrievalService.from_paths(tmp_path / "missing.index", tmp_path / "missing.json")

    assert not service.index_loaded
    assert service.model_info()["index_type"] == "not_loaded"

