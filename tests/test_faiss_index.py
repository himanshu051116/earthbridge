import numpy as np

from earthbridge.retrieval.faiss_index import ExactFaissIndex


def test_exact_faiss_index_returns_nearest_vectors():
    index = ExactFaissIndex.build(
        ["A", "B", "C"],
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        ),
    )

    response = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)

    assert [result.sample_id for result in response.results] == ["A", "C"]


def test_exact_faiss_index_save_and_load(tmp_path):
    index_path = tmp_path / "demo.index"
    ids_path = tmp_path / "demo_ids.json"

    index = ExactFaissIndex.build(
        ["A", "B"],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    index.save(index_path, ids_path)

    loaded = ExactFaissIndex.load(index_path, ids_path)
    response = loaded.search(np.array([0.0, 1.0], dtype=np.float32), top_k=1)

    assert response.results[0].sample_id == "B"

