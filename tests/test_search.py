import numpy as np

from earthbridge.retrieval.search import NumpyCosineIndex


def test_numpy_cosine_index_returns_nearest_vectors():
    index = NumpyCosineIndex(
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
    assert response.search_time_ms >= 0


def test_numpy_cosine_index_can_exclude_self_match():
    index = NumpyCosineIndex(
        ["Q", "G"],
        np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
    )

    response = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1, exclude_ids={"Q"})

    assert [result.sample_id for result in response.results] == ["G"]

