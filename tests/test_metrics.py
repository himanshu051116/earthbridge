from earthbridge.evaluation.metrics import f1_at_k, metrics_at_k, precision_at_k, recall_at_k


def test_f1_at_5_matches_hand_calculation():
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "X"}

    expected = 2 * (2 / 5) * (2 / 3) / ((2 / 5) + (2 / 3))

    assert abs(f1_at_k(retrieved, relevant, 5) - expected) < 1e-8


def test_empty_relevance_returns_zero():
    assert f1_at_k(["A", "B"], set(), 5) == 0.0
    assert recall_at_k(["A", "B"], set(), 5) == 0.0


def test_precision_uses_requested_k_denominator():
    assert precision_at_k(["A", "B"], {"A", "B"}, 5) == 2 / 5


def test_metrics_bundle():
    metrics = metrics_at_k(["A", "B", "C"], {"A", "D"}, 3)

    assert metrics.k == 3
    assert metrics.precision == 1 / 3
    assert metrics.recall == 1 / 2
    assert metrics.f1 == f1_at_k(["A", "B", "C"], {"A", "D"}, 3)

