from pathlib import Path


def test_dashboard_omits_latency_from_visible_status():
    html = Path("src/earthbridge/api/static/index.html").read_text(encoding="utf-8")

    assert "retrieval_time_ms" not in html
    assert "latency" not in html.lower()
    assert "Retrieved Matches" in html
    assert "Retrieve Matches" in html
    assert "/model-info" in html
