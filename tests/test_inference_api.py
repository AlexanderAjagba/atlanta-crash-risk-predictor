import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

from src.inference_api import create_server, load_grid_candidates


def test_grid_candidates_match_research_data():
    candidates = load_grid_candidates()
    assert candidates
    assert {"id", "latitude", "longitude"} <= candidates[0].keys()


def test_rank_endpoint_returns_structured_results(monkeypatch):
    def fake_rank(model, candidates, scenario_datetime, top_k):
        assert scenario_datetime == "2024-06-03 08:00:00"
        assert top_k == 1
        return [{"id": candidates[0]["id"], "latitude": 33.7, "longitude": -84.4, "riskProbability": 0.75}]

    monkeypatch.setattr("src.inference_api.rank_area_candidates", fake_rank)
    server = create_server(SimpleNamespace(), load_grid_candidates()[:2], "127.0.0.1", 0)
    server_thread = __import__("threading").Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/areas/rank"
        request = Request(
            url,
            data=json.dumps({"scenarioDatetime": "2024-06-03 08:00:00", "topK": 1}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            body = json.load(response)
        assert body["conditionalOnRecordedCrash"] is True
        assert body["areas"][0]["riskProbability"] == 0.75
    finally:
        server.shutdown()
        server.server_close()