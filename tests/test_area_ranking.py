from types import SimpleNamespace

import numpy as np

from src.models.area_ranking import rank_area_candidates


def test_rank_area_candidates_uses_one_scenario_and_sorts(monkeypatch):
    calls = []

    def fake_predict(model, records):
        calls.append(records)
        return [
            {"latitude": row["latitude"], "longitude": row["longitude"], "riskProbability": probability}
            for row, probability in zip(records, [0.2, 0.8, 0.8])
        ]

    monkeypatch.setattr("src.models.area_ranking.predict_records", fake_predict)
    candidates = [
        {"id": "a", "latitude": 33.70, "longitude": -84.40},
        {"id": "b", "latitude": 33.71, "longitude": -84.41},
        {"id": "c", "latitude": 33.72, "longitude": -84.42},
    ]

    result = rank_area_candidates(SimpleNamespace(), candidates, "2024-06-03 08:00:00", top_k=2)

    assert [item["id"] for item in result] == ["b", "c"]
    assert len(calls) == 1
    assert {row["datetime"] for row in calls[0]} == {"2024-06-03 08:00:00"}
    assert {row["geolocated_city"] for row in calls[0]} == {"Atlanta"}
    assert {row["geolocated_county"] for row in calls[0]} == {"Fulton"}


def test_rank_area_candidates_validates_inputs():
    try:
        rank_area_candidates(SimpleNamespace(), [], "2024-06-03 08:00:00")
    except ValueError as error:
        assert "candidate" in str(error)
    else:
        raise AssertionError("empty candidates should fail")

    try:
        rank_area_candidates(SimpleNamespace(), [{"id": "a", "latitude": 1, "longitude": 1}], "2024-06-03 08:00:00", 0)
    except ValueError as error:
        assert "top_k" in str(error)
    else:
        raise AssertionError("invalid top_k should fail")