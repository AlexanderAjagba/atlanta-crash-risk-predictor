"""Rank Atlanta grid cells by predicted severe-crash probability."""
from __future__ import annotations

from typing import Any, Sequence

from src.models.hf_inference import predict_records


def rank_area_candidates(
    model: Any,
    candidates: Sequence[dict[str, Any]],
    scenario_datetime: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Score every candidate under one shared scenario and return highest first."""
    if not candidates:
        raise ValueError("At least one area candidate is required.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1.")

    records = [
        {
            "datetime": scenario_datetime,
            "latitude": candidate["latitude"],
            "longitude": candidate["longitude"],
            "geolocated_city": candidate.get("geolocated_city", "Atlanta"),
            "geolocated_county": candidate.get("geolocated_county", "Fulton"),
        }
        for candidate in candidates
    ]
    predictions = predict_records(model, records)
    ranked = [
        {
            "id": candidate["id"],
            "latitude": prediction["latitude"],
            "longitude": prediction["longitude"],
            "riskProbability": prediction["riskProbability"],
        }
        for candidate, prediction in zip(candidates, predictions)
    ]
    ranked.sort(key=lambda item: (-item["riskProbability"], item["id"]))
    return ranked[:top_k] if top_k is not None else ranked