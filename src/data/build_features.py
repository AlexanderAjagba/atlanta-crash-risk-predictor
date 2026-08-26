from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


NUMERIC_TIME_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_rush_hour",
    "is_night",
    "latitude",
    "longitude",
]

CATEGORICAL_FEATURES = [
    "geolocated_city",
    "geolocated_county",
    "light_conditions_crash_level",
    "surface_condition_crash_level",
    "weather_conditions_crash_level",
    "location_at_impact_crash_level",
    "manner_of_collision_crash_level",
]

FEATURE_COLUMNS = NUMERIC_TIME_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "severe_crash"
SPLIT_METADATA_COLUMNS = ["crash_datetime", "year"]

LEAKAGE_COLUMNS = [
    "kabco_severity",
    "number_serious_injuries",
    "number_visible_injuries",
    "number_of_fatalities_per_crash",
]

POST_INVESTIGATION_COLUMNS = [
    "first_harmful_event_unit_order",
    "most_harmful_event_crash_level",
    "safety_equipment_crash_level",
    "operator_pedestrian_contrib_factor_excl_none_other_no_contrib_factors",
]

FEATURE_LIST_PATH = Path("data/processed/feature_columns.json")


def build_model_table(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN] + SPLIT_METADATA_COLUMNS
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    model_table = df[required_columns].copy()

    for column in CATEGORICAL_FEATURES:
        model_table[column] = model_table[column].fillna("Unknown")

    return model_table


def save_feature_list(output_path: Path | str = FEATURE_LIST_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(FEATURE_COLUMNS, handle, indent=2)
        handle.write("\n")
    return output_path
