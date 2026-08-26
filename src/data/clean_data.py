from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


RAW_DATA_PATH = Path("data/raw/Collisions Dataset.csv")
CLEANED_DATA_PATH = Path("data/processed/collisions_cleaned.parquet")

SEVERE_KABCO_VALUES = {
    "(A) Suspected Serious Injury",
    "(K) Fatal Injury",
}

VALID_KABCO_VALUES = {
    "(O) No Injury",
    "(C) Possible Injury / Complaint",
    "(B) Suspected Minor/Visible Injury",
    "(A) Suspected Serious Injury",
    "(K) Fatal Injury",
}

COUNT_COLUMNS = [
    "number_serious_injuries",
    "number_visible_injuries",
    "number_of_fatalities_per_crash",
]

DESCRIPTIVE_CATEGORICAL_COLUMNS = [
    "agency_name_crash_level",
    "intersection_name_from_crash_report",
    "light_conditions_crash_level",
    "location_at_impact_crash_level",
    "manner_of_collision_crash_level",
    "roadway_from_crash_report",
    "surface_condition_crash_level",
    "safety_equipment_crash_level",
    "weather_conditions_crash_level",
    "geolocated_city",
    "geolocated_county",
    "first_harmful_event_unit_order",
    "operator_pedestrian_contrib_factor_excl_none_other_no_contrib_factors",
    "most_harmful_event_crash_level",
]

LIST_LIKE_COLUMNS = [
    "safety_equipment_crash_level",
    "first_harmful_event_unit_order",
    "operator_pedestrian_contrib_factor_excl_none_other_no_contrib_factors",
    "most_harmful_event_crash_level",
]


@dataclass
class CleaningReport:
    original_row_count: int
    cleaned_row_count: int = 0
    duplicates_removed: int = 0
    rows_removed: dict[str, int] = field(default_factory=dict)
    missing_before_pct: dict[str, float] = field(default_factory=dict)
    missing_after_pct: dict[str, float] = field(default_factory=dict)
    unknown_kabco_rows: int = 0
    list_parse_failures: dict[str, int] = field(default_factory=dict)


def standardize_column_name(column_name: str) -> str:
    name = column_name.strip().lower().replace("#", "number")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def missing_percentages(df: pd.DataFrame) -> dict[str, float]:
    return (df.isna().mean().mul(100).round(2)).to_dict()


def _parse_list_like(value: Any) -> list[Any]:
    if pd.isna(value):
        return []

    if not isinstance(value, str):
        return [value]

    stripped = value.strip()
    if not stripped or stripped.lower() in {"unknown", "(none)", "none"}:
        return [] if not stripped else [stripped]

    if stripped.startswith("[") and stripped.endswith("]"):
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, list):
            return parsed
        return [parsed]

    return [stripped]


def add_list_like_counts(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    df = df.copy()

    for column in LIST_LIKE_COLUMNS:
        if column not in df.columns:
            continue

        parsed_values: list[list[Any]] = []
        failures = 0
        for value in df[column]:
            try:
                parsed_values.append(_parse_list_like(value))
            except (SyntaxError, ValueError):
                failures += 1
                parsed_values.append([])

        df[f"{column}_entry_count"] = [len(items) for items in parsed_values]
        report.list_parse_failures[column] = failures

    return df


def clean_data(raw_path: Path | str = RAW_DATA_PATH) -> tuple[pd.DataFrame, CleaningReport]:
    raw_path = Path(raw_path)
    df = pd.read_csv(raw_path)
    report = CleaningReport(original_row_count=len(df))

    df.columns = [standardize_column_name(column) for column in df.columns]
    report.missing_before_pct = missing_percentages(df)

    duplicate_mask = df.duplicated()
    report.duplicates_removed = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].copy()

    df = df.replace({"": pd.NA, "(None)": pd.NA})

    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df["crash_datetime"] = pd.to_datetime(
        df["date"].astype("string").str.strip()
        + " "
        + df["time"].astype("string").str.strip(),
        errors="coerce",
    )

    date_time_invalid = df["date_parsed"].isna() | df["crash_datetime"].isna()
    report.rows_removed["invalid_or_unparsed_date_time"] = int(date_time_invalid.sum())
    df = df.loc[~date_time_invalid].copy()

    for column in COUNT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_counts = pd.Series(False, index=df.index)
    for column in COUNT_COLUMNS:
        values = df[column]
        invalid_counts |= values.isna() | (values < 0) | (values % 1 != 0)

    report.rows_removed["invalid_injury_or_fatality_counts"] = int(invalid_counts.sum())
    df = df.loc[~invalid_counts].copy()
    df[COUNT_COLUMNS] = df[COUNT_COLUMNS].astype("int64")

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    invalid_location = (
        df["latitude"].isna()
        | df["longitude"].isna()
        | ~df["latitude"].between(-90, 90)
        | ~df["longitude"].between(-180, 180)
    )
    report.rows_removed["invalid_latitude_or_longitude"] = int(invalid_location.sum())
    df = df.loc[~invalid_location].copy()

    unknown_kabco = df["kabco_severity"].eq("Unknown") | df["kabco_severity"].isna()
    invalid_kabco = ~df["kabco_severity"].isin(VALID_KABCO_VALUES) & ~unknown_kabco
    report.unknown_kabco_rows = int(unknown_kabco.sum())
    report.rows_removed["unknown_kabco_severity_not_labelable"] = report.unknown_kabco_rows
    report.rows_removed["invalid_kabco_severity"] = int(invalid_kabco.sum())
    df = df.loc[~unknown_kabco & ~invalid_kabco].copy()

    df["year"] = df["crash_datetime"].dt.year.astype("int64")
    df["month"] = df["crash_datetime"].dt.month.astype("int64")
    df["day_of_week"] = df["crash_datetime"].dt.dayofweek.astype("int64")
    df["hour"] = df["crash_datetime"].dt.hour.astype("int64")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int64")
    df["is_rush_hour"] = df["hour"].between(7, 9).astype("int64") | df[
        "hour"
    ].between(16, 18).astype("int64")
    df["is_rush_hour"] = df["is_rush_hour"].astype("int64")
    df["is_night"] = (df["hour"].ge(20) | df["hour"].lt(6)).astype("int64")

    for column in DESCRIPTIVE_CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown")

    df = add_list_like_counts(df, report)
    df["severe_crash"] = df["kabco_severity"].isin(SEVERE_KABCO_VALUES).astype("int64")

    df = df.drop(columns=["date_parsed"]).reset_index(drop=True)
    report.cleaned_row_count = len(df)
    report.missing_after_pct = missing_percentages(df)

    return df, report


def save_cleaned_data(
    df: pd.DataFrame, output_path: Path | str = CLEANED_DATA_PATH
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path
