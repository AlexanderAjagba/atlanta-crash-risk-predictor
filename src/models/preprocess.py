from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS_PATH = Path("data/processed/feature_columns.json")

NUMERIC_CONTINUOUS_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "latitude",
    "longitude",
]

BINARY_FEATURES = [
    "is_weekend",
    "is_rush_hour",
    "is_night",
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

LEAKAGE_COLUMNS = [
    "kabco_severity",
    "number_serious_injuries",
    "number_visible_injuries",
    "number_of_fatalities_per_crash",
]


def load_feature_columns(path: Path | str = FEATURE_COLUMNS_PATH) -> list[str]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def get_feature_matrix(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    feature_columns = feature_columns or load_feature_columns()
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    leakage = sorted(set(feature_columns) & set(LEAKAGE_COLUMNS))
    if leakage:
        raise ValueError(f"Leakage columns cannot be used as features: {leakage}")

    return df[feature_columns].copy()


def build_preprocessor(
    X_train: pd.DataFrame,
    dense_output: bool = False,
) -> ColumnTransformer:
    continuous_steps = []
    if X_train[NUMERIC_CONTINUOUS_FEATURES].isna().any().any():
        continuous_steps.append(("imputer", SimpleImputer(strategy="median")))
    continuous_steps.append(("scaler", StandardScaler()))

    binary_steps = []
    if X_train[BINARY_FEATURES].isna().any().any():
        binary_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))

    categorical_steps = []
    if X_train[CATEGORICAL_FEATURES].isna().any().any():
        categorical_steps.append(
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown"))
        )
    categorical_steps.append(
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=not dense_output,
            ),
        )
    )

    transformers = [
        ("continuous", Pipeline(continuous_steps), NUMERIC_CONTINUOUS_FEATURES),
        ("categorical", Pipeline(categorical_steps), CATEGORICAL_FEATURES),
    ]

    if binary_steps:
        transformers.append(("binary", Pipeline(binary_steps), BINARY_FEATURES))
    else:
        transformers.append(("binary", "passthrough", BINARY_FEATURES))

    return ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0.0 if dense_output else 0.3,
    )
