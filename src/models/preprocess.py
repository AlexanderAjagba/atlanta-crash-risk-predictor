from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


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
    feature_columns = load_feature_columns() if feature_columns is None else feature_columns
    if not feature_columns:
        raise ValueError("At least one feature column is required.")
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    leakage = sorted(set(feature_columns) & set(LEAKAGE_COLUMNS))
    if leakage:
        raise ValueError(f"Leakage columns cannot be used as features: {leakage}")

    return df[feature_columns].copy()


def normalize_geographic_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize spelling without fitting on future data or changing inputs.

    County/city case differences are source formatting, not distinct locations.
    Other categorical fields and missing values retain their original semantics.
    Kept inside the pipeline so fit, inference and saved models apply the same rule.
    """
    normalized = frame.copy()
    for column in ("geolocated_city", "geolocated_county"):
        if column in normalized:
            normalized[column] = normalized[column].map(
                lambda value: value.strip().casefold() if isinstance(value, str) else value
            )
    return normalized


def build_preprocessor(
    X_train: pd.DataFrame,
    dense_output: bool = False,
) -> ColumnTransformer:
    # Fit exactly the selected feature subset; the default full set is unchanged.
    known = NUMERIC_CONTINUOUS_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    unknown = sorted(set(X_train.columns) - set(known))
    if unknown or X_train.shape[1] == 0:
        raise ValueError(f"Unsupported or empty feature set: {unknown}")
    continuous = [c for c in NUMERIC_CONTINUOUS_FEATURES if c in X_train]
    binary = [c for c in BINARY_FEATURES if c in X_train]
    categorical = [c for c in CATEGORICAL_FEATURES if c in X_train]
    transformers = []
    if continuous:
        steps = []
        if X_train[continuous].isna().any().any():
            steps.append(("imputer", SimpleImputer(strategy="median")))
        steps.append(("scaler", StandardScaler()))
        transformers.append(("continuous", Pipeline(steps), continuous))
    if categorical:
        steps = [("normalize_geography", FunctionTransformer(
            normalize_geographic_categories, feature_names_out="one-to-one"
        ))]
        if X_train[categorical].isna().any().any():
            steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")))
        steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)))
        transformers.append(("categorical", Pipeline(steps), categorical))
    if binary:
        transformer = (
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
            if X_train[binary].isna().any().any() else "passthrough"
        )
        transformers.append(("binary", transformer, binary))
    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0 if dense_output else 0.3)
