from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_features import FEATURE_COLUMNS, LEAKAGE_COLUMNS
from src.data.clean_data import COUNT_COLUMNS


PROCESSED_DIR = Path("data/processed")
CLEANED_PATH = PROCESSED_DIR / "collisions_cleaned.parquet"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
FEATURE_COLUMNS_PATH = PROCESSED_DIR / "feature_columns.json"


@pytest.mark.holdout
def test_processed_data_outputs():
    cleaned = pd.read_parquet(CLEANED_PATH)
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    with FEATURE_COLUMNS_PATH.open(encoding="utf-8") as handle:
        feature_columns = json.load(handle)

    assert not cleaned.duplicated().any()
    assert set(cleaned["severe_crash"].unique()) <= {0, 1}
    assert not train["year"].eq(2025).any()
    assert test["year"].eq(2025).all()
    assert not set(LEAKAGE_COLUMNS) & set(feature_columns)
    assert set(FEATURE_COLUMNS) <= set(train.columns)
    assert set(FEATURE_COLUMNS) <= set(test.columns)
    assert cleaned["latitude"].between(-90, 90).all()
    assert cleaned["longitude"].between(-180, 180).all()

    for column in COUNT_COLUMNS:
        assert cleaned[column].notna().all()
        assert (cleaned[column] >= 0).all()
        assert (cleaned[column] % 1 == 0).all()


if __name__ == "__main__":
    test_processed_data_outputs()
    print("Processed data validation passed.")
