from __future__ import annotations

from pathlib import Path

import pandas as pd


TRAIN_PATH = Path("data/processed/train.parquet")
TEST_PATH = Path("data/processed/test.parquet")


def chronological_split(
    model_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_table = model_table.sort_values("crash_datetime").reset_index(drop=True)

    train = sorted_table.loc[
        sorted_table["year"].between(2021, 2024)
    ].reset_index(drop=True)
    test = sorted_table.loc[sorted_table["year"].eq(2025)].reset_index(drop=True)

    return train, test


def save_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_path: Path | str = TRAIN_PATH,
    test_path: Path | str = TEST_PATH,
) -> tuple[Path, Path]:
    train_path = Path(train_path)
    test_path = Path(test_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)

    return train_path, test_path
