from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate import REQUIRED_METRICS
from src.models.preprocess import LEAKAGE_COLUMNS, build_preprocessor, get_feature_matrix
from src.models.time_validation import VALIDATION_FOLDS


TRAIN_PATH = Path("data/processed/train.parquet")
METRICS_PATH = Path("reports/metrics/baseline_metrics.json")
LOGISTIC_PIPELINE_PATH = Path("models/logistic_regression_baseline_pipeline.joblib")
UNWEIGHTED_LOGISTIC_PIPELINE_PATH = Path("models/logistic_regression_unweighted_pipeline.joblib")
HGB_PIPELINE_PATH = Path("models/hist_gradient_boosting_pipeline.joblib")


def _load_metrics() -> dict:
    with METRICS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_no_leakage_columns_are_in_model_feature_set():
    train = pd.read_parquet(TRAIN_PATH)
    features = get_feature_matrix(train)
    assert not set(LEAKAGE_COLUMNS) & set(features.columns)


def test_preprocessor_transforms_unseen_categorical_values():
    train = pd.read_parquet(TRAIN_PATH)
    X_train = get_feature_matrix(train)
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    sample = X_train.head(1).copy()
    sample["geolocated_city"] = "Not A City Seen During Training"
    transformed = preprocessor.transform(sample)
    assert transformed.shape[0] == 1


def test_saved_pipeline_outputs_probabilities_between_zero_and_one():
    train = pd.read_parquet(TRAIN_PATH)
    X_train = get_feature_matrix(train).head(25)
    pipeline = joblib.load(LOGISTIC_PIPELINE_PATH)
    probabilities = pipeline.predict_proba(X_train)[:, 1]
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_validation_training_dates_precede_validation_dates():
    train = pd.read_parquet(TRAIN_PATH)
    for train_years, validation_year in VALIDATION_FOLDS:
        fold_train = train.loc[train["year"].isin(train_years)]
        fold_validation = train.loc[train["year"].eq(validation_year)]
        assert fold_train["crash_datetime"].max() < fold_validation["crash_datetime"].min()


def test_2025_is_never_used_in_cross_validation():
    for train_years, validation_year in VALIDATION_FOLDS:
        assert 2025 not in train_years
        assert validation_year != 2025


def test_saved_pipeline_can_be_loaded_and_used_for_prediction():
    train = pd.read_parquet(TRAIN_PATH)
    X_train = get_feature_matrix(train).head(5)
    pipeline = joblib.load(LOGISTIC_PIPELINE_PATH)
    predictions = pipeline.predict(X_train)
    assert len(predictions) == 5


def test_candidate_pipelines_can_be_loaded_and_used_for_prediction():
    train = pd.read_parquet(TRAIN_PATH)
    X_train = get_feature_matrix(train).head(5)
    for pipeline_path in [UNWEIGHTED_LOGISTIC_PIPELINE_PATH, HGB_PIPELINE_PATH]:
        pipeline = joblib.load(pipeline_path)
        predictions = pipeline.predict(X_train)
        assert len(predictions) == 5


@pytest.mark.holdout
def test_all_required_metrics_are_generated():
    metrics = _load_metrics()
    for section in [
        "dummy_baseline",
        "logistic_regression_threshold_0_5",
        "logistic_regression_selected_threshold",
    ]:
        assert set(REQUIRED_METRICS) <= set(metrics[section])

    assert "cross_validation" in metrics
    assert "summary" in metrics["cross_validation"]
    assert "threshold_analysis" in metrics
    assert "candidate_models" in metrics
    assert "model_comparison" in metrics
    assert "precision_recall_tradeoff" in metrics
    assert Path(metrics["artifacts"]["predictions"]).exists()
    assert Path(metrics["artifacts"]["logistic_regression_pipeline"]).exists()
    assert Path(metrics["artifacts"]["logistic_regression_unweighted_pipeline"]).exists()
    assert Path(metrics["artifacts"]["hist_gradient_boosting_pipeline"]).exists()
    assert Path(metrics["artifacts"]["model_comparison"]).exists()

    for model_key in [
        "logistic_regression_weighted",
        "logistic_regression_unweighted",
        "hist_gradient_boosting",
    ]:
        assert set(REQUIRED_METRICS) <= set(
            metrics["candidate_models"][model_key]["holdout_selected_threshold"]
        )


if __name__ == "__main__":
    test_no_leakage_columns_are_in_model_feature_set()
    test_preprocessor_transforms_unseen_categorical_values()
    test_saved_pipeline_outputs_probabilities_between_zero_and_one()
    test_validation_training_dates_precede_validation_dates()
    test_2025_is_never_used_in_cross_validation()
    test_saved_pipeline_can_be_loaded_and_used_for_prediction()
    test_all_required_metrics_are_generated()
    print("Baseline modeling validation passed.")
