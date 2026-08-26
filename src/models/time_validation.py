from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.models.evaluate import classification_metrics, metric_subset
from src.models.preprocess import build_preprocessor, get_feature_matrix


VALIDATION_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

THRESHOLD_GRID = np.round(np.concatenate([np.arange(0.01, 0.21, 0.01), [0.25, 0.3, 0.4, 0.5]]), 3)
THRESHOLD_PRECISION_FLOOR = 0.05


@dataclass
class ValidationPrediction:
    fold: str
    y_true: pd.Series
    y_probability: np.ndarray


def make_logistic_pipeline(
    X_train: pd.DataFrame,
    weighted: bool = True,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X_train)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced" if weighted else None,
                    max_iter=1000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def make_hist_gradient_boosting_pipeline(X_train: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X_train, dense_output=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=6,
                    max_iter=250,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    random_state=42,
                    early_stopping=False,
                ),
            ),
        ]
    )


def expanding_year_validation(
    train_df: pd.DataFrame,
    make_pipeline,
    use_sample_weight: bool = False,
) -> dict[str, Any]:
    fold_metrics: list[dict[str, Any]] = []
    fold_predictions: list[ValidationPrediction] = []

    for train_years, validation_year in VALIDATION_FOLDS:
        fold_train = train_df.loc[train_df["year"].isin(train_years)].copy()
        fold_validation = train_df.loc[train_df["year"].eq(validation_year)].copy()

        if fold_train["crash_datetime"].max() >= fold_validation["crash_datetime"].min():
            raise ValueError("Validation fold dates are not strictly after training dates.")

        X_train = get_feature_matrix(fold_train)
        y_train = fold_train["severe_crash"]
        X_validation = get_feature_matrix(fold_validation)
        y_validation = fold_validation["severe_crash"]

        pipeline = make_pipeline(X_train)
        fit_kwargs: dict[str, Any] = {}
        if use_sample_weight:
            fit_kwargs["model__sample_weight"] = compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )
        pipeline.fit(X_train, y_train, **fit_kwargs)
        y_probability = pipeline.predict_proba(X_validation)[:, 1]

        metrics = classification_metrics(y_validation, y_probability, threshold=0.5)
        metrics.update(
            {
                "train_years": train_years,
                "validation_year": validation_year,
                "train_size": int(len(fold_train)),
                "validation_size": int(len(fold_validation)),
            }
        )
        fold_metrics.append(metrics)
        fold_predictions.append(
            ValidationPrediction(
                fold=f"{min(train_years)}-{max(train_years)}_to_{validation_year}",
                y_true=y_validation.reset_index(drop=True),
                y_probability=y_probability,
            )
        )

    metric_names = [
        "average_precision",
        "precision_recall_auc",
        "brier_score",
        "recall",
        "precision",
        "f1",
        "roc_auc",
        "accuracy",
    ]
    summary = {}
    for metric_name in metric_names:
        values = [metrics[metric_name] for metrics in fold_metrics]
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }

    return {
        "folds": fold_metrics,
        "summary": summary,
        "validation_predictions": fold_predictions,
    }


def threshold_analysis(
    validation_predictions: list[ValidationPrediction],
    thresholds=THRESHOLD_GRID,
    precision_floor: float = THRESHOLD_PRECISION_FLOOR,
) -> dict[str, Any]:
    rows = []
    y_true = pd.concat([prediction.y_true for prediction in validation_predictions], ignore_index=True)
    y_probability = np.concatenate(
        [prediction.y_probability for prediction in validation_predictions]
    )

    for threshold in thresholds:
        metrics = classification_metrics(y_true, y_probability, threshold=float(threshold))
        rows.append({"threshold": float(threshold), **metric_subset(metrics)})

    threshold_table = pd.DataFrame(rows)
    eligible = threshold_table.loc[threshold_table["precision"].ge(precision_floor)]
    if not eligible.empty:
        selected = eligible.sort_values(
            ["recall", "precision", "f1"], ascending=False
        ).iloc[0]
        objective = (
            "max_validation_recall_with_precision_at_least_"
            f"{precision_floor:.2f}"
        )
    else:
        selected = threshold_table.sort_values(["f1", "recall"], ascending=False).iloc[0]
        objective = "best_validation_f1_no_threshold_met_precision_floor"

    return {
        "objective": objective,
        "precision_floor": precision_floor,
        "selected_threshold": float(selected["threshold"]),
        "threshold_table": threshold_table.to_dict(orient="records"),
    }
