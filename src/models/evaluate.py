from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("reports/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


REQUIRED_METRICS = [
    "confusion_matrix",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "precision_recall_auc",
    "brier_score",
    "positive_prediction_rate",
    "actual_positive_rate",
    "accuracy",
]


def _calibration_bins(
    y_true: pd.Series,
    y_probability,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    frame = pd.DataFrame(
        {
            "y_true": pd.Series(y_true).reset_index(drop=True).astype(float),
            "y_probability": pd.Series(y_probability).reset_index(drop=True).astype(float),
        }
    )

    if frame["y_probability"].nunique() <= 1:
        return [
            {
                "bin": 1.0,
                "mean_predicted_probability": float(frame["y_probability"].mean()),
                "observed_positive_rate": float(frame["y_true"].mean()),
                "count": float(len(frame)),
            }
        ]

    bin_count = min(n_bins, frame["y_probability"].nunique())
    try:
        frame["bin"] = pd.qcut(
            frame["y_probability"],
            q=bin_count,
            duplicates="drop",
        )
    except ValueError:
        frame["bin"] = 1

    grouped = (
        frame.groupby("bin", observed=True)
        .agg(
            mean_predicted_probability=("y_probability", "mean"),
            observed_positive_rate=("y_true", "mean"),
            count=("y_true", "size"),
        )
        .reset_index(drop=True)
    )

    grouped.insert(0, "bin", range(1, len(grouped) + 1))
    return [
        {
            "bin": float(row["bin"]),
            "mean_predicted_probability": float(row["mean_predicted_probability"]),
            "observed_positive_rate": float(row["observed_positive_rate"]),
            "count": float(row["count"]),
        }
        for _, row in grouped.iterrows()
    ]


def classification_metrics(
    y_true: pd.Series,
    y_probability,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    probability_series = pd.Series(y_probability).reset_index(drop=True)
    y_pred = (y_probability >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true_series, probability_series)

    metrics = {
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "average_precision": float(average_precision_score(y_true, y_probability)),
        "precision_recall_auc": float(auc(recall, precision)),
        "brier_score": float(brier_score_loss(y_true_series, probability_series)),
        "positive_prediction_rate": float(y_pred.mean()),
        "actual_positive_rate": float(y_true_series.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "calibration_curve": _calibration_bins(y_true_series, probability_series),
    }
    return metrics


def save_json(payload: dict[str, Any], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return output_path


def save_predictions(
    y_true: pd.Series,
    y_probability,
    threshold: float,
    output_path: Path | str,
    crash_datetime: pd.Series | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "actual_label": y_true.to_numpy(),
            "predicted_label": (y_probability >= threshold).astype(int),
            "predicted_severe_crash_probability": y_probability,
        }
    )
    if crash_datetime is not None:
        predictions.insert(0, "crash_datetime", crash_datetime.to_numpy())
    predictions.to_csv(output_path, index=False)
    return output_path


def save_evaluation_figures(
    y_true: pd.Series,
    y_probability,
    threshold: float,
    output_prefix: Path | str,
) -> dict[str, str]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    y_pred = (y_probability >= threshold).astype(int)

    paths = {
        "confusion_matrix": output_prefix.with_name(
            f"{output_prefix.name}_confusion_matrix.png"
        ),
        "calibration_curve": output_prefix.with_name(
            f"{output_prefix.name}_calibration_curve.png"
        ),
        "precision_recall_curve": output_prefix.with_name(
            f"{output_prefix.name}_precision_recall_curve.png"
        ),
        "roc_curve": output_prefix.with_name(f"{output_prefix.name}_roc_curve.png"),
    }

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, values_format="d")
    plt.title(f"Confusion Matrix, threshold={threshold:.3f}")
    plt.tight_layout()
    plt.savefig(paths["confusion_matrix"], dpi=150)
    plt.close()

    CalibrationDisplay.from_predictions(y_true, y_probability, n_bins=10, strategy="quantile")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig(paths["calibration_curve"], dpi=150)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_probability)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(paths["precision_recall_curve"], dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(y_true, y_probability)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(paths["roc_curve"], dpi=150)
    plt.close()

    return {key: str(path) for key, path in paths.items()}


def metric_subset(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "average_precision": metrics["average_precision"],
        "precision_recall_auc": metrics["precision_recall_auc"],
        "brier_score": metrics["brier_score"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    }
