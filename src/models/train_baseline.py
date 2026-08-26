from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.models.evaluate import (
    classification_metrics,
    save_evaluation_figures,
    save_json,
    save_predictions,
)
from src.models.preprocess import build_preprocessor, get_feature_matrix
from src.models.time_validation import (
    expanding_year_validation,
    make_hist_gradient_boosting_pipeline,
    make_logistic_pipeline,
    threshold_analysis,
)


TRAIN_PATH = Path("data/processed/train.parquet")
TEST_PATH = Path("data/processed/test.parquet")
MODELS_DIR = Path("models")
METRICS_DIR = Path("reports/metrics")
FIGURES_DIR = Path("reports/figures")


MODEL_LABELS = {
    "logistic_regression_weighted": "Logistic regression (weighted)",
    "logistic_regression_unweighted": "Logistic regression (unweighted)",
    "hist_gradient_boosting": "HistGradientBoostingClassifier",
}


def _date_range(df: pd.DataFrame) -> str:
    return f"{df['crash_datetime'].min().date()} to {df['crash_datetime'].max().date()}"


def _class_distribution(df: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in df["severe_crash"].value_counts().sort_index().items()
    }


def _print_metric_line(label: str, metrics: dict[str, Any]) -> None:
    print(
        f"{label}: AP={metrics['average_precision']:.4f}, "
        f"PR-AUC={metrics['precision_recall_auc']:.4f}, "
        f"Brier={metrics['brier_score']:.4f}, "
        f"ROC-AUC={metrics['roc_auc']:.4f}, "
        f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}, accuracy={metrics['accuracy']:.4f}"
    )


def _fit_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_sample_weight: bool = False,
) -> Pipeline:
    fit_kwargs: dict[str, Any] = {}
    if use_sample_weight:
        fit_kwargs["model__sample_weight"] = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )
    pipeline.fit(X_train, y_train, **fit_kwargs)
    return pipeline


def _run_model_experiment(
    model_key: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    build_pipeline,
    output_prefix: str,
    model_path: Path,
    prediction_path: Path,
    threshold_table_path: Path,
    use_sample_weight: bool = False,
) -> dict[str, Any]:
    validation = expanding_year_validation(
        train,
        build_pipeline,
        use_sample_weight=use_sample_weight,
    )
    thresholds = threshold_analysis(validation["validation_predictions"])
    selected_threshold = thresholds["selected_threshold"]

    X_train = get_feature_matrix(train)
    y_train = train["severe_crash"]
    X_test = get_feature_matrix(test)
    y_test = test["severe_crash"]

    pipeline = build_pipeline(X_train)
    pipeline = _fit_pipeline(
        pipeline,
        X_train,
        y_train,
        use_sample_weight=use_sample_weight,
    )
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics_0_5 = classification_metrics(y_test, probabilities, threshold=0.5)
    metrics_selected = classification_metrics(
        y_test,
        probabilities,
        threshold=selected_threshold,
    )

    joblib.dump(pipeline, model_path)

    save_predictions(
        y_test,
        probabilities,
        selected_threshold,
        prediction_path,
        crash_datetime=test["crash_datetime"],
    )

    figure_paths = save_evaluation_figures(
        y_test,
        probabilities,
        selected_threshold,
        FIGURES_DIR / output_prefix,
    )

    pd.DataFrame(thresholds["threshold_table"]).to_csv(threshold_table_path, index=False)

    return {
        "label": MODEL_LABELS[model_key],
        "validation": {
            "folds": validation["folds"],
            "summary": validation["summary"],
        },
        "threshold_analysis": {
            "objective": thresholds["objective"],
            "precision_floor": thresholds["precision_floor"],
            "selected_threshold": selected_threshold,
            "threshold_table_path": str(threshold_table_path),
        },
        "holdout_threshold_0_5": metrics_0_5,
        "holdout_selected_threshold": metrics_selected,
        "artifacts": {
            "pipeline": str(model_path),
            "predictions": str(prediction_path),
            "figures": figure_paths,
        },
    }


def _build_model_comparison_rows(
    model_runs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key, run in model_runs.items():
        validation_summary = run["validation"]["summary"]
        selected = run["holdout_selected_threshold"]
        default = run["holdout_threshold_0_5"]
        rows.append(
            {
                "model_key": model_key,
                "model_label": run["label"],
                "validation_average_precision_mean": validation_summary["average_precision"]["mean"],
                "validation_precision_recall_auc_mean": validation_summary["precision_recall_auc"]["mean"],
                "validation_brier_score_mean": validation_summary["brier_score"]["mean"],
                "validation_recall_mean": validation_summary["recall"]["mean"],
                "validation_precision_mean": validation_summary["precision"]["mean"],
                "validation_f1_mean": validation_summary["f1"]["mean"],
                "validation_roc_auc_mean": validation_summary["roc_auc"]["mean"],
                "validation_accuracy_mean": validation_summary["accuracy"]["mean"],
                "selected_threshold": run["threshold_analysis"]["selected_threshold"],
                "holdout_precision_at_0_5": default["precision"],
                "holdout_recall_at_0_5": default["recall"],
                "holdout_f1_at_0_5": default["f1"],
                "holdout_precision_at_selected": selected["precision"],
                "holdout_recall_at_selected": selected["recall"],
                "holdout_f1_at_selected": selected["f1"],
                "holdout_precision_recall_auc": selected["precision_recall_auc"],
                "holdout_brier_score": selected["brier_score"],
                "holdout_roc_auc": selected["roc_auc"],
            }
        )
    return rows


def _summarize_precision_recall_tradeoff(
    weighted_logistic: dict[str, Any],
    unweighted_logistic: dict[str, Any],
    hgb: dict[str, Any],
    train: pd.DataFrame,
) -> dict[str, Any]:
    train_positive_rate = float(train["severe_crash"].mean())

    weighted = weighted_logistic["holdout_selected_threshold"]
    unweighted = unweighted_logistic["holdout_selected_threshold"]
    hgb_selected = hgb["holdout_selected_threshold"]

    return {
        "train_positive_rate": train_positive_rate,
        "weighted_vs_unweighted_logistic": {
            "weighted_precision": weighted["precision"],
            "weighted_recall": weighted["recall"],
            "weighted_brier_score": weighted["brier_score"],
            "unweighted_precision": unweighted["precision"],
            "unweighted_recall": unweighted["recall"],
            "unweighted_brier_score": unweighted["brier_score"],
            "interpretation": (
                "Class weighting shifts the logistic model toward more positive predictions, "
                "which raises recall but usually lowers precision by increasing false positives."
            ),
        },
        "threshold_effect": {
            "weighted_logistic_threshold_0_5_precision": weighted_logistic["holdout_threshold_0_5"][
                "precision"
            ],
            "weighted_logistic_threshold_0_5_recall": weighted_logistic["holdout_threshold_0_5"][
                "recall"
            ],
            "weighted_logistic_selected_threshold": weighted_logistic["threshold_analysis"][
                "selected_threshold"
            ],
            "weighted_logistic_selected_precision": weighted["precision"],
            "weighted_logistic_selected_recall": weighted["recall"],
            "interpretation": (
                "Threshold choice is a major driver of the alert rate: the validation-selected threshold "
                "optimizes recall subject to a precision floor, which is more useful than raw accuracy for this imbalanced problem."
            ),
        },
        "model_capacity_effect": {
            "hist_gradient_boosting_precision": hgb_selected["precision"],
            "hist_gradient_boosting_recall": hgb_selected["recall"],
            "hist_gradient_boosting_pr_auc": hgb_selected["precision_recall_auc"],
            "weighted_logistic_precision": weighted["precision"],
            "weighted_logistic_recall": weighted["recall"],
            "weighted_logistic_pr_auc": weighted["precision_recall_auc"],
            "interpretation": (
                "If HistGradientBoosting improves PR-AUC or Brier score over logistic regression, "
                "the feature interactions are limiting performance more than the threshold alone."
            ),
        },
    }


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    return train, test


def train_dummy_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X_train)),
            ("model", DummyClassifier(strategy="prior", random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def run_baseline_workflow() -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_processed_data()
    X_train = get_feature_matrix(train)
    y_train = train["severe_crash"]
    X_test = get_feature_matrix(test)
    y_test = test["severe_crash"]

    dummy_pipeline = train_dummy_pipeline(X_train, y_train)

    dummy_model_path = MODELS_DIR / "dummy_baseline_pipeline.joblib"
    joblib.dump(dummy_pipeline, dummy_model_path)

    logistic_weighted = _run_model_experiment(
        model_key="logistic_regression_weighted",
        train=train,
        test=test,
        build_pipeline=lambda X: make_logistic_pipeline(X, weighted=True),
        output_prefix="logistic_regression_test",
        model_path=MODELS_DIR / "logistic_regression_baseline_pipeline.joblib",
        prediction_path=METRICS_DIR / "logistic_regression_test_predictions.csv",
        threshold_table_path=METRICS_DIR / "threshold_analysis.csv",
        use_sample_weight=False,
    )
    logistic_unweighted = _run_model_experiment(
        model_key="logistic_regression_unweighted",
        train=train,
        test=test,
        build_pipeline=lambda X: make_logistic_pipeline(X, weighted=False),
        output_prefix="logistic_regression_unweighted_test",
        model_path=MODELS_DIR / "logistic_regression_unweighted_pipeline.joblib",
        prediction_path=METRICS_DIR / "logistic_regression_unweighted_test_predictions.csv",
        threshold_table_path=METRICS_DIR / "logistic_regression_unweighted_threshold_analysis.csv",
        use_sample_weight=False,
    )
    hist_gradient_boosting = _run_model_experiment(
        model_key="hist_gradient_boosting",
        train=train,
        test=test,
        build_pipeline=make_hist_gradient_boosting_pipeline,
        output_prefix="hist_gradient_boosting_test",
        model_path=MODELS_DIR / "hist_gradient_boosting_pipeline.joblib",
        prediction_path=METRICS_DIR / "hist_gradient_boosting_test_predictions.csv",
        threshold_table_path=METRICS_DIR / "hist_gradient_boosting_threshold_analysis.csv",
        use_sample_weight=True,
    )

    validation = logistic_weighted["validation"]
    thresholds = logistic_weighted["threshold_analysis"]
    selected_threshold = thresholds["selected_threshold"]

    dummy_probability = dummy_pipeline.predict_proba(X_test)[:, 1]
    dummy_metrics = classification_metrics(y_test, dummy_probability, threshold=0.5)

    logistic_default_metrics = logistic_weighted["holdout_threshold_0_5"]
    logistic_selected_metrics = logistic_weighted["holdout_selected_threshold"]

    model_comparison_rows = _build_model_comparison_rows(
        {
            "logistic_regression_weighted": logistic_weighted,
            "logistic_regression_unweighted": logistic_unweighted,
            "hist_gradient_boosting": hist_gradient_boosting,
        }
    )
    model_comparison_path = METRICS_DIR / "model_comparison.csv"
    pd.DataFrame(model_comparison_rows).to_csv(model_comparison_path, index=False)

    precision_recall_tradeoff = _summarize_precision_recall_tradeoff(
        logistic_weighted,
        logistic_unweighted,
        hist_gradient_boosting,
        train,
    )

    metrics_payload = {
        "data": {
            "train_size": int(len(train)),
            "test_size": int(len(test)),
            "train_date_range": _date_range(train),
            "test_date_range": _date_range(test),
            "train_class_distribution": _class_distribution(train),
            "test_class_distribution": _class_distribution(test),
        },
        "cross_validation": {
            "folds": validation["folds"],
            "summary": validation["summary"],
        },
        "threshold_analysis": {
            "objective": thresholds["objective"],
            "precision_floor": thresholds["precision_floor"],
            "selected_threshold": selected_threshold,
            "threshold_table_path": str(METRICS_DIR / "threshold_analysis.csv"),
        },
        "dummy_baseline": dummy_metrics,
        "logistic_regression_threshold_0_5": logistic_default_metrics,
        "logistic_regression_selected_threshold": logistic_selected_metrics,
        "logistic_regression_unweighted_threshold_0_5": logistic_unweighted["holdout_threshold_0_5"],
        "logistic_regression_unweighted_selected_threshold": logistic_unweighted["holdout_selected_threshold"],
        "hist_gradient_boosting_threshold_0_5": hist_gradient_boosting["holdout_threshold_0_5"],
        "hist_gradient_boosting_selected_threshold": hist_gradient_boosting["holdout_selected_threshold"],
        "candidate_models": {
            "logistic_regression_weighted": logistic_weighted,
            "logistic_regression_unweighted": logistic_unweighted,
            "hist_gradient_boosting": hist_gradient_boosting,
        },
        "model_comparison": {
            "table_path": str(model_comparison_path),
            "rows": model_comparison_rows,
        },
        "precision_recall_tradeoff": precision_recall_tradeoff,
        "artifacts": {
            "dummy_pipeline": str(dummy_model_path),
            "logistic_regression_pipeline": str(
                MODELS_DIR / "logistic_regression_baseline_pipeline.joblib"
            ),
            "logistic_regression_unweighted_pipeline": str(
                MODELS_DIR / "logistic_regression_unweighted_pipeline.joblib"
            ),
            "hist_gradient_boosting_pipeline": str(
                MODELS_DIR / "hist_gradient_boosting_pipeline.joblib"
            ),
            "metrics": str(METRICS_DIR / "baseline_metrics.json"),
            "predictions": str(METRICS_DIR / "logistic_regression_test_predictions.csv"),
            "figures": logistic_weighted["artifacts"]["figures"],
            "model_comparison": str(model_comparison_path),
        },
    }

    metrics_path = save_json(metrics_payload, METRICS_DIR / "baseline_metrics.json")
    metrics_payload["artifacts"]["metrics"] = str(metrics_path)

    print("Baseline modeling summary")
    print(f"Train size: {len(train):,} ({_date_range(train)})")
    print(f"Test size: {len(test):,} ({_date_range(test)})")
    print(f"Train class distribution: {_class_distribution(train)}")
    print(f"Test class distribution: {_class_distribution(test)}")
    print("Cross-validation metric means:")
    for metric_name, values in validation["summary"].items():
        print(f"  - {metric_name}: {values['mean']:.4f} +/- {values['std']:.4f}")
    _print_metric_line("Dummy baseline, threshold 0.5", dummy_metrics)
    _print_metric_line("Logistic regression (weighted), threshold 0.5", logistic_default_metrics)
    _print_metric_line(
        f"Logistic regression (weighted), selected threshold {selected_threshold:.3f}",
        logistic_selected_metrics,
    )
    _print_metric_line(
        "Logistic regression (unweighted), selected threshold",
        logistic_unweighted["holdout_selected_threshold"],
    )
    _print_metric_line(
        "HistGradientBoostingClassifier, selected threshold",
        hist_gradient_boosting["holdout_selected_threshold"],
    )
    print(f"Threshold objective: {thresholds['objective']}")
    print(f"Precision-recall tradeoff summary: {json.dumps(precision_recall_tradeoff, indent=2)}")
    print("Saved artifacts:")
    print(json.dumps(metrics_payload["artifacts"], indent=2))

    return metrics_payload


def main() -> None:
    run_baseline_workflow()


if __name__ == "__main__":
    main()
