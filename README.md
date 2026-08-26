# atlanta-crash-risk-predictor

Predictive modeling project for Atlanta crash severity risk.

The current repository focuses on the data and modeling pipeline. The trained logistic regression model remains the documented baseline, but the evaluation workflow now also compares an unweighted logistic regression and a HistGradientBoostingClassifier using the same time-based validation strategy.

This is a baseline demo for analysis and screening, not a production decision system.

## What the project does

- Cleans raw crash records into a modeling table with engineered time and location features.
- Uses a chronological split so 2025 remains a holdout test set.
- Trains and compares:
	- weighted logistic regression as the baseline,
	- unweighted logistic regression,
	- HistGradientBoostingClassifier as a stronger tabular model.
- Tunes classification thresholds on validation data instead of the holdout set.
- Reports precision-recall tradeoffs, calibration, Brier score, confusion matrices, ROC-AUC, and PR-AUC.

## Validation strategy

The training pipeline preserves the time ordering of the data.

- Validation uses expanding year folds:
	- train on 2021, validate on 2022
	- train on 2021-2022, validate on 2023
	- train on 2021-2023, validate on 2024
- Final holdout testing remains on 2025 only.
- No resampling is applied before the split.

## Reports and artifacts

The main outputs live under `reports/metrics/` and `reports/figures/`:

- `reports/metrics/baseline_metrics.json` - full model report with validation results, holdout results, calibration, and threshold analysis.
- `reports/metrics/model_comparison.csv` - compact comparison across weighted logistic regression, unweighted logistic regression, and HistGradientBoostingClassifier.
- `reports/metrics/threshold_analysis.csv` - validation threshold sweep for the baseline model.
- `reports/metrics/logistic_regression_test_predictions.csv` - baseline holdout predictions.
- `reports/figures/` - confusion matrix, precision-recall, ROC, and calibration plots.
- `reports/model_performance_findings.md` - summary of current model performance and recommended interpretation.

The saved model artifacts live under `models/`.

## Interpreting the current baseline

Use precision, recall, PR-AUC, and calibration before relying on accuracy. The positive class is rare, so accuracy can look strong even when the model produces many false positives.

The new report also separates the main contributors to low precision:

- class weighting versus no weighting,
- threshold choice,
- model capacity and feature interactions,
- the limited number of positive examples in the dataset.

## Running the baseline workflow

Once the processed data exists, run the modeling pipeline from the project root:

```bash
python -m src.models.train_baseline
```

To validate the saved artifacts and report schema:

```bash
pytest tests/test_baseline_modeling.py
```

## Notes

- The backend and frontend folders are intentionally empty for now.
- The logistic regression baseline is kept as the documented reference point.
- The stronger model comparison is for evaluation only; it does not replace the baseline unless its validation and holdout behavior is clearly better.
