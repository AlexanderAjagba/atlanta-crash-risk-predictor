# Model Performance Findings

This note summarizes the regenerated baseline metrics in `reports/metrics/baseline_metrics.json` and `reports/metrics/model_comparison.csv`.

## Data Split

- Training period: 2021-01-01 to 2024-12-31.
- Holdout period: 2025-01-01 to 2025-12-31.
- Training rows: 140,806, with 2,475 severe crashes.
- Holdout rows: 34,163, with 521 severe crashes.
- Severe crashes are rare in both splits, about 1.8% in training and 1.5% in holdout.

## Main Takeaways

- The weighted logistic regression remains the best documented baseline for the current workflow. On the 2025 holdout it achieved ROC-AUC 0.806, PR-AUC 0.069, precision 0.058, recall 0.628, and F1 0.106 at the selected threshold of 0.50.
- HistGradientBoostingClassifier did not clearly improve the holdout results. Its holdout ROC-AUC was 0.799 and PR-AUC was 0.069, with precision 0.055, recall 0.624, and F1 0.101 at threshold 0.50.
- The unweighted logistic regression had the strongest holdout PR-AUC, 0.079, and the best Brier score, 0.0147, but it predicted no positives at the default 0.50 threshold. With the validation-selected threshold of 0.02, recall rose to 0.708 while precision fell to 0.042.
- Accuracy is not a useful headline metric here. The unweighted logistic model had high validation accuracy because severe crashes are rare, but that default behavior missed the positive class.
- Calibration is a concern for the weighted logistic baseline. Its holdout Brier score was 0.130, much worse than the unweighted logistic model's 0.0147, which suggests the weighted model's probabilities should be treated as ranking scores rather than calibrated event probabilities.

## Interpretation

The models can rank higher-risk crashes better than random, but the absolute precision is low because the positive class is very sparse. The current models are more appropriate for analysis, screening, and prioritization experiments than for automated decisions.

The stronger tabular model is useful as a comparison point, but the latest artifacts do not justify replacing the weighted logistic regression baseline. The most promising next step is probably probability calibration and feature improvement, not simply adding model capacity.

