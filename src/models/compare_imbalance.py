"""Compare binary severity imbalance strategies using development years only.

python -m src.models.compare_imbalance --output-dir reports/metrics/imbalance_comparison
Requires xgboost in addition to the project's existing modeling dependencies.
No resampling, early stopping, holdout access, model replacement or Kaggle join.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
from src.models.compare_features import FEATURE_SETS, validate_development_data, alert_budget_metrics
from src.models.preprocess import build_preprocessor
from src.models.time_validation import VALIDATION_FOLDS, make_logistic_pipeline
from src.models.evaluate import classification_metrics

FEATURES = FEATURE_SETS['time_location_only']
XGB_PARAMETERS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=5.0, objective='binary:logistic',
    eval_metric='logloss', tree_method='hist', n_jobs=2, random_state=42)


def training_class_ratio(y):
    labels = pd.Series(y)
    if not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError('Training labels must include both binary classes.')
    return float(labels.eq(0).sum() / labels.eq(1).sum())


def make_xgb_pipeline(X, scale_pos_weight=1.0):
    # Dense one-hot output keeps explicit zeros distinct from XGBoost missing values.
    return Pipeline([('preprocess', build_preprocessor(X, dense_output=True)),
        ('model', XGBClassifier(**XGB_PARAMETERS, scale_pos_weight=scale_pos_weight))])


def spatial_cells(frame, size_m=1000):
    """Exploratory local equirectangular grid; NOT road segments or event linkage."""
    if size_m <= 0:
        raise ValueError('Grid size must be positive.')
    xy = frame[['latitude', 'longitude']].astype(float)
    if not np.isfinite(xy.to_numpy()).all():
        raise ValueError('Spatial audit requires finite coordinates.')
    east = np.radians(xy.longitude + 84.388) * 6371008.8 * np.cos(np.radians(33.749))
    north = np.radians(xy.latitude - 33.749) * 6371008.8
    return (np.floor(east / size_m).astype(int).astype(str) + ':' +
            np.floor(north / size_m).astype(int).astype(str))


def run_comparison(train_path='data/processed/train.parquet',
                   output_dir='reports/metrics/imbalance_comparison'):
    path = Path(train_path)
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError('Choose a new output directory; existing reports are preserved.')
    frame = pd.read_parquet(path)
    validate_development_data(frame)
    before = pd.util.hash_pandas_object(frame, index=True)
    out.mkdir(parents=True)
    yearly = frame.groupby('year').severe_crash.agg(rows='size', severe='sum', prevalence='mean')
    yearly['non_severe'] = yearly.rows - yearly.severe
    yearly.to_csv(out / 'class_counts_by_year.csv')
    spatial_summary = []
    for size in [500, 1000, 2000]:
        audit = frame[['year', 'severe_crash']].copy()
        audit['cell'] = spatial_cells(frame, size)
        by_cell = audit.groupby('cell').severe_crash.agg(rows='size', severe='sum')
        by_cell.to_csv(out / f'spatial_support_{size}m.csv')
        spatial_summary.append(dict(cell_size_m=size, occupied_cells=len(by_cell),
            cells_without_severe=int(by_cell.severe.eq(0).sum()),
            cells_with_under_5_severe=int(by_cell.severe.lt(5).sum()),
            median_crash_records=float(by_cell.rows.median())))
    pd.DataFrame(spatial_summary).to_csv(out / 'spatial_support_summary.csv', index=False)
    metrics_rows, budget_rows, calibration_rows, fold_rows, support_rows = [], [], [], [], []
    for years, year in VALIDATION_FOLDS:
        train = frame[frame.year.isin(years)]
        valid = frame[frame.year.eq(year)]
        if train.crash_datetime.max() >= valid.crash_datetime.min():
            raise ValueError('Training must precede validation.')
        ratio = training_class_ratio(train.severe_crash)
        fold_rows.append(dict(train_years=str(years), validation_year=year,
            train_rows=len(train), train_severe=int(train.severe_crash.sum()),
            validation_rows=len(valid), validation_severe=int(valid.severe_crash.sum()),
            training_negative_positive_ratio=ratio))
        # Local evidence available BEFORE each validation year, not cumulative future support.
        counts = train.assign(cell=spatial_cells(train)).groupby('cell').severe_crash.agg(['size','sum'])
        cells = spatial_cells(valid)
        support_rows.append(dict(validation_year=year,
            fraction_rows_in_unseen_cells=float((~cells.isin(counts.index)).mean()),
            fraction_rows_in_cells_with_under_5_prior_severe=float(cells.map(counts['sum']).fillna(0).lt(5).mean())))
        X, V = train[FEATURES].copy(), valid[FEATURES].copy()
        makers = {
            'logistic_unweighted': (lambda: make_logistic_pipeline(X, weighted=False), 1.0),
            'xgboost_unweighted': (lambda: make_xgb_pipeline(X, 1.0), 1.0),
            'xgboost_weighted': (lambda: make_xgb_pipeline(X, ratio), ratio),
        }
        for name, (maker, weight) in makers.items():
            print(f'{year}: {name}, positive weight={weight:.4f}', flush=True)
            with threadpool_limits(limits=2):
                model = maker()
                # No sample_weight or eval_set: outer validation cannot influence fitting.
                model.fit(X, train.severe_crash)
                probabilities = model.predict_proba(V)[:, 1]
            m = classification_metrics(valid.severe_crash, probabilities)
            tn, fp, fn, tp = np.asarray(m['confusion_matrix']).ravel()
            record = dict(model=name, validation_year=year, scale_pos_weight=weight,
                mean_probability=float(probabilities.mean()), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                dummy_average_precision=float(valid.severe_crash.mean()),
                dummy_brier_score=float(np.mean((valid.severe_crash-train.severe_crash.mean())**2)),
                **{k:v for k,v in m.items() if k not in ['calibration_curve','confusion_matrix']})
            metrics_rows.append(record)
            calibration_rows.extend(dict(model=name, validation_year=year, **row) for row in m['calibration_curve'])
            for fraction in [.05,.1,.2]:
                budget_rows.append(dict(model=name, validation_year=year,
                    **alert_budget_metrics(valid.severe_crash, probabilities, fraction)))
            pd.DataFrame(metrics_rows).to_csv(out / 'fold_metrics.csv', index=False)
            pd.DataFrame(budget_rows).to_csv(out / 'alert_budgets.csv', index=False)
            pd.DataFrame(calibration_rows).to_csv(out / 'calibration_bins.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out / 'fold_class_weights.csv', index=False)
    pd.DataFrame(support_rows).to_csv(out / 'prior_spatial_support.csv', index=False)
    selected = ['average_precision','brier_score','roc_auc','precision','recall',
                'positive_prediction_rate','mean_probability','actual_positive_rate',
                'dummy_average_precision','dummy_brier_score']
    summary = pd.DataFrame(metrics_rows).groupby('model')[selected].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(out / 'summary.csv')
    pd.testing.assert_series_equal(before, pd.util.hash_pandas_object(frame, index=True))
    metadata = dict(source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        features=FEATURES, xgboost_parameters=XGB_PARAMETERS, validation_folds=VALIDATION_FOLDS,
        xgboost=xgboost.__version__, sklearn=sklearn.__version__, pandas=pd.__version__, python=platform.python_version(),
        holdout_read=False, resampling=False, sample_weight_used=False, early_stopping=False,
        saved_models_replaced=False, input_frame_unchanged=True,
        notes=['Fixed initial XGBoost configuration, not a hyperparameter search.',
               'Natural validation prevalence retained; weights computed from each training fold only.',
               'Alert-budget metrics are retrospective rank-based capacity comparisons.',
               'Scores are conditional on a recorded crash; spatial support is an exploratory grid.',
               'Equal-weight yearly means; standard deviations are not confidence intervals.'])
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(summary.to_string(), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-path', default='data/processed/train.parquet')
    parser.add_argument('--output-dir', default='reports/metrics/imbalance_comparison')
    args = parser.parse_args()
    run_comparison(args.train_path, args.output_dir)

if __name__ == '__main__':
    main()
