"""Validation-only severity feature ablation. Never reads the final holdout.

Run from the project root:
    python -m src.models.compare_features
Reports go to reports/metrics/feature_comparison; existing models are not saved over.
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
from threadpoolctl import threadpool_limits
from src.data.build_features import FEATURE_COLUMNS
from src.models.time_validation import (
    VALIDATION_FOLDS, expanding_year_validation, make_logistic_pipeline,
    make_hist_gradient_boosting_pipeline,
)

EVENT_FEATURES = ['location_at_impact_crash_level', 'manner_of_collision_crash_level']
REPORT_ENVIRONMENT_FEATURES = ['light_conditions_crash_level', 'surface_condition_crash_level', 'weather_conditions_crash_level']
FEATURE_SETS = {
    'current': list(FEATURE_COLUMNS),
    'without_event_details': [c for c in FEATURE_COLUMNS if c not in EVENT_FEATURES],
    'time_location_only': [c for c in FEATURE_COLUMNS if c not in EVENT_FEATURES + REPORT_ENVIRONMENT_FEATURES],
}

def validate_development_data(frame: pd.DataFrame) -> None:
    """Reject holdout contamination rather than silently dropping unexpected years."""
    if frame.empty or frame['crash_datetime'].isna().any():
        raise ValueError('Development data must have nonmissing timestamps.')
    if not frame['year'].isin([2021, 2022, 2023, 2024]).all():
        raise ValueError('Feature comparison accepts 2021–2024 development rows only; no holdout.')
    if not frame['crash_datetime'].dt.year.eq(frame['year']).all():
        raise ValueError('Year metadata disagrees with timestamps.')
    if not frame['severe_crash'].isin([0, 1]).all():
        raise ValueError('Target must contain only binary severity labels.')
    for years, year in VALIDATION_FOLDS:
        for subset in [frame[frame.year.isin(years)], frame[frame.year.eq(year)]]:
            if subset.empty or subset.severe_crash.nunique() != 2:
                raise ValueError(f'Each fold must contain both severity classes: {year}')

def alert_budget_metrics(y_true, probabilities, fraction: float) -> dict:
    """Retrospective capacity analysis; rank ties by row order, never by label.
    These are NOT calibrated deployed probability thresholds.
    """
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities)
    count = int(np.ceil(len(y) * fraction))
    ranked = np.argsort(-scores, kind='stable')
    chosen = ranked[:count]
    tp = int(y[chosen].sum())
    cutoff = float(scores[chosen[-1]])
    return {'alert_budget_fraction': fraction, 'alerts': count,
        'true_positives': tp, 'false_positives': count-tp,
        'false_negatives': int(y.sum())-tp, 'precision': tp/count,
        'recall': tp/int(y.sum()), 'actual_alert_rate': count/len(y),
        'boundary_score': cutoff, 'rows_tied_at_boundary': int((scores == cutoff).sum())}

def run_feature_comparison(train_path='data/processed/train.parquet',
                           output_dir='reports/metrics/feature_comparison') -> pd.DataFrame:
    train_path=Path(train_path); output=Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a fresh output directory to preserve prior comparisons.')
    frame=pd.read_parquet(train_path)
    validate_development_data(frame)
    output.mkdir(parents=True, exist_ok=True)
    constructors = {
        'logistic_weighted': (lambda x: make_logistic_pipeline(x, weighted=True), False),
        'logistic_unweighted': (lambda x: make_logistic_pipeline(x, weighted=False), False),
        'hist_gradient_boosting_weighted': (make_hist_gradient_boosting_pipeline, True),
    }
    rows=[]; budget_rows=[]; calibration_rows=[]
    for feature_set, columns in FEATURE_SETS.items():
        for model, (constructor, weighted) in constructors.items():
            print(f'Running {feature_set}: {model} ({len(columns)} features)', flush=True)
            # Avoid excessive OpenMP threads on desktop machines. Same limit for all fits.
            with threadpool_limits(limits=2):
                result=expanding_year_validation(frame, constructor,
                    use_sample_weight=weighted, feature_columns=columns)
            for metrics, prediction in zip(result['folds'], result['validation_predictions']):
                row={'feature_set':feature_set, 'model':model, 'feature_count':len(columns),
                     **{k:v for k,v in metrics.items() if k not in ['calibration_curve','confusion_matrix','train_years']}}
                tn,fp,fn,tp=np.asarray(metrics['confusion_matrix']).ravel()
                row.update(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
                train_fold=frame[frame.year.isin(metrics['train_years'])]
                prior=float(train_fold.severe_crash.mean())
                row['dummy_average_precision']=float(prediction.y_true.mean())
                row['dummy_brier_score']=float(np.mean((prediction.y_true.to_numpy()-prior)**2))
                rows.append(row)
                for entry in metrics['calibration_curve']:
                    calibration_rows.append({'feature_set':feature_set,'model':model,
                        'validation_year':metrics['validation_year'],**entry})
                for fraction in [.05,.10,.20]:
                    budget_rows.append({'feature_set':feature_set,'model':model,
                        'validation_year':metrics['validation_year'],
                        **alert_budget_metrics(prediction.y_true,prediction.y_probability,fraction)})
            # Checkpoint only this comparison's new reports.
            pd.DataFrame(rows).to_csv(output/'fold_metrics.csv',index=False)
            pd.DataFrame(budget_rows).to_csv(output/'alert_budgets.csv',index=False)
            pd.DataFrame(calibration_rows).to_csv(output/'calibration_bins.csv',index=False)
    folds=pd.DataFrame(rows)
    metrics=['average_precision','brier_score','roc_auc','precision','recall','positive_prediction_rate','dummy_average_precision','dummy_brier_score']
    summary=folds.groupby(['feature_set','model'],sort=False)[metrics].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index()
    summary.to_csv(output/'summary.csv',index=False)
    metadata={'source':str(train_path.resolve()),'source_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),
        'rows':len(frame),'date_min':str(frame.crash_datetime.min()),'date_max':str(frame.crash_datetime.max()),
        'feature_sets':FEATURE_SETS,'folds':VALIDATION_FOLDS,'python':platform.python_version(),
        'pandas':pd.__version__,'sklearn':sklearn.__version__, 'holdout_read':False,
        'threshold':0.5,'primary_metric':'average_precision',
        'notes':['No model or threshold selection on 2025.',
                 'Equal-weight fold means; standard deviations across three years are not confidence intervals.',
                 'Alert budgets rank validation scores; no label-driven threshold tuning.',
                 'Time/location features are candidate-context proxies; this remains severity conditional on a recorded crash.',
                 'Report-derived environmental variables are not verified as prediction-time measurements.']}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-path',default='data/processed/train.parquet')
    parser.add_argument('--output-dir',default='reports/metrics/feature_comparison')
    args=parser.parse_args()
    run_feature_comparison(args.train_path,args.output_dir)

if __name__=='__main__':
    main()
