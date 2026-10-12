import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.preprocess import build_preprocessor, get_feature_matrix
from src.models.compare_features import FEATURE_SETS, EVENT_FEATURES, REPORT_ENVIRONMENT_FEATURES, validate_development_data, alert_budget_metrics
from src.models.time_validation import expanding_year_validation, make_logistic_pipeline


def small_frame():
    rows=[]
    for year in [2021,2022,2023,2024]:
        for i in range(12):
            rows.append({'crash_datetime':pd.Timestamp(year,1,i+1),'year':year,'severe_crash':i%2,
                         'hour':i,'geolocated_city':'A' if i%2 else 'B'})
    return pd.DataFrame(rows)


def test_restricted_sets_exclude_event_and_report_environment():
    assert len(FEATURE_SETS['current'])==15
    assert len(FEATURE_SETS['without_event_details'])==13
    assert len(FEATURE_SETS['time_location_only'])==10
    assert not set(FEATURE_SETS['time_location_only']) & set(EVENT_FEATURES+REPORT_ENVIRONMENT_FEATURES)


def test_subset_preprocessor_handles_unseen_categories_without_excluded_columns():
    x=pd.DataFrame({'hour':[1.,2.,np.nan], 'geolocated_city':['A','B','A']})
    before=x.copy(deep=True)
    pre=build_preprocessor(x)
    pre.fit(x)
    assert pre.transform(pd.DataFrame({'hour':[3.], 'geolocated_city':['unseen']})).shape[0]==1
    pd.testing.assert_frame_equal(x,before)
    assert list(pre.feature_names_in_)==list(x.columns)
    assert build_preprocessor(x[['hour']]).fit_transform(x[['hour']]).shape==(3,1)


def test_empty_feature_selection_is_not_silently_full_set():
    with pytest.raises(ValueError,match='At least one'):
        get_feature_matrix(small_frame(),[])


def test_holdout_and_mismatched_dates_rejected():
    frame=small_frame();validate_development_data(frame)
    bad=frame.copy();bad.loc[0,'year']=2025
    with pytest.raises(ValueError,match='no holdout'):validate_development_data(bad)
    bad=frame.copy();bad.loc[0,'crash_datetime']=pd.Timestamp('2025-01-01')
    with pytest.raises(ValueError,match='disagrees'):validate_development_data(bad)


def test_selected_features_used_in_each_fold_and_sources_unchanged():
    frame=small_frame();before=frame.copy(deep=True);seen=[]
    def constructor(x):
        seen.append(tuple(x.columns))
        return make_logistic_pipeline(x,weighted=False)
    result=expanding_year_validation(frame,constructor,feature_columns=['hour'])
    assert seen==[('hour',)]*3
    assert [m['validation_year'] for m in result['folds']]==[2022,2023,2024]
    pd.testing.assert_frame_equal(frame,before)


def test_budget_does_not_break_ties_using_target():
    scores=np.array([.9,.9,.2,.1])
    m=alert_budget_metrics([0,1,1,0],scores,.25)
    assert m['alerts']==1 and m['true_positives']==0
    assert m['rows_tied_at_boundary']==2
