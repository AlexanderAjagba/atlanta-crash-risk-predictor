import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.compare_imbalance import training_class_ratio,make_xgb_pipeline,spatial_cells,run_comparison


def test_ratio_uses_only_supplied_training_labels():
    assert training_class_ratio([0,0,0,1]) == 3
    with pytest.raises(ValueError): training_class_ratio([0,0])


def test_xgb_normalizes_geography_and_preserves_input():
    X=pd.DataFrame({'hour':np.arange(20,dtype=float),'geolocated_county':['FULTON']*20})
    before=X.copy(deep=True)
    model=make_xgb_pipeline(X,3)
    model.set_params(model__n_estimators=3,model__min_child_weight=1)
    model.fit(X,np.array([0,0,0,1]*5))
    np.testing.assert_allclose(model.predict_proba(X),model.predict_proba(X.assign(geolocated_county=' Fulton ')))
    assert model.named_steps['model'].get_params()['scale_pos_weight']==3
    assert isinstance(model.named_steps['preprocess'].transform(X),np.ndarray)
    pd.testing.assert_frame_equal(X,before)


def test_spatial_audit_preserves_rows_and_is_location_only():
    frame=pd.DataFrame({'latitude':[33.75,33.75],'longitude':[-84.38,-84.38],'severe_crash':[0,1]})
    result=spatial_cells(frame)
    assert result.iloc[0]==result.iloc[1]
    with pytest.raises(ValueError):spatial_cells(frame,0)


def test_runner_rejects_holdout_before_creating_reports(tmp_path):
    path=tmp_path/'bad.parquet'
    pd.DataFrame({'year':[2025],'crash_datetime':[pd.Timestamp('2025-01-01')],'severe_crash':[0]}).to_parquet(path)
    out=tmp_path/'reports'
    with pytest.raises(ValueError,match='no holdout'):run_comparison(path,out)
    assert not out.exists()
