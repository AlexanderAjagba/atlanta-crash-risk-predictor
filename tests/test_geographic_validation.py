from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
import pytest
from src.models.geographic_validation import regions,buffered_training,local_support,run


def test_buffer_excludes_held_quadrant_and_near_boundary_without_labels():
    train=pd.DataFrame({'latitude':[33.77,33.77,33.77,33.74,33.74],
        'longitude':[-84.39,-84.4001,-84.42,-84.42,-84.38], 'severe_crash':[1,0,1,0,1]})
    valid=pd.DataFrame({'latitude':[33.7701],'longitude':[-84.3901]})
    before=train.copy(deep=True)
    kept,dist=buffered_training(train,valid,'NE',500)
    assert set(kept.index)=={2,3,4}
    assert dist.min()>500 and not regions(kept).eq('NE').any()
    changed,_=buffered_training(train.assign(severe_crash=1-train.severe_crash),valid,'NE',500)
    assert changed.index.equals(kept.index)
    pd.testing.assert_frame_equal(train,before)


def test_local_support_counts_only_given_training_records():
    train=pd.DataFrame({'latitude':[33.77,33.77,34.5],'longitude':[-84.39,-84.39,-84.0],'severe_crash':[0,1,1]})
    valid=train.iloc[[0]].copy()
    support=local_support(train,valid)
    assert support.iloc[0].prior_crashes_1km==2
    assert support.iloc[0].prior_severe_1km==1


def test_holdout_rejected_before_reports_created(tmp_path):
    path=tmp_path/'data.parquet'
    pd.DataFrame({'year':[2025],'crash_datetime':[pd.Timestamp('2025-01-01')],'severe_crash':[0]}).to_parquet(path)
    out=tmp_path/'reports'
    with pytest.raises(ValueError,match='no holdout'):run(path,out)
    assert not out.exists()


def test_regions_are_deterministic_on_boundaries():
    x=pd.DataFrame({'latitude':[33.76,33.759,33.76,33.759],'longitude':[-84.4,-84.4,-84.401,-84.401]})
    assert regions(x).tolist()==['NE','SE','NW','SW']
