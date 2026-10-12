import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.preprocess import build_preprocessor, normalize_geographic_categories
from src.models.time_validation import make_logistic_pipeline


def data():
    return pd.DataFrame({'hour':[1.,2.,3.,4.,5.,6.,7.,8.],
        'geolocated_city':['Atlanta']*8,
        'geolocated_county':['FULTON','Fulton','DEKALB','DeKalb']*2})


def test_spelling_variants_produce_identical_encoded_inputs():
    frame=data();pre=build_preprocessor(frame,dense_output=True).fit(frame)
    a=frame.iloc[[0]].copy();b=a.assign(geolocated_county=' fUlToN ',geolocated_city=' ATLANTA ')
    np.testing.assert_array_equal(pre.transform(a),pre.transform(b))
    names=pre.get_feature_names_out()
    assert len([n for n in names if 'county' in n])==2


def test_normalization_preserves_missing_values_other_fields_and_source():
    frame=data();frame.loc[0,'geolocated_county']=None
    frame['weather_conditions_crash_level']='Clear'
    before=frame.copy(deep=True);normalized=normalize_geographic_categories(frame)
    assert pd.isna(normalized.loc[0,'geolocated_county'])
    assert normalized.weather_conditions_crash_level.eq('Clear').all()
    pd.testing.assert_frame_equal(frame,before)


def test_saved_pipeline_applies_same_rule_at_inference(tmp_path):
    frame=data();model=make_logistic_pipeline(frame,weighted=True)
    model.fit(frame,[0,1,0,1,1,0,1,0])
    path=tmp_path/'model.joblib';joblib.dump(model,path);loaded=joblib.load(path)
    variant=frame.assign(geolocated_city=' ATLANTA ',geolocated_county=frame.geolocated_county.str.swapcase())
    np.testing.assert_allclose(loaded.predict_proba(frame),loaded.predict_proba(variant),rtol=0,atol=0)


def test_new_geography_and_non_geo_feature_subsets_still_work():
    frame=data();pre=build_preprocessor(frame).fit(frame)
    assert pre.transform(frame.iloc[[0]].assign(geolocated_county='Unseen')).shape[0]==1
    other=pd.DataFrame({'weather_conditions_crash_level':['Clear','Rain']})
    pre=build_preprocessor(other).fit(other)
    assert pre.transform(other).shape==(2,2)
