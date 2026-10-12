import sys,types
from pathlib import Path
import numpy as np
import pytest
from src.models.hf_inference import make_features,load_model,predict_records,FILENAME
ROOT=Path(__file__).resolve().parents[1]
RECORD={'datetime':'2024-06-03 08:00:00','latitude':33.76,'longitude':-84.39,'geolocated_city':'Atlanta','geolocated_county':'Fulton'}
def test_feature_contract():
 x=make_features([RECORD]);assert x.hour[0]==8 and x.is_rush_hour[0]==1 and x.is_night[0]==0 and x.day_of_week[0]==0
 assert make_features([{**RECORD,'datetime':'2024-06-08 23:00:00'}]).is_weekend[0]==1
 with pytest.raises(ValueError):make_features([{**RECORD,'latitude':float('nan')}])
 with pytest.raises(ValueError):make_features([{**RECORD,'datetime':'2024-06-03T08:00:00Z'}])
def test_local_model_and_download_scaffold(monkeypatch):
 path=ROOT/'models'/FILENAME
 local=load_model(local_path=path);calls=[]
 def download(**kwargs):calls.append(kwargs);return str(path)
 monkeypatch.setitem(sys.modules,'huggingface_hub',types.SimpleNamespace(hf_hub_download=download))
 model=load_model('test-owner/test-model',revision='test-revision')
 assert calls[0]['repo_id']=='test-owner/test-model' and calls[0]['revision']=='test-revision'
 points=predict_records(model,[RECORD]);assert 0<=points[0]['riskProbability']<=1
 assert np.allclose(local.predict_proba(make_features([RECORD]))[:,1],[points[0]['riskProbability']])
 assert RECORD['datetime']=='2024-06-03 08:00:00'
def test_missing_repo(monkeypatch):
 monkeypatch.delenv('HF_MODEL_REPO',raising=False)
 with pytest.raises(ValueError,match='HF_MODEL_REPO'):load_model()

@pytest.mark.parametrize('hour,rush,night',[(0,0,1),(5,0,1),(6,0,0),(7,1,0),(9,1,0),(10,0,0),(16,1,0),(18,1,0),(19,0,0),(20,0,1),(23,0,1)])
def test_time_boundaries(hour,rush,night):
 row=make_features([{**RECORD,'datetime':f'2024-06-03 {hour:02}:00:00'}]).iloc[0]
 assert row.is_rush_hour==rush and row.is_night==night

@pytest.mark.parametrize('change',[{'longitude':181},{'latitude':-91},{'geolocated_city':''},{'geolocated_county':None},{'datetime':None}])
def test_invalid_requests(change):
 with pytest.raises(ValueError):make_features([{**RECORD,**change}])

def test_batch_order_and_input_unchanged():
 import copy
 records=[RECORD.copy(),{**RECORD,'latitude':33.78,'datetime':'2024-06-08 23:00:00'}]
 before=copy.deepcopy(records)
 model=load_model(local_path=ROOT/'models'/FILENAME)
 result=predict_records(model,records)
 assert [x['latitude'] for x in result]==[33.76,33.78]
 assert records==before
 assert all(0<=x['riskProbability']<=1 for x in result)

def test_unexpected_model_schema_rejected(monkeypatch):
 import src.models.hf_inference as inference
 monkeypatch.setattr(inference.joblib,'load',lambda path:types.SimpleNamespace(feature_names_in_=['wrong'],classes_=[0,1]))
 with pytest.raises(ValueError,match='schema'):load_model(local_path='unused.joblib')
