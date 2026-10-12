"""Hugging Face / local loading for the final severity pipeline. No training."""
from pathlib import Path
import argparse,json,os
import joblib
import numpy as np
import pandas as pd

FEATURES=['hour','day_of_week','month','is_weekend','is_rush_hour','is_night','latitude','longitude','geolocated_city','geolocated_county']
FILENAME='final_severity_logistic_2021_2024.joblib'

def load_model(repo_id=None,filename=FILENAME,revision=None,local_path=None):
    # The trusted local src.models.preprocess module must remain importable.
    if local_path is not None:
        path=Path(local_path)
    else:
        repo_id=repo_id or os.getenv('HF_MODEL_REPO')
        if not repo_id or 'YOUR_' in repo_id:
            raise ValueError('Set HF_MODEL_REPO=username/repository or provide --repo-id.')
        from huggingface_hub import hf_hub_download
        path=hf_hub_download(repo_id=repo_id,filename=filename,revision=revision or os.getenv('HF_MODEL_REVISION','main'),token=os.getenv('HF_TOKEN') or None)
    model=joblib.load(path)
    if list(model.feature_names_in_) != FEATURES or list(model.classes_) != [0,1]:
        raise ValueError('Model schema/classes do not match this severity pipeline.')
    return model

def make_features(records):
    frame=pd.DataFrame(records).copy()
    needed=['datetime','latitude','longitude','geolocated_city','geolocated_county']
    missing=set(needed)-set(frame.columns)
    if missing or frame.empty:raise ValueError(f'Nonempty records with {needed} required; missing {sorted(missing)}')
    # Require explicit local wall-clock strings, avoiding ambiguous UTC conversion.
    dates=pd.to_datetime(frame['datetime'],errors='raise')
    if dates.dt.tz is not None:raise ValueError('Supply Atlanta local wall-clock datetimes without a timezone offset.')
    if dates.isna().any():raise ValueError('Missing datetime')
    for name,limit in [('latitude',90),('longitude',180)]:
        frame[name]=pd.to_numeric(frame[name],errors='raise')
        if not np.isfinite(frame[name]).all() or not frame[name].between(-limit,limit).all():raise ValueError(f'Invalid {name}')
    for name in needed[-2:]:
        if not frame[name].map(lambda v:isinstance(v,str) and bool(v.strip())).all():raise ValueError(f'Nonempty {name} required')
    frame['hour']=dates.dt.hour;frame['day_of_week']=dates.dt.dayofweek;frame['month']=dates.dt.month
    frame['is_weekend']=frame.day_of_week.ge(5).astype(int)
    frame['is_rush_hour']=(frame.hour.between(7,9)|frame.hour.between(16,18)).astype(int)
    frame['is_night']=(frame.hour.ge(20)|frame.hour.lt(6)).astype(int)
    return frame[FEATURES].copy()

def predict_records(model,records):
    X=make_features(records)
    probabilities=model.predict_proba(X)[:,list(model.classes_).index(1)]
    # No hard severity label: a decision threshold has not been validated.
    return [{'latitude':float(lat),'longitude':float(lon),'riskProbability':float(p)} for lat,lon,p in zip(X.latitude,X.longitude,probabilities)]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id');parser.add_argument('--filename',default=FILENAME)
    parser.add_argument('--revision');parser.add_argument('--local-model');parser.add_argument('--input',required=True)
    args=parser.parse_args()
    model=load_model(args.repo_id,args.filename,args.revision,args.local_model)
    records=json.loads(Path(args.input).read_text())
    print(json.dumps(predict_records(model,records),indent=2,allow_nan=False))
if __name__=='__main__':main()
