"""Paired chronological and buffered geographic validation on 2021–2024 only."""
from __future__ import annotations
import argparse, hashlib, json
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from threadpoolctl import threadpool_limits
from src.models.compare_features import validate_development_data, alert_budget_metrics
from src.models.compare_imbalance import FEATURES, make_xgb_pipeline
from src.models.time_validation import VALIDATION_FOLDS, make_logistic_pipeline
from src.models.evaluate import classification_metrics
R = 6371008.8
LAT_SPLIT, LON_SPLIT = 33.76, -84.40
REGIONS = ['NE','NW','SE','SW']


def coordinates(frame):
    a=frame[['latitude','longitude']].to_numpy(dtype=float)
    if not np.isfinite(a).all() or (np.abs(a[:,0])>90).any() or (np.abs(a[:,1])>180).any():
        raise ValueError('Finite valid coordinates required.')
    return np.radians(a)


def regions(frame):
    return pd.Series(np.where(frame.latitude.ge(LAT_SPLIT),'N','S'),index=frame.index)+pd.Series(np.where(frame.longitude.ge(LON_SPLIT),'E','W'),index=frame.index)


def buffered_training(train, valid, region, buffer_m=500):
    """Exclude the whole held region plus a 510m approximate boundary halo.
    Verify/refine using true Haversine distances to validation coordinates.
    Validation labels are never accessed for selecting training rows.
    """
    if region not in REGIONS or buffer_m<0: raise ValueError('Invalid region or buffer.')
    east=np.radians(train.longitude-LON_SPLIT)*R*np.cos(np.radians(LAT_SPLIT))
    north=np.radians(train.latitude-LAT_SPLIT)*R
    margin=buffer_m+10
    near_e=east.ge(-margin) if region[1]=='E' else east.le(margin)
    near_n=north.ge(-margin) if region[0]=='N' else north.le(margin)
    kept=train.loc[~(near_e&near_n)].copy()
    if kept.empty or valid.empty: raise ValueError('Empty spatial training/validation block.')
    # Exact guard handles projection approximation near the boundaries.
    distances=BallTree(coordinates(valid),metric='haversine').query(coordinates(kept),k=1)[0][:,0]*R
    kept=kept.loc[distances>buffer_m].copy()
    if kept.empty: raise ValueError('Buffer excluded all training rows.')
    nearest=BallTree(coordinates(kept),metric='haversine').query(coordinates(valid),k=1)[0][:,0]*R
    assert (nearest>buffer_m).all()
    assert not regions(kept).eq(region).any()
    return kept,nearest


def local_support(train, valid, radius_m=1000):
    indexes=BallTree(coordinates(train),metric='haversine').query_radius(coordinates(valid),r=radius_m/R)
    labels=train.severe_crash.to_numpy()
    return pd.DataFrame({'prior_crashes_1km':[len(i) for i in indexes],
        'prior_severe_1km':[int(labels[i].sum()) for i in indexes]},index=valid.index)


def run(train_path='data/processed/train.parquet', output_dir='reports/metrics/geographic_validation', buffer_m=500):
    out=Path(output_dir)
    if out.exists(): raise FileExistsError('Use a fresh output directory.')
    path=Path(train_path);frame=pd.read_parquet(path);validate_development_data(frame);coordinates(frame)
    before=pd.util.hash_pandas_object(frame,index=True)
    out.mkdir(parents=True)
    assigned=regions(frame)
    pd.DataFrame({'year':frame.year,'region':assigned,'severe':frame.severe_crash}).groupby(['year','region']).severe.agg(['size','sum','mean']).to_csv(out/'regional_counts.csv')
    factories={'logistic_unweighted':lambda x:make_logistic_pipeline(x,weighted=False),'xgboost_unweighted':lambda x:make_xgb_pipeline(x,1.)}
    metrics=[];budgets=[];bins=[];audits=[];predictions=[]
    for years,year in VALIDATION_FOLDS:
        prior=frame[frame.year.isin(years)]
        future=frame[frame.year.eq(year)]
        if prior.crash_datetime.max()>=future.crash_datetime.min():raise ValueError('Chronology violated.')
        historical=local_support(prior,future)
        baseline={}
        for name,factory in factories.items():
            with threadpool_limits(limits=2):
                model=factory(prior[FEATURES]);model.fit(prior[FEATURES],prior.severe_crash)
                baseline[name]=pd.Series(model.predict_proba(future[FEATURES])[:,1],index=future.index)
        for region in REGIONS:
            valid=future.loc[assigned.reindex(future.index).eq(region)]
            kept,nearest=buffered_training(prior,valid,region,buffer_m)
            if kept.severe_crash.nunique()!=2 or valid.severe_crash.nunique()!=2:
                raise ValueError('Each train/validation block requires both classes; do not silently skip.')
            support=local_support(kept,valid)
            audits.append(dict(validation_year=year,region=region,chronological_train_rows=len(prior),spatial_train_rows=len(kept),
                spatial_train_severe=int(kept.severe_crash.sum()),validation_rows=len(valid),validation_severe=int(valid.severe_crash.sum()),
                minimum_train_validation_distance_m=float(nearest.min()),median_nearest_train_m=float(np.median(nearest)),
                fraction_under_5_severe_full_history=float(historical.loc[valid.index].prior_severe_1km.lt(5).mean()),
                fraction_under_5_severe_spatial_training=float(support.prior_severe_1km.lt(5).mean())))
            for name,factory in factories.items():
                print(year,region,name,flush=True)
                with threadpool_limits(limits=2):
                    model=factory(kept[FEATURES]);model.fit(kept[FEATURES],kept.severe_crash)
                    spatial=model.predict_proba(valid[FEATURES])[:,1]
                for setting,probs,t in [('chronological',baseline[name].loc[valid.index].to_numpy(),prior),('buffered_spatial',spatial,kept)]:
                    key=dict(validation_year=year,region=region,model=name,setting=setting)
                    m=classification_metrics(valid.severe_crash,probs)
                    row={**key,**{k:v for k,v in m.items() if k not in ['calibration_curve','confusion_matrix']},
                        'rows':len(valid),'severe':int(valid.severe_crash.sum()),'mean_probability':float(np.mean(probs)),
                        'dummy_brier':float(np.mean((valid.severe_crash-t.severe_crash.mean())**2))}
                    row.update(dict(zip(['tn','fp','fn','tp'],np.asarray(m['confusion_matrix']).ravel().astype(int))))
                    metrics.append(row)
                    bins.extend({**key,**b} for b in m['calibration_curve'])
                    for frac in [.05,.1,.2]:budgets.append({**key,**alert_budget_metrics(valid.severe_crash,probs,frac)})
                    predictions.append(pd.DataFrame({**key,'source_row':valid.index,'label':valid.severe_crash.to_numpy(),'probability':probs,
                        'prior_crashes_1km':historical.loc[valid.index].prior_crashes_1km.to_numpy(),
                        'prior_severe_1km':historical.loc[valid.index].prior_severe_1km.to_numpy()}))
            pd.DataFrame(metrics).to_csv(out/'regional_metrics.csv',index=False)
            pd.DataFrame(budgets).to_csv(out/'regional_alert_budgets.csv',index=False)
            pd.DataFrame(audits).to_csv(out/'split_audit.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'calibration_bins.csv',index=False)
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_parquet(out/'validation_predictions.parquet',index=False)
    # Pools each validation row exactly once per year/model/setting across regions.
    pooled=[];pooled_budget=[];support_metrics=[]
    for key,p in pred.groupby(['validation_year','model','setting']):
        meta=dict(zip(['validation_year','model','setting'],key))
        assert p.source_row.is_unique
        m=classification_metrics(p.label,p.probability.to_numpy())
        pooled.append({**meta,**{k:v for k,v in m.items() if k not in ['calibration_curve','confusion_matrix']},'mean_probability':p.probability.mean()})
        for frac in [.05,.1,.2]:pooled_budget.append({**meta,**alert_budget_metrics(p.label,p.probability.to_numpy(),frac)})
        for label,mask in [('under_5_prior_severe',p.prior_severe_1km.lt(5)),('at_least_5_prior_severe',p.prior_severe_1km.ge(5))]:
            q=p[mask]
            if q.label.nunique()!=2:continue
            m=classification_metrics(q.label,q.probability.to_numpy())
            support_metrics.append({**meta,'support_group':label,'rows':len(q),'severe':int(q.label.sum()),
                **{k:v for k,v in m.items() if k not in ['calibration_curve','confusion_matrix']}})
    pd.DataFrame(pooled).to_csv(out/'pooled_year_metrics.csv',index=False)
    pd.DataFrame(pooled_budget).to_csv(out/'pooled_alert_budgets.csv',index=False)
    pd.DataFrame(support_metrics).to_csv(out/'historical_support_metrics.csv',index=False)
    summary=pd.DataFrame(metrics).groupby(['model','setting'])[['average_precision','brier_score','actual_positive_rate','mean_probability']].mean()
    summary.to_csv(out/'macro_region_year_summary.csv')
    pd.testing.assert_series_equal(before,pd.util.hash_pandas_object(frame,index=True))
    (out/'metadata.json').write_text(json.dumps(dict(source=str(path.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        latitude_split=LAT_SPLIT,longitude_split=LON_SPLIT,buffer_m=buffer_m,features=FEATURES,folds=VALIDATION_FOLDS,
        holdout_read=False,resampling=False,models_replaced=False,
        versions={name:version(name) for name in ["numpy","pandas","scikit-learn","xgboost"]},
        notes=['Four fixed contiguous quadrants; region definitions do not use labels.',
            'Spatial training loses both nearby observations and overall sample size; differences are not a pure causal geography effect.',
            'Regional metrics compare identical future records under both settings.',
            'Pooled spatial metrics mix predictions from four models; regional metrics are primary.',
            'Support strata use full prior-year history, never future outcomes; five cases is a descriptive flag, not a reliability guarantee.']),indent=2))
    print(summary.to_string(),flush=True)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-path',default='data/processed/train.parquet')
    parser.add_argument('--output-dir',default='reports/metrics/geographic_validation')
    parser.add_argument('--buffer-m',type=float,default=500)
    args=parser.parse_args();run(args.train_path,args.output_dir,args.buffer_m)
