"""Development-only ablations and audits; never changes final model selection.

Run after the primary batch. These exploratory analyses have their own files,
do not read test predictions and do not replace any selected checkpoint.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import seven_model_protocol as p
from seven_model_experiments import config_for,fit_xgb,fit_neural,load,dataset,read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('experiments/seven_models_20261030'))
    parser.add_argument('--stage',choices=['quality','ablations','prior'],required=True);args=parser.parse_args()
    root=args.output;config=read(root/'config.json');p.seed_all(42,config['threads'])
    if args.stage=='quality':
        records=[];time_audit=[]
        for split in ['train','val']:
            frame,_=load(root,split)
            ordered=frame.sort_values(['PATNO','visit_date','row_id'],kind='stable')
            expected_delta=ordered.groupby('PATNO').visit_date.diff().dt.days.fillna(0)
            expected_elapsed=(ordered.visit_date-ordered.groupby('PATNO').visit_date.transform('min')).dt.days
            checks=dict(split=split,rows=len(frame),patients=int(frame.PATNO.nunique()),
                negative_intervals=int((frame.delta_t_days<0).sum()),
                delta_day_mismatches=int((~np.isclose(ordered.delta_t_days,expected_delta)).sum()),
                elapsed_month_mismatches=int((~np.isclose(ordered.elapsed_months_bl,expected_elapsed/30.4375)).sum()),
                duplicate_patient_dates=int(frame.duplicated(['PATNO','visit_date']).sum()))
            if any(checks[key] for key in ['negative_intervals','delta_day_mismatches','elapsed_month_mismatches']):
                raise ValueError('Invalid time features: '+str(checks))
            time_audit.append(checks)
            for h in (12,24):
                rows=pd.read_parquet(root/'data'/f'{split}_tabular_{h}m.parquet')
                for cols,name in [(p.fusion.IMAGING_COLS,'imaging'),(p.fusion.CLINICAL_COLS,'clinical')]:
                    records.append(dict(split=split,horizon=h,block=name,n_patients=len(rows),
                        all_missing=int(rows[cols].isna().all(axis=1).sum()),missing_fraction=float(rows[cols].isna().mean().mean())))
        pd.DataFrame(records).to_csv(root/'development_modality_quality.csv',index=False)
        p.write_json(root/'development_temporal_audit.json',time_audit)
    elif args.stage=='ablations':
        records=[]
        for family in ['tabular_xgb','imaging_xgb']:
            base=config_for(config,family,'baseline')
            for objective,name in [('reg:absoluteerror','absolute_loss_only'),('reg:pseudohubererror','huber_loss_only')]:
                cfg={**base,'objective':objective,'name':name}
                if objective=='reg:pseudohubererror':cfg['huber_slope']=3.
                results=fit_xgb(root,family,cfg,42,config,'ablations')
                for h,result in results.items():records.append(dict(family=family,ablation=name,horizon=h,seed=42,**result['metrics']))
        for change,name in [({'missing':True},'missing_indicators_only'),({'loss':'mae'},'absolute_loss_only')]:
            cfg={**config_for(config,'imaging_mlp','baseline'),**change,'name':name}
            result=fit_neural(root,'imaging_mlp',cfg,42,config,'ablations')
            for h,m in result['metrics'].items():records.append(dict(family='imaging_mlp',ablation=name,horizon=h,seed=42,**m))
        # Check the contribution of availability masks with fixed learned weights.
        # This is an inference ablation, not an estimate of retraining without masks.
        deadline=time.monotonic()+1200
        while not (root/'selection.json').exists():
            if time.monotonic()>deadline:raise RuntimeError('Run the primary batch through selection before branch diagnostics')
            time.sleep(2)
        for family in ['late','cross']:
            selection=read(root/'selection.json')['decisions'][family]['joint']['retained']
            path=root/'final'/family/selection/'42'/'best.pt'
            saved=torch.load(path,map_location='cpu',weights_only=False)
            ds,_=dataset(root,'val',family,saved['config'],p.Preprocessor.restore(saved['preprocessor']))
            for enabled in [True,False]:
                net=p.build(family,saved['config']);net.load_state_dict(saved['state_dict']);net.mask_absent=enabled
                pred=p.predict(net,ds,batch=len(ds) if family=='late' else 128)
                for h,m in p.metrics(pred).items():records.append(dict(family=family,ablation='fixed_weights_mask_'+str(enabled),horizon=h,seed=42,**m))
                if family=='late' and enabled:
                    for j,h in enumerate([12,24]):
                        for branch,k in [('clinical',0),('imaging',1)]:
                            pred[f'{branch}_{h}m']=net.branch_outputs[j][:,k].detach().numpy()
                            eligible=pred[pred[f'eval{h}']]
                            records.append(dict(family=family,ablation='fixed_weights_branch_'+branch,horizon=f'{h}m',seed=42,
                                **p.regression(eligible[f'true_{h}m'],eligible[f'{branch}_{h}m'])))
                    pred.to_csv(root/'late_branches_validation.csv',index=False)
        pd.DataFrame(records).to_csv(root/'development_ablations.csv',index=False)
    elif args.stage=='prior':
        # Keep a trace of previously retained models on the same corrected
        # validation rows. They are not overwritten or selected by test scores.
        manifest=read(Path('experiments/model12_round2_20261023/retained_checkpoints.json'))
        records=[]
        for item in manifest:
            if item['task']=='cls':continue
            if item['model']=='xgboost':
                h=int(item['task'][:-1]);rows=pd.read_parquet(root/'data'/f'val_tabular_{h}m.parquet')
                net=xgb.XGBRegressor();net.load_model(item['checkpoint'])
                m=p.regression(rows[f'target_np3tot_{h}m'],net.predict(rows[p.TABULAR]))
                records.append(dict(family='tabular_xgb',seed=item['seed'],horizon=h,config=item['config'],checkpoint=item['checkpoint'],**m))
            else:
                saved=torch.load(item['checkpoint'],map_location='cpu',weights_only=False)
                state=saved['preprocessor'];prep=p.Preprocessor.restore(dict(cols=state['feature_cols'],clip=None,missing=False,medians=state['medians'],mean=state['mean'],scale=state['scale']))
                cfg=dict(hidden=64,layers=2,dropout=saved['config']['dropout'],clip=None,missing=False)
                ds,_=dataset(root,'val','bilstm',cfg,prep);net=p.build('bilstm',cfg);net.load_state_dict(saved['model_state_dict']);pred=p.predict(net,ds)
                for h,m in p.metrics(pred).items():records.append(dict(family='bilstm',seed=item['seed'],horizon=int(h[:-1]),config=item['config'],checkpoint=item['checkpoint'],**m))
        pd.DataFrame(records).to_csv(root/'previously_retained_validation.csv',index=False)


if __name__=='__main__':main()
