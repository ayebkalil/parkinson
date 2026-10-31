"""Audited experiments for seven preserved model families.

Stages: prepare, reproduce, pilot, screen, repeat, select, test, report, verify.
Only reproduce/test read test features for prediction; their outputs are never
used by selection. Configuration and checkpoint hashes are frozen before test.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import xgboost as xgb

import seven_model_protocol as p


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def record_source(root):
    source_files=[*Path('src').glob('*seven_model*.py'),Path('src/test_seven_preprocessing.py'),Path('src/model12_data.py'),
        *[Path('src')/name for name in ['train_model1_xgboost.py','train_model2_lstm.py','train_model3_imaging.py','train_fusion_models.py','evaluate.py']]]
    source_dir=root/'source_snapshot'/'src';source_dir.mkdir(parents=True,exist_ok=True)
    for path in source_files:shutil.copy2(path,source_dir/path.name)
    p.write_json(root/'source_sha256.json',{str(path):p.sha(path) for path in source_files})


def config_for(config,family,name):
    if family.endswith('xgb'):
        trials=config['imaging_xgb_trials' if family=='imaging_xgb' else 'xgb_trials']
        return {**config['xgb_defaults'],**trials[name],'name':name}
    trials=config['imaging_mlp_trials' if family=='imaging_mlp' else 'neural_trials']
    cfg={**config['neural_defaults'],**trials[name],'name':name}
    if name=='baseline':
        if family=='late':cfg['hidden']=48
        if family=='bilstm':cfg.update(hidden=64,dropout=.2,lr=.0015,weight_decay=.0001)
    return cfg


def load(root,split):
    return (pd.read_parquet(root/'data'/f'{split}_longitudinal.parquet'),
            pd.read_parquet(root/'data'/f'{split}_evaluation.parquet'))


def dataset(root,split,family,cfg,prep=None,training=False):
    frame,rows=load(root,split)
    cols=p.sequence.FEATURE_COLS if family=='bilstm' else p.imaging.IMAGING_FEATURES if family=='imaging_mlp' else p.FEATURES
    if prep is None:prep=p.Preprocessor(cols,cfg['clip'],cfg['missing']).fit(frame)
    if training and family!='imaging_mlp':
        rows=frame[frame.target_np3tot_12m.notna()].copy()
        rows['eval12']=rows.target_np3tot_12m.notna();rows['eval24']=rows.target_np3tot_24m.notna()
    return p.Examples(frame,rows,prep,static=family=='imaging_mlp'),prep


def fit_neural(root,family,cfg,seed,config,stage='runs',pilot=False):
    folder=root/stage/family/cfg['name']/str(seed)
    if (folder/'metrics.json').exists():return read(folder/'metrics.json')
    folder.mkdir(parents=True,exist_ok=True)
    p.seed_all(seed,config['threads'])
    train,prep=dataset(root,'train',family,cfg,training=True)
    val,_=dataset(root,'val',family,cfg,prep)
    print(f'DATA {family} {cfg["name"]} train_examples={len(train)} validation_rows={len(val)}',flush=True)
    net=p.build(family,cfg)
    optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=2)
    best=float('inf');best_epoch=0;history=[];start=time.time()
    # No patient balancing changes between reference and candidate: every
    # eligible historical prefix has equal weight; evaluation is per patient.
    for epoch in range(1,(2 if pilot else cfg['epochs'])+1):
        tick=time.time();net.train();loss_sum=0.;count=0
        for x,mask,lengths,cur,y,avail,ids in p.loader(train,cfg['batch'],True,seed+epoch):
            if len(x)==1 and family=='imaging_mlp':
                # BatchNorm needs >=2 training items. Use running statistics;
                # this keeps the final patient instead of dropping its row.
                for m in net.modules():
                    if isinstance(m,nn.BatchNorm1d):m.eval()
            if family=='imaging_mlp':
                flags=train.rows.iloc[ids][['eval12','eval24']].to_numpy(bool)
                y=y.clone();y[:,:2]=y[:,:2].masked_fill(~torch.from_numpy(flags),float('nan'))
                y[:,2]=y[:,2].masked_fill(~torch.from_numpy(flags[:,0]),float('nan'))
            optimizer.zero_grad(set_to_none=True)
            a,b,c,_=net(x,mask,lengths,cur,avail)
            loss=p.masked_loss(a,y[:,0],cfg['loss'],cfg['beta'])+cfg['weight24']*p.masked_loss(b,y[:,1],cfg['loss'],cfg['beta'])
            valid=torch.isfinite(y[:,2])
            if valid.any():loss=loss+cfg['classification_weight']*nn.functional.binary_cross_entropy_with_logits(c[valid],y[valid,2])
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),cfg['clip_grad'],error_if_nonfinite=True);optimizer.step()
            loss_sum+=float(loss.detach())*len(x);count+=len(x)
        pred=p.predict(net,val);m=p.metrics(pred);score=np.mean([m[h]['mae'] for h in ['12m','24m']])
        if score<best-1e-7:
            best=score;best_epoch=epoch
            torch.save(dict(state_dict=p.snapshot_weights(net),preprocessor=prep.state(),
                config=cfg,seed=seed,family=family,epoch=epoch,validation_score=float(score)),folder/'best.pt')
        scheduler.step(score)
        history.append(dict(epoch=epoch,train_loss=loss_sum/count,mae12=m['12m']['mae'],mae24=m['24m']['mae'],
            score=score,seconds=time.time()-tick,lr=optimizer.param_groups[0]['lr']))
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        print(f'{stage} {family} {cfg["name"]} seed={seed} epoch={epoch} valMAE={score:.4f} sec={time.time()-tick:.1f}',flush=True)
        if epoch-best_epoch>=cfg['patience']:break
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    net.load_state_dict(saved['state_dict']);pred=p.predict(net,val)
    pred.to_csv(folder/'validation_predictions.csv',index=False)
    result=dict(family=family,name=cfg['name'],seed=seed,metrics=p.metrics(pred),classification=p.classification(pred),
                seconds=time.time()-start,best_epoch=best_epoch,epochs=len(history),checkpoint=str((folder/'best.pt').relative_to(root)),config=cfg)
    p.write_json(folder/'metrics.json',result);return result


def xgb_predict(model,rows,cols,h):
    result=rows[['PATNO','row_id','visit_date']].copy()
    result['true']=rows[f'target_np3tot_{h}m'].to_numpy()
    result['pred']=model.predict(rows[cols])
    return result


def tabular_rows(root,split,h,cols,locf=False):
    rows=pd.read_parquet(root/'data'/f'{split}_tabular_{h}m.parquet')
    if locf:
        frame=pd.read_parquet(root/'data'/f'{split}_longitudinal.parquet').sort_values(['PATNO','visit_date','row_id'],kind='stable')
        values=frame.groupby('PATNO')[cols].ffill()
        values['row_id']=frame.row_id
        rows.loc[:,cols]=values.set_index('row_id').loc[rows.row_id,cols].to_numpy()
    return rows


def fit_xgb(root,family,cfg,seed,config,stage='runs'):
    results={};cols=p.TABULAR if family=='tabular_xgb' else p.imaging.IMAGING_FEATURES
    for h in (12,24):
        folder=root/stage/family/cfg['name']/str(seed)/f'{h}m'
        if (folder/'metrics.json').exists():results[f'{h}m']=read(folder/'metrics.json');continue
        folder.mkdir(parents=True,exist_ok=True);tick=time.time()
        train=tabular_rows(root,'train',h,cols,cfg.get('locf',False))
        val=tabular_rows(root,'val',h,cols,cfg.get('locf',False))
        params={k:v for k,v in cfg.items() if k not in ['name','locf']}
        model=xgb.XGBRegressor(**params,eval_metric='mae',early_stopping_rounds=50,
            n_jobs=config['threads'],tree_method='hist',random_state=seed)
        model.fit(train[cols],train[f'target_np3tot_{h}m'],eval_set=[(val[cols],val[f'target_np3tot_{h}m'])],verbose=False)
        model.get_booster().set_attr(feature_preprocessing='causal_locf' if cfg.get('locf',False) else 'native_missing')
        model.save_model(folder/'best.json');pred=xgb_predict(model,val,cols,h)
        pred.to_csv(folder/'validation_predictions.csv',index=False)
        result=dict(family=family,name=cfg['name'],seed=seed,horizon=h,metrics=p.regression(pred.true,pred.pred),
            seconds=time.time()-tick,best_iteration=model.best_iteration,checkpoint=str((folder/'best.json').relative_to(root)),config=cfg)
        p.write_json(folder/'metrics.json',result);results[f'{h}m']=result
        print(f'{stage} {family} {cfg["name"]} seed={seed} {h}m MAE={result["metrics"]["mae"]:.4f} sec={time.time()-tick:.1f}',flush=True)
    return results


def gather(root,stage='runs'):
    rows=[]
    for path in (root/stage).rglob('metrics.json'):
        m=read(path)
        if 'horizon' in m:rows.append(dict(family=m['family'],config=m['name'],seed=m['seed'],horizon=m['horizon'],**m['metrics']))
        else:
            for h in (12,24):rows.append(dict(family=m['family'],config=m['name'],seed=m['seed'],horizon=h,**m['metrics'][f'{h}m']))
    df=pd.DataFrame(rows)
    if len(df):df.to_csv(root/f'{stage}_validation_metrics.csv',index=False)
    return df


def shortlist(root,config):
    df=gather(root);decisions={}
    for family in p.FAMILIES:
        subset=df[df.family==family]
        if family.endswith('xgb'):
            decisions[family]={}
            for h in (12,24):
                ranked=subset[subset.horizon==h].groupby('config').mae.mean().sort_values()
                candidates=ranked.drop('baseline',errors='ignore')
                decisions[family][f'{h}m']=candidates.index[0]
        else:
            ranked=subset.groupby('config').mae.mean().sort_values()
            decisions[family]=ranked.drop('baseline',errors='ignore').index[0]
    p.write_json(root/'shortlist.json',dict(challengers=decisions,policy=config['selection']))
    return decisions


def validation_gain_interval(root,family,challenger,seeds,horizons,samples):
    differences=[]
    for h in horizons:
        per_seed=[]
        for seed in seeds:
            frames=[]
            for name in ['baseline',challenger]:
                folder=root/'final'/family/name/str(seed)
                if family.endswith('xgb'):
                    frame=pd.read_csv(folder/f'{h}m/validation_predictions.csv')
                else:
                    frame=pd.read_csv(folder/'validation_predictions.csv')
                    frame=frame[frame[f'eval{h}']].rename(columns={f'true_{h}m':'true',f'pred_{h}m':'pred'})
                frames.append(frame[['PATNO','row_id','visit_date','true','pred']])
            paired=frames[0].merge(frames[1],on=['PATNO','row_id','visit_date'],suffixes=('_a','_b'),validate='one_to_one')
            if len(paired)!=len(frames[0]) or len(paired)!=len(frames[1]) or not np.allclose(paired.true_a,paired.true_b):
                raise ValueError('Unpaired validation predictions')
            paired=paired.sort_values('PATNO')
            per_seed.append(pd.Series((abs(paired.pred_b-paired.true_b)-abs(paired.pred_a-paired.true_a)).to_numpy(),index=paired.PATNO))
        differences.append(pd.concat(per_seed,axis=1).mean(axis=1).rename(f'{h}m'))
    # Same patient is resampled jointly across horizons and all fixed seeds.
    values=pd.concat(differences,axis=1).sort_index().to_numpy()
    rng=np.random.default_rng(161803)
    point=float(np.mean(np.nanmean(values,axis=0)))
    draws=np.array([np.mean(np.nanmean(values[rng.integers(0,len(values),len(values))],axis=0)) for _ in range(samples)])
    return dict(delta_mae=point,ci95_low=float(np.quantile(draws,.025)),ci95_high=float(np.quantile(draws,.975)),
        unit='patient',bootstrap_samples=samples,seeds=seeds,
        note='Development safeguard only; conditional on these seeds, exploratory after validation tuning, no independent validation claim')


def select(root,config):
    record_source(root)
    df=gather(root,'final');challengers=read(root/'shortlist.json')['challengers'];decisions={};manifest=[]
    for family in p.FAMILIES:
        decisions[family]={}
        horizons=(12,24) if family.endswith('xgb') else (0,)
        for h in horizons:
            challenger=challengers[family][f'{h}m'] if h else challengers[family]
            subset=df[(df.family==family)&(df.config.isin(['baseline',challenger]))]
            if h:subset=subset[subset.horizon==h]
            counts=subset.groupby('config').seed.nunique()
            if any(counts.get(k,0)!=len(config['final_seeds']) for k in ('baseline',challenger)):
                raise ValueError('All prespecified seeds are required before selection')
            means=subset.groupby('config').mae.mean()
            interval=validation_gain_interval(root,family,challenger,config['final_seeds'],[h] if h else [12,24],config['bootstrap_samples'])
            passed=means[challenger]<means['baseline'] and interval['ci95_high']<0
            retained=challenger if passed else 'baseline'
            decisions[family][f'{h}m' if h else 'joint']=dict(challenger=challenger,retained=retained,baseline_mae=float(means['baseline']),challenger_mae=float(means[challenger]),validation_paired_interval=interval,passed_validation_safeguard=bool(passed))
        for role in ('baseline','selected'):
            for seed in config['final_seeds']:
                for h in (12,24):
                    name='baseline' if role=='baseline' else decisions[family][f'{h}m' if family.endswith('xgb') else 'joint']['retained']
                    folder=root/'final'/family/name/str(seed)
                    checkpoint=folder/f'{h}m'/'best.json' if family.endswith('xgb') else folder/'best.pt'
                    manifest.append(dict(family=family,role=role,seed=seed,horizon=h,config=name,
                        checkpoint=str(checkpoint.relative_to(root)),sha256=p.sha(checkpoint)))
    p.write_json(root/'selection.json',dict(decisions=decisions,policy=config['selection'],seeds=config['final_seeds'],test_evaluated=False))
    p.write_json(root/'frozen_checkpoints.json',manifest)
    # Compare against already-retained development checkpoints as well. This
    # delivery manifest is separate from the before/after experiment: a reused
    # checkpoint is never presented as a newly achieved gain.
    available=[];prior_file=root/'previously_retained_validation.csv'
    prior=pd.read_csv(prior_file) if prior_file.exists() else pd.DataFrame()
    for family in p.FAMILIES:
        existing=prior[prior.family==family] if len(prior) else pd.DataFrame()
        use_previous_joint=False
        if family=='bilstm' and len(existing):
            chosen=decisions[family]['joint']['retained']
            use_previous_joint=existing.mae.mean()<=df[(df.family==family)&(df.config==chosen)].mae.mean()+1e-9
        for h in (12,24):
            item=next(m for m in manifest if m['family']==family and m['role']=='selected' and m['seed']==42 and m['horizon']==h)
            entry={**item,'checkpoint_base':'experiment','origin':'new_experiment','delivery_seed_policy':'Fixed seed 42, never chosen by test performance'}
            previous=existing[existing.horizon==h] if len(existing) else pd.DataFrame()
            chosen=item['config'];new_mean=df[(df.family==family)&(df.config==chosen)&(df.horizon==h)].mae.mean()
            if len(previous) and (use_previous_joint or family=='tabular_xgb' and previous.mae.mean()<=new_mean+1e-9):
                old=previous[previous.seed==42].iloc[0]
                entry.update(checkpoint=old.checkpoint,checkpoint_base='project',origin='previously_retained',
                    sha256=p.sha(old.checkpoint),config=old.config,test_evaluated_in_this_run=False)
            available.append(entry)
    p.write_json(root/'best_available_checkpoints.json',available)
    p.write_json(root/'freeze.json',dict(selection_sha256=p.sha(root/'selection.json'),manifest_sha256=p.sha(root/'frozen_checkpoints.json'),config_sha256=p.sha(root/'config.json'),source_sha256=p.sha(root/'source_sha256.json'),best_available_sha256=p.sha(root/'best_available_checkpoints.json')))


def test(root,config):
    freeze=read(root/'freeze.json')
    for file,key in [('selection.json','selection_sha256'),('frozen_checkpoints.json','manifest_sha256'),('config.json','config_sha256')]:
        if p.sha(root/file)!=freeze[key]:raise ValueError('Frozen selection/configuration changed')
    for file,expected in read(root/'prepared_sha256.json').items():
        if p.sha(root/file)!=expected:raise ValueError('Prepared dataset changed')
    if p.sha(root/'source_sha256.json')!=freeze['source_sha256']:raise ValueError('Source manifest changed')
    for file,expected in read(root/'source_sha256.json').items():
        if p.sha(file)!=expected:raise ValueError('Training source changed: '+file)
    manifest=read(root/'frozen_checkpoints.json');cache={};summaries=[];class_rows=[]
    for item in manifest:
        family,role,seed,h=item['family'],item['role'],item['seed'],item['horizon']
        path=root/item['checkpoint']
        if p.sha(path)!=item['sha256']:raise ValueError('Checkpoint integrity failure')
        folder=root/'test_predictions'/family/role/str(seed);folder.mkdir(parents=True,exist_ok=True)
        dest=folder/f'{h}m.csv'
        if dest.exists():pred=pd.read_csv(dest)
        elif family.endswith('xgb'):
            net=xgb.XGBRegressor();net.load_model(path)
            cols=p.TABULAR if family=='tabular_xgb' else p.imaging.IMAGING_FEATURES
            rows=tabular_rows(root,'test',h,cols,net.get_booster().attr('feature_preprocessing')=='causal_locf')
            pred=xgb_predict(net,rows,cols,h)
            pred.to_csv(dest,index=False)
        else:
            key=str(path)
            if key not in cache:
                saved=torch.load(path,map_location='cpu',weights_only=False);net=p.build(family,saved['config']);net.load_state_dict(saved['state_dict'])
                ds,_=dataset(root,'test',family,saved['config'],p.Preprocessor.restore(saved['preprocessor']))
                cache[key]=p.predict(net,ds)
                c=p.classification(cache[key]);class_rows.append(dict(family=family,role=role,seed=seed,**c))
            all_pred=cache[key];pred=all_pred[all_pred[f'eval{h}']][['PATNO','row_id','visit_date',f'true_{h}m',f'pred_{h}m']].rename(columns={f'true_{h}m':'true',f'pred_{h}m':'pred'})
            pred.to_csv(dest,index=False)
        m=p.regression(pred.true,pred.pred);summaries.append(dict(split='test',family=family,role=role,config=item['config'],seed=seed,horizon=h,**m))
    pd.DataFrame(summaries).to_csv(root/'test_metrics.csv',index=False)
    if class_rows:pd.DataFrame(class_rows).to_csv(root/'test_classification_separate.csv',index=False)


def report(root,config):
    final=gather(root,'final');selection=read(root/'selection.json');rows=[]
    test_df=pd.read_csv(root/'test_metrics.csv') if (root/'test_metrics.csv').exists() else None
    for split in ('validation','test'):
        if split=='test' and test_df is None:continue
        for family in p.FAMILIES:
            for h in (12,24):
                chosen=selection['decisions'][family][f'{h}m' if family.endswith('xgb') else 'joint']['retained']
                for role,name in [('baseline','baseline'),('selected',chosen)]:
                    subset=final[(final.family==family)&(final.config==name)&(final.horizon==h)] if split=='validation' else test_df[(test_df.family==family)&(test_df.role==role)&(test_df.horizon==h)]
                    row=dict(split=split,family=family,horizon=h,role=role,config=name,seeds=len(subset),n_patients=int(subset.n_patients.iloc[0]))
                    for metric in ['mae','rmse','r2','pearson_r']:
                        row[metric]=float(subset[metric].mean());row[metric+'_std']=float(subset[metric].std(ddof=1))
                    rows.append(row)
    df=pd.DataFrame(rows);comparisons=[]
    for (split,family,h),group in df.groupby(['split','family','horizon'],sort=False):
        a=group[group.role=='baseline'].iloc[0];b=group[group.role=='selected'].iloc[0]
        comparisons.append(dict(split=split,family=family,horizon=h,n_patients=a.n_patients,
            baseline_config=a.config,selected_config=b.config,baseline_mae=a.mae,selected_mae=b.mae,
            delta_mae=b.mae-a.mae,relative_delta_percent=100*(b.mae-a.mae)/a.mae,
            baseline_mae_std=a.mae_std,selected_mae_std=b.mae_std,
            baseline_rmse=a.rmse,selected_rmse=b.rmse,baseline_r2=a.r2,selected_r2=b.r2,
            baseline_pearson_r=a.pearson_r,selected_pearson_r=b.pearson_r))
    df.to_csv(root/'before_after_all_metrics.csv',index=False)
    comp=pd.DataFrame(comparisons);comp.to_csv(root/'before_after_comparison.csv',index=False)
    intervals=[];rng=np.random.default_rng(314159)
    if test_df is not None:
        for family in p.FAMILIES:
            for h in (12,24):
                paired=[]
                for seed in config['final_seeds']:
                    folder=root/'test_predictions'/family
                    a=pd.read_csv(folder/'baseline'/str(seed)/f'{h}m.csv');b=pd.read_csv(folder/'selected'/str(seed)/f'{h}m.csv')
                    keys=['PATNO','row_id','visit_date'];merged=a.merge(b,on=keys,suffixes=('_a','_b'),validate='one_to_one')
                    if len(merged)!=len(a) or len(a)!=len(b) or not np.allclose(merged.true_a,merged.true_b):raise ValueError('Unpaired before/after')
                    merged=merged.sort_values('PATNO');paired.append(abs(merged.pred_b-merged.true_b).to_numpy()-abs(merged.pred_a-merged.true_a).to_numpy())
                per_patient=np.mean(paired,axis=0)
                draws=np.array([np.mean(rng.choice(per_patient,len(per_patient),replace=True)) for _ in range(config['bootstrap_samples'])])
                intervals.append(dict(family=family,horizon=h,n_patients=len(per_patient),delta_mae=float(np.mean(per_patient)),ci95_low=float(np.quantile(draws,.025)),ci95_high=float(np.quantile(draws,.975)),unit='patient',seed_policy='Mean paired error differences over all prespecified seeds; conditional on these seeds, no ensemble predictions'))
        pd.DataFrame(intervals).to_csv(root/'paired_patient_bootstrap.csv',index=False)
    lines=['SEPT FAMILLES — AUDIT ET OPTIMISATION','',
        'Cible : score moteur futur NP3TOT brut, et non variation. Horizons 12 et 24 mois distincts.',
        'Le tableau historique N=462 est exclusivement à 24 mois. Ses 444 cibles de test mal rattachées rendent la comparaison directe invalide.',
        'La référence corrigée et les candidats utilisent les mêmes patients, lignes index exactes et cibles. N=583 test à 12m ; N=462 test à 24m.',
        'Splits PATNO conservés. Préfixes bornés à la ligne index. Médianes/scalers appris sur train uniquement.',
        'Préfixes train équipondérés pour les modèles de séquence ; une observation par patient/horizon pour les modèles statiques et les métriques.',
        'Les deux modèles imagerie gardent uniquement les 11 mesures imagerie et les 2 contrôles normatifs âge/sexe déjà présents ; aucun score moteur ni résidu clinique.',
        'Sélection sur validation ; moyenne des métriques de trois seeds, aucun choix de seed, aucun ensemble de prédictions.',
        'Bootstrap apparié au niveau patient ; il est conditionnel aux trois graines utilisées.',
        'Le test historique a déjà guidé des choix du projet : ces scores ne constituent pas une confirmation indépendante.',
        'data/raw absent : les dates d’acquisition des modalités et l’ordre des examens intra-mois ne sont pas certifiés.',
        'Aucune modification manuelle du registre de benchmarks ou des anciens checkpoints.', '',comp.to_string(index=False), '',
        'Commandes : python src/seven_model_experiments.py --stage STAGE --output '+str(root),
        'STAGE : prepare, reproduce, pilot, screen, repeat, select, test, report, verify.']
    (root/'RAPPORT_FR.txt').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['prepare','reproduce','pilot','screen','shortlist','repeat','select','test','report','verify'])
    parser.add_argument('--output',type=Path,default=Path('experiments/seven_models_20261030'))
    parser.add_argument('--config',type=Path,default=Path('configs/seven_models.json'))
    parser.add_argument('--family',choices=p.FAMILIES);args=parser.parse_args();root=args.output
    if args.stage=='prepare':
        root.mkdir(parents=True,exist_ok=False);config=read(args.config);p.write_json(root/'config.json',config)
        protected={str(path):p.sha(path) for path in [*Path('models').glob('*'),*Path('data/processed').glob('*')] if path.is_file()}
        p.write_json(root/'protected_sha256.json',protected)
        p.prepare(root)
        record_source(root)
        p.write_json(root/'environment.json',dict(python=sys.version,torch=torch.__version__,xgboost=xgb.__version__,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),cuda=torch.cuda.is_available(),threads=config['threads'],optuna='Not installed; fixed structured search'))
        (root/'requirements-lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
        print('PREPARED',root);return
    config=read(root/'config.json');p.seed_all(config['screen_seed'],config['threads'])
    families=[args.family] if args.family else p.FAMILIES
    if args.stage=='reproduce':
        from seven_model_reproduce import reproduce
        reproduce(root);return
    if args.stage in ['pilot','screen']:
        if args.stage=='screen':record_source(root)
        for family in families:
            names=['baseline'] if args.stage=='pilot' else list(config['imaging_xgb_trials' if family=='imaging_xgb' else 'xgb_trials' if family=='tabular_xgb' else 'imaging_mlp_trials' if family=='imaging_mlp' else 'neural_trials'])
            for name in names:
                cfg=config_for(config,family,name)
                if family.endswith('xgb'):fit_xgb(root,family,cfg,config['screen_seed'],config,stage='pilot' if args.stage=='pilot' else 'runs')
                else:fit_neural(root,family,cfg,config['screen_seed'],config,stage='pilot' if args.stage=='pilot' else 'runs',pilot=args.stage=='pilot')
        gather(root,'pilot' if args.stage=='pilot' else 'runs')
    elif args.stage=='shortlist':shortlist(root,config)
    elif args.stage=='repeat':
        choices=shortlist(root,config) if not (root/'shortlist.json').exists() else read(root/'shortlist.json')['challengers']
        for family in families:
            names=set(['baseline',*choices[family].values()]) if family.endswith('xgb') else {'baseline',choices[family]}
            for name in sorted(names):
                for seed in config['final_seeds']:
                    cfg=config_for(config,family,name)
                    screened=root/'runs'/family/name/str(seed)
                    final_dir=root/'final'/family/name/str(seed)
                    if seed==config['screen_seed'] and screened.exists() and not final_dir.exists():
                        shutil.copytree(screened,final_dir)
                    if family.endswith('xgb'):fit_xgb(root,family,cfg,seed,config,'final')
                    else:fit_neural(root,family,cfg,seed,config,'final')
        gather(root,'final')
    elif args.stage=='select':select(root,config)
    elif args.stage=='test':test(root,config)
    elif args.stage=='report':report(root,config)
    elif args.stage=='verify':
        protected=read(root/'protected_sha256.json')
        for path,value in protected.items():
            if p.sha(path)!=value:raise ValueError('Protected historical file changed: '+path)
        for item in read(root/'frozen_checkpoints.json'):
            if p.sha(root/item['checkpoint'])!=item['sha256']:raise ValueError('Checkpoint changed')
        p.write_json(root/'verification.json',dict(protected_files_unchanged=True,checkpoint_hashes_valid=True,verified_files=len(protected)))
        print('VERIFIED',len(protected),'protected files')


if __name__=='__main__':main()
