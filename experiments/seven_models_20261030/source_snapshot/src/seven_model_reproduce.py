"""Replay existing checkpoints under their historical preprocessing, without training."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import seven_model_protocol as p


def reproduce(root):
    folder=root/'historical_reproduction';folder.mkdir(exist_ok=True)
    train=pd.read_parquet('data/processed/train_longitudinal.parquet')
    train_tab=pd.read_parquet('data/processed/train_tabular_latest.parquet')
    records=[]
    for split in ['val','test']:
        tab=pd.read_parquet(f'data/processed/{split}_tabular_latest.parquet')
        longitudinal=pd.read_parquet(f'data/processed/{split}_longitudinal.parquet')
        for family in p.FAMILIES:
            if family.endswith('xgb'):
                cols=p.TABULAR if family=='tabular_xgb' else p.imaging.IMAGING_FEATURES
                outputs=[]
                for h in (12,24):
                    name=f'model1_xgboost_{h}m.json' if family=='tabular_xgb' else f'model3_imaging_xgboost_{h}m.json'
                    net=xgb.XGBRegressor();net.load_model(Path('models')/name);outputs.append(net.predict(tab[cols]))
            elif family=='imaging_mlp':
                imputer=SimpleImputer(strategy='median');scaler=StandardScaler()
                scaler.fit(imputer.fit_transform(train_tab[p.imaging.IMAGING_FEATURES]))
                x=scaler.transform(imputer.transform(tab[p.imaging.IMAGING_FEATURES]))
                net=p.imaging.ImagingMLP(13,64,.25);net.load_state_dict(torch.load('models/model3_imaging_mlp.pt',weights_only=True,map_location='cpu'));net.eval()
                with torch.inference_mode():a,b,_=net(torch.tensor(x,dtype=torch.float32))
                outputs=[a.numpy(),b.numpy()]
            else:
                old=p.sequence if family=='bilstm' else p.fusion
                cols=p.sequence.FEATURE_COLS if family=='bilstm' else p.FEATURES
                medians=train[cols].median().fillna(0)
                # Fusion checkpoints used raw-visit fit; BiLSTM main uses LOCF fit.
                fit=train[cols] if family!='bilstm' else train.groupby('PATNO')[cols].ffill()
                scaler=StandardScaler().fit(fit.fillna(medians))
                scaled=old.build_trajectories(longitudinal,cols,scaler,medians)
                if family=='bilstm':
                    net=old.LongitudinalBiLSTM(44,64,2,.2)
                    checkpoint=torch.load('models/model2_bilstm.pt',weights_only=False,map_location='cpu')
                    net.load_state_dict(checkpoint.get('model_state_dict',checkpoint))
                    values=old.extract_labeled_sequences(scaled,cols,'latest_only',tab)
                    ds=old.LongitudinalSequenceDataset(*values);collate=old.collate_sequences
                else:
                    net={'early':old.EarlyFusionModel,'late':old.LateFusionModel,'cross':old.CrossAttentionFusionModel}[family]()
                    name={'early':'early_fusion.pt','late':'late_fusion.pt','cross':'cross-attention_transformer.pt'}[family]
                    net.load_state_dict(torch.load(Path('models')/name,weights_only=True,map_location='cpu'))
                    values=old.extract_labeled_sequences(scaled,cols,'latest_only',tab)
                    ds=old.SequenceDataset(*values);collate=old.pad_collate_fn
                net.eval();aa=[];bb=[]
                with torch.inference_mode():
                    for x,mask,lengths,cur,*_ in DataLoader(ds,batch_size=64,collate_fn=collate):
                        a,b,_,_=net(x,mask,lengths,cur);aa.extend(a.numpy());bb.extend(b.numpy())
                outputs=[np.array(aa),np.array(bb)]
            predictions=tab[['PATNO','visit_date','target_np3tot_12m','target_np3tot_24m']].copy()
            for h,values in zip([12,24],outputs):
                predictions[f'pred_{h}m']=values
                m=p.regression(tab[f'target_np3tot_{h}m'],values)
                records.append(dict(split=split,family=family,horizon=h,**m))
            predictions.to_csv(folder/f'{split}_{family}.csv',index=False)
            print('REPRODUCED',split,family,flush=True)
    results=pd.DataFrame(records);results.to_csv(root/'historical_reproduced_metrics.csv',index=False)
    ledger=pd.read_csv('report/benchmark_results.csv');mapping={
        'tabular_xgb':'Model 1: Tabular XGBoost','bilstm':'Model 2: Longitudinal BiLSTM',
        'late':'Fusion: Late Fusion','early':'Fusion: Early Fusion','cross':'Fusion: Cross-Attention Transformer',
        'imaging_mlp':'Model 3: Monomodal Neuroimaging (Deep MLP)','imaging_xgb':'Model 3: Monomodal Neuroimaging (XGBoost)'}
    comparison=[]
    for row in results[results.split=='test'].itertuples():
        reference=ledger[(ledger.model_name==mapping[row.family])&ledger.target_horizon.str.startswith(f'+{row.horizon} Months')].iloc[0]
        comparison.append(dict(family=row.family,horizon=row.horizon,n_patients=row.n_patients,
            reported_mae=float(reference.mae),reproduced_mae=row.mae,mae_difference=row.mae-reference.mae,
            reported_rmse=float(reference.rmse),reproduced_rmse=row.rmse))
    pd.DataFrame(comparison).to_csv(root/'historical_reproduction_comparison.csv',index=False)
