"""Common, row-preserving protocol for seven existing model families.

No source Parquet or historical checkpoint is overwritten. Inference/evaluation
is at the latest *complete eligible row* per patient and horizon. Model choice
uses development data only. Run seven_model_experiments.py for experiments.
"""
from __future__ import annotations
import copy
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'parkinson-matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import train_model2_lstm as sequence
import train_fusion_models as fusion
import train_model3_imaging as imaging
from train_model1_xgboost import FEATURE_COLS as TABULAR
from model12_data import write_json, sha, prepare

FAMILIES = ['tabular_xgb', 'bilstm', 'late', 'early', 'cross', 'imaging_mlp', 'imaging_xgb']
FEATURES = fusion.ALL_FEATURE_COLS
TARGETS = ['target_np3tot_12m', 'target_np3tot_24m', 'prog_mcid_12m']


def seed_all(seed, threads=2):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def regression(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError('Prediction/target shape mismatch')
    valid = np.isfinite(y)
    if not np.isfinite(p[valid]).all():
        raise ValueError('Nonfinite predictions must not remove difficult patients')
    y, p = y[valid], p[valid]
    return dict(n_patients=len(y), mae=float(mean_absolute_error(y,p)),
        rmse=float(np.sqrt(mean_squared_error(y,p))), r2=float(r2_score(y,p)),
        pearson_r=float(np.corrcoef(y,p)[0,1]) if np.std(y)>0 and np.std(p)>0 else None,
        bias=float(np.mean(p-y)))


def metrics(pred):
    return {f'{h}m': regression(d[f'true_{h}m'], d[f'pred_{h}m'])
            for h in (12,24) for d in [pred[pred[f'eval{h}']]]}


def classification(pred):
    d=pred[pred.eval12 & pred.true_mcid.notna()]
    y,p=d.true_mcid.to_numpy(),d.prob_mcid.to_numpy()
    if not np.isfinite(p).all(): raise ValueError('Nonfinite probabilities')
    return {'n_patients':len(y),'roc_auc':float(roc_auc_score(y,p)),
            'average_precision':float(average_precision_score(y,p)),
            'threshold':.5, 'accuracy':float(np.mean((p>=.5)==y))}


class Preprocessor:
    def __init__(self, cols, clip=None, missing=False):
        self.cols=list(cols); self.clip=clip; self.missing=missing
    def fit(self, frame):
        frame=frame.sort_values(['PATNO','visit_date','row_id'],kind='stable')
        self.medians=frame[self.cols].median().fillna(0)
        filled=frame.groupby('PATNO')[self.cols].ffill().fillna(self.medians)
        self.scaler=StandardScaler().fit(filled)
        return self
    def transform(self, frame):
        df=frame.sort_values(['PATNO','visit_date','row_id'],kind='stable').reset_index(drop=True)
        observed=df[self.cols].notna().to_numpy(np.float32)
        filled=df.groupby('PATNO')[self.cols].ffill()
        values=self.scaler.transform(filled.fillna(self.medians)).astype(np.float32)
        if self.clip: values=np.clip(values,-self.clip,self.clip)
        if self.missing: values=np.column_stack([values,1-observed])
        # Availability follows LOCF, before median imputation, and is not fed as biology.
        all_filled=df.groupby('PATNO')[fusion.IMAGING_COLS].ffill()
        availability=np.column_stack([all_filled.iloc[:,:5].notna().any(axis=1),
            all_filled.iloc[:,5:].notna().any(axis=1)]).astype(np.float32)
        if not np.isfinite(values).all(): raise ValueError('Nonfinite input')
        return df,values,availability
    def state(self):
        return dict(cols=self.cols,clip=self.clip,missing=self.missing,
            medians=self.medians.to_dict(),mean=self.scaler.mean_.tolist(),scale=self.scaler.scale_.tolist())
    @classmethod
    def restore(cls,s):
        obj=cls(s['cols'],s['clip'],s['missing']);obj.medians=pd.Series(s['medians'])
        obj.scaler=StandardScaler();obj.scaler.mean_=np.array(s['mean']);obj.scaler.scale_=np.array(s['scale'])
        obj.scaler.var_=obj.scaler.scale_**2;obj.scaler.n_features_in_=len(obj.cols)
        obj.scaler.feature_names_in_=np.array(obj.cols,dtype=object)
        return obj


class Examples(Dataset):
    def __init__(self, frame, rows, prep, static=False):
        self.frame,self.values,self.avail=prep.transform(frame)
        self.rows=rows.reset_index(drop=True);self.static=static
        self.positions={rid:i for i,rid in enumerate(self.frame.row_id)}
        self.starts={p:g.index[0] for p,g in self.frame.groupby('PATNO',sort=False)}
        self.ends=[self.positions[r.row_id] for r in self.rows.itertuples()]
        self.begins=[self.starts[pat] for pat in self.rows.PATNO]
        self.current=self.rows.NP3TOT.to_numpy(np.float32,copy=True)
        self.targets=self.rows[TARGETS].to_numpy(np.float32,copy=True)
        endpoints=self.frame.iloc[self.ends]
        if not np.array_equal(endpoints.PATNO.to_numpy(),self.rows.PATNO.to_numpy()):
            raise ValueError('Exact patient alignment failed')
        if not np.allclose(endpoints.NP3TOT,self.rows.NP3TOT):
            raise ValueError('Raw residual anchor alignment failed')
        if not np.array_equal(endpoints.visit_date.to_numpy(),self.rows.visit_date.to_numpy()):
            raise ValueError('Exact index date alignment failed')
        if (self.frame.groupby('PATNO').visit_date.diff().dropna()<pd.Timedelta(0)).any():
            raise ValueError('Nonchronological prefix')
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        end=self.ends[i]
        values=self.values[end:end+1] if self.static else self.values[self.begins[i]:end+1]
        return (torch.from_numpy(values),torch.tensor(self.current[i]),
            torch.from_numpy(self.targets[i]),torch.from_numpy(self.avail[end]),i)


def collate(batch):
    x,cur,y,avail,ids=zip(*batch)
    lengths=torch.tensor([len(a) for a in x])
    values=nn.utils.rnn.pad_sequence(x,batch_first=True)
    mask=torch.arange(values.size(1))[None,:]<lengths[:,None]
    return values,mask,lengths,torch.stack(cur),torch.stack(y),torch.stack(avail),list(ids)


def loader(ds,batch=128,shuffle=False,seed=42):
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),num_workers=0)


class Early(fusion.EarlyFusionModel):
    def __init__(self,cfg):
        h=cfg['hidden'];super().__init__(44, h,cfg['dropout'])
        if cfg['layers']!=2:
            self.bilstm=nn.LSTM(h,h,cfg['layers'],batch_first=True,bidirectional=True,
                dropout=cfg['dropout'] if cfg['layers']>1 else 0)
    def forward(self,x,mask,lengths,cur,avail):
        return super().forward(x,mask,lengths,cur)


class Late(fusion.LateFusionModel):
    def __init__(self,cfg):
        h=cfg['hidden'];super().__init__(hidden_dim=h,dropout=cfg['dropout'])
        if cfg['layers']!=2:
            self.clin_lstm=nn.LSTM(h,h,cfg['layers'],batch_first=True,bidirectional=True,
                dropout=cfg['dropout'] if cfg['layers']>1 else 0)
        self.mask_absent=cfg.get('mask_absent',True)
    def forward(self,x,mask,lengths,cur,avail):
        clinical=torch.cat([x[:,:,:30],x[:,:,41:]],-1)
        last=x[torch.arange(len(x)),lengths-1]
        out,_=self.clin_lstm(nn.utils.rnn.pack_padded_sequence(self.clin_proj(clinical),lengths.cpu(),batch_first=True,enforce_sorted=False))
        out,_=nn.utils.rnn.pad_packed_sequence(out,batch_first=True,total_length=x.size(1))
        hc,_=self.clin_attn(out,mask)
        hi=self.img_encoder(torch.cat([last[:,30:41],last[:,41:]],-1))
        joint=torch.cat([hc,hi],-1);gates=self.gate_net(joint)
        if self.mask_absent: gates=torch.where(avail.any(1)[:,None],gates,gates.new_tensor([1.,0.]))
        a=torch.stack([self.clin_head_12(hc).squeeze(-1),self.img_head_12(hi).squeeze(-1)],1)
        b=torch.stack([self.clin_head_24(hc).squeeze(-1),self.img_head_24(hi).squeeze(-1)],1)
        self.branch_outputs=(cur[:,None]+a,cur[:,None]+b)
        return cur+(a*gates).sum(1),cur+(b*gates).sum(1),self.head_cls(joint).squeeze(-1),gates


class Cross(fusion.CrossAttentionFusionModel):
    def __init__(self,cfg):
        h=cfg['hidden'];super().__init__(hidden_dim=h,num_heads=cfg.get('heads',4),dropout=cfg['dropout'])
        if cfg['layers']!=2:
            self.clin_lstm=nn.LSTM(h,h//2,cfg['layers'],batch_first=True,bidirectional=True,
                dropout=cfg['dropout'] if cfg['layers']>1 else 0)
        self.mask_absent=cfg.get('mask_absent',True)
    def forward(self,x,mask,lengths,cur,avail):
        clinical=torch.cat([x[:,:,:30],x[:,:,41:]],-1)
        packed=nn.utils.rnn.pack_padded_sequence(self.clin_proj(clinical),lengths.cpu(),batch_first=True,enforce_sorted=False)
        q,_=self.clin_lstm(packed)
        q,_=nn.utils.rnn.pad_packed_sequence(q,batch_first=True,total_length=x.size(1))
        last=x[torch.arange(len(x)),lengths-1]
        tokens=torch.stack([self.token_datscan(last[:,30:35]),self.token_mri(last[:,35:41]),self.token_demo(last[:,41:])],1)
        # The demographic token is always available, including when both scan blocks are absent.
        missing=torch.column_stack([~avail.bool(),torch.zeros(len(x),dtype=torch.bool)])
        a,weights=self.cross_attn(q,tokens,tokens,key_padding_mask=missing if self.mask_absent else None)
        context,_=self.temporal_attn(self.norm_cross(q+a),mask)
        return cur+self.head_12m(context).squeeze(-1),cur+self.head_24m(context).squeeze(-1),self.head_cls(context).squeeze(-1),weights


class BiLSTM(sequence.LongitudinalBiLSTM):
    def __init__(self,cfg):
        super().__init__(44,cfg['hidden'],cfg['layers'],cfg['dropout'])
    def forward(self,x,mask,lengths,cur,avail): return super().forward(x,mask,lengths,cur)


class Imaging(imaging.ImagingMLP):
    def __init__(self,cfg):
        super().__init__(13*(2 if cfg.get('missing',False) else 1),cfg['hidden'],cfg['dropout'])
    def forward(self,x,mask,lengths,cur,avail):
        a,b,c=super().forward(x[:,0]);return a,b,c,None


def build(family,cfg):
    return {'bilstm':BiLSTM,'early':Early,'late':Late,'cross':Cross,'imaging_mlp':Imaging}[family](cfg)


def masked_loss(pred,target,kind='huber',beta=1.):
    valid=torch.isfinite(target)
    if not valid.any(): return pred.sum()*0
    if kind=='mae': return (pred[valid]-target[valid]).abs().mean()
    if kind=='mse': return ((pred[valid]-target[valid])**2).mean()
    return nn.functional.smooth_l1_loss(pred[valid],target[valid],beta=beta)


def predict(net,ds,batch=128):
    net.eval();parts=[]
    with torch.inference_mode():
        for x,mask,lengths,cur,y,avail,ids in loader(ds,batch):
            a,b,c,extra=net(x,mask,lengths,cur,avail)
            rows=ds.rows.iloc[ids][['PATNO','row_id','visit_date','eval12','eval24']].copy()
            rows['pred_12m']=a.numpy();rows['pred_24m']=b.numpy();rows['prob_mcid']=c.sigmoid().numpy()
            for i,h in enumerate((12,24)):rows[f'true_{h}m']=y[:,i].numpy()
            rows['true_mcid']=y[:,2].numpy()
            if isinstance(net,Late):
                rows['weight_clinical']=extra[:,0].numpy();rows['weight_imaging']=extra[:,1].numpy()
                rows['imaging_available']=avail.any(1).numpy()
            parts.append(rows)
    result=pd.concat(parts,ignore_index=True)
    if result.row_id.tolist()!=ds.rows.row_id.tolist():raise ValueError('Prediction order mismatch')
    for h in (12,24):
        if not np.allclose(result[f'true_{h}m'],ds.rows[f'target_np3tot_{h}m'],equal_nan=True):raise ValueError('Target mismatch')
    return result


def snapshot_weights(net):
    return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
