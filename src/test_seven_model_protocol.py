import unittest
import numpy as np
import pandas as pd
import torch
from torch import nn
import seven_model_protocol as p


CFG=dict(hidden=16,layers=1,dropout=0.,heads=4,mask_absent=True,missing=False)


class CriticalProtocolTest(unittest.TestCase):
    def setUp(self):p.seed_all(42,2)
    def batch(self):
        x=torch.randn(2,5,44);mask=torch.tensor([[1,1,0,0,0],[1,1,1,1,1]],dtype=torch.bool)
        return x,mask,torch.tensor([2,5]),torch.tensor([10.,22.]),torch.ones(2,2)
    def test_padding_invariance(self):
        for family in ['bilstm','early','late','cross']:
            with self.subTest(family=family):
                net=p.build(family,CFG).eval();x,mask,l,cur,avail=self.batch()
                changed=x.clone();changed[0,2:]=999
                a=net(x,mask,l,cur,avail);b=net(changed,mask,l,cur,avail)
                single=net(x[:1,:2],mask[:1,:2],l[:1],cur[:1],avail[:1])
                for i in (0,1):
                    torch.testing.assert_close(a[i],b[i],atol=1e-5,rtol=1e-5)
                    torch.testing.assert_close(a[i][:1],single[i],atol=1e-5,rtol=1e-5)
    def test_residual_added_exactly_once(self):
        for family in ['bilstm','early','late','cross']:
            net=p.build(family,CFG).eval()
            for module in net.modules():
                if isinstance(module,nn.Linear):nn.init.zeros_(module.weight);nn.init.zeros_(module.bias) if module.bias is not None else None
            args=self.batch();a,b,_,_=net(*args)
            torch.testing.assert_close(a,args[3]);torch.testing.assert_close(b,args[3])
    def test_absent_imaging_is_safe(self):
        args=list(self.batch());args[-1]=torch.zeros(2,2)
        for family in ['late','cross']:
            net=p.build(family,CFG).eval();outputs=net(*args)
            self.assertTrue(all(torch.isfinite(x).all() for x in outputs[:3]))
            if family=='late':torch.testing.assert_close(outputs[3],torch.tensor([[1.,0.],[1.,0.]]))
    def test_missing_targets_do_not_poison_gradients(self):
        pred=torch.tensor([1.,2.],requires_grad=True)
        loss=p.masked_loss(pred,torch.tensor([float('nan'),float('nan')]))
        loss.backward();torch.testing.assert_close(pred.grad,torch.zeros(2))
        pred.grad=None;loss=p.masked_loss(pred,torch.tensor([1.,float('nan')]),'mae');loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
    def test_metrics_do_not_hide_bad_predictions(self):
        with self.assertRaises(ValueError):p.regression([1.,2.],[1.,float('nan')])
        result=p.regression([1.,2.,float('nan')],[2.,4.,0.]);self.assertEqual(result['mae'],1.5)
    def test_best_state_is_independent(self):
        net=p.build('late',CFG);snapshot=p.snapshot_weights(net);first=next(iter(snapshot));before=snapshot[first].clone()
        with torch.no_grad():next(net.parameters()).add_(5)
        torch.testing.assert_close(snapshot[first],before)
    def test_imaging_scope(self):
        self.assertEqual(len(p.imaging.IMAGING_FEATURES),13)
        self.assertNotIn('NP3TOT',p.imaging.IMAGING_FEATURES)
        self.assertTrue(all(k.startswith(('datscan_','mri_')) or k in ['age_at_visit','is_male'] for k in p.imaging.IMAGING_FEATURES))
    def test_preprocessing_is_causal_and_train_only(self):
        rows=[]
        for i in range(3):
            row={c:float(i) for c in p.FEATURES};row.update(PATNO=1,row_id=i,visit_date=pd.Timestamp('2020-01-01')+pd.Timedelta(days=180*i));rows.append(row)
        frame=pd.DataFrame(rows);frame.loc[0,p.FEATURES[0]]=np.nan
        prep=p.Preprocessor(p.FEATURES).fit(frame)
        _,a,_=prep.transform(frame);altered=frame.copy();altered.loc[2,p.FEATURES]=999
        _,b,_=prep.transform(altered);np.testing.assert_allclose(a[:2],b[:2])
        state=prep.state();restored=p.Preprocessor.restore(state);_,c,_=restored.transform(frame);np.testing.assert_allclose(a,c)
        np.testing.assert_allclose(prep.scaler.mean_,restored.scaler.mean_)


if __name__=='__main__':unittest.main(verbosity=2)
