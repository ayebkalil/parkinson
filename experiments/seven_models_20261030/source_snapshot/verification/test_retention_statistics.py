"""Numerical invariants for the validation-only patient bootstrap safeguard."""
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
from seven_model_experiments import validation_gain_interval


class ValidationBootstrapTest(unittest.TestCase):
    def frame(self,shift=0.):
        return pd.DataFrame(dict(PATNO=[1,2,3,4],row_id=[10,11,12,13],
            visit_date=['2020-01-01']*4,eval12=[True]*4,eval24=[True]*4,
            true_12m=[10.]*4,true_24m=[20.]*4,
            pred_12m=[11.+shift,12.+shift,13.+shift,14.+shift],
            pred_24m=[21.+shift,22.+shift,23.+shift,24.+shift]))
    def interval(self,shift):
        def read(path):return self.frame(0. if 'baseline' in Path(path).parts else shift)
        with patch('seven_model_experiments.pd.read_csv',side_effect=read):
            return validation_gain_interval(Path('.'),'bilstm','candidate',[42,17,2026],[12,24],100)
    def test_identical_predictions_cannot_pass_safeguard(self):
        result=self.interval(0.)
        self.assertEqual(result['delta_mae'],0.)
        self.assertEqual(result['ci95_high'],0.)
    def test_constant_paired_gain_is_recovered(self):
        result=self.interval(-1.)
        self.assertAlmostEqual(result['delta_mae'],-1.)
        self.assertAlmostEqual(result['ci95_low'],-1.)
        self.assertAlmostEqual(result['ci95_high'],-1.)
    def test_target_mismatch_is_rejected(self):
        def read(path):
            frame=self.frame()
            if 'candidate' in Path(path).parts:frame.loc[0,'true_12m']=99.
            return frame
        with patch('seven_model_experiments.pd.read_csv',side_effect=read):
            with self.assertRaises(ValueError):
                validation_gain_interval(Path('.'),'bilstm','candidate',[42,17,2026],[12],100)


if __name__=='__main__':unittest.main(verbosity=2)
