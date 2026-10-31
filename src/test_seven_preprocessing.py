import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pathlib import Path
from seven_model_experiments import tabular_rows


class CausalTabularTest(unittest.TestCase):
    def test_locf_never_reads_future_or_another_patient(self):
        frame=pd.DataFrame(dict(PATNO=[1,1,1,2],row_id=[17,18,19,20],
            visit_date=pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01','2020-01-01']),
            datscan_caudate_sbr=[2.5,np.nan,999.,np.nan],target_np3tot_12m=[1.,2.,3.,4.]))
        rows=frame.iloc[[1,3]].copy().reset_index(drop=True)
        def parquet(path):return (frame if 'longitudinal' in str(path) else rows).copy()
        with patch('seven_model_experiments.pd.read_parquet',side_effect=parquet):
            native=tabular_rows(Path('.'),'val',12,['datscan_caudate_sbr'],False)
            causal=tabular_rows(Path('.'),'val',12,['datscan_caudate_sbr'],True)
        self.assertTrue(native.datscan_caudate_sbr.isna().all())
        self.assertEqual(causal.datscan_caudate_sbr.iloc[0],2.5)
        self.assertTrue(np.isnan(causal.datscan_caudate_sbr.iloc[1]))
        pd.testing.assert_frame_equal(causal.drop(columns='datscan_caudate_sbr'),rows.drop(columns='datscan_caudate_sbr'))


if __name__=='__main__':unittest.main(verbosity=2)
