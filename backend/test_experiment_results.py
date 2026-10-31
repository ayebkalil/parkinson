from pathlib import Path
import tempfile
import unittest
from backend.experiment_results import audited_benchmarks
from backend.service import ROOT


class ExperimentResultsTest(unittest.TestCase):
    def test_missing_results_do_not_fabricate_a_benchmark(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(audited_benchmarks(Path(folder)))
    def test_completed_results_match_computed_files(self):
        result=audited_benchmarks(ROOT)
        if result is None:self.skipTest('Experiments have not completed verification yet')
        import pandas as pd
        frame=pd.read_csv(ROOT/'experiments/seven_models_20261030/before_after_all_metrics.csv')
        self.assertEqual(len(result['items']),14)
        self.assertFalse(result['historical'])
        for item in result['items']:
            horizon=int(item['target_horizon'].split()[0][1:])
            self.assertEqual(item['n_samples'],583 if horizon==12 else 462)
            self.assertEqual(item['n_seeds'],3)
            self.assertAlmostEqual(item['mae']-item['baseline_mae'],item['delta_mae'],places=9)
        actual=sorted(item['mae'] for item in result['items'])
        expected=sorted(frame[(frame.split=='test')&(frame.role=='selected')].mae)
        self.assertEqual(actual,expected)


if __name__=='__main__':unittest.main(verbosity=2)
