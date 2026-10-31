"""Integration checks against real project artifacts, without changing datasets."""
import math
import time
import unittest

from fastapi.testclient import TestClient

from backend.main import app
from backend.service import service


class ResearchApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        assert cls.client.get('/api/health').json()['status'] == 'ok'
        assert cls.client.get('/api/patients').status_code == 503
        started = cls.client.post('/api/initialize')
        assert started.status_code == 202
        start_time = started.json()['started_at']
        assert cls.client.post('/api/initialize').json()['started_at'] == start_time
        deadline = time.monotonic() + 120
        progress = []
        while time.monotonic() < deadline:
            state = cls.client.get('/api/status').json()
            progress.append(state['progress'])
            if state['state'] in ('ready', 'error'):
                break
            time.sleep(.2)
        assert state['state'] == 'ready', state
        assert progress == sorted(progress), progress
        assert state['loaded_models'] == ['late_fusion', 'bilstm', 'xgboost']

    def test_patients_filters_and_errors(self):
        rows = self.client.get('/api/patients').json()
        self.assertEqual(rows['total'], 583)
        self.assertEqual(self.client.get('/api/patients?search=3054').json()['items'][0]['id'], 3054)
        rapid = self.client.get('/api/patients?group=rapid').json()['items']
        self.assertTrue(rapid)
        self.assertTrue(all(p['rapid_progression'] for p in rapid))
        self.assertTrue(all(p['visits'] >= 4 for p in self.client.get('/api/patients?group=history').json()['items']))
        self.assertEqual(self.client.get('/api/patients?group=unknown').status_code, 422)
        self.assertEqual(self.client.get('/api/patients/99999999').status_code, 404)
        self.assertEqual(self.client.get('/api/patients/not-an-id').status_code, 422)

    def test_prediction_parity_with_existing_dashboard(self):
        # Reuse the established dashboard's prediction path with the same loaded
        # resources, so this detects serving/serialization and index alignment drift.
        import app as dashboard
        artifacts = {'test_tabular': service.tabular, 'test_longitudinal': service.longitudinal,
            'df_all': service.all_sequences, 'df_lstm': service.lstm_sequences,
            'late_model': service.late_model, 'bilstm_model': service.bilstm_model,
            'retained_xgboost': service.xgb}
        expected = dashboard.predict_patient(3054, artifacts)
        response = self.client.get('/api/patients/3054')
        self.assertEqual(response.status_code, 200)
        result = response.json()
        for model, suffix in [('late_fusion', 'lf'), ('bilstm', 'bi')]:
            for horizon in (12, 24):
                self.assertAlmostEqual(result['predictions'][model][f'{horizon}m']['score'], expected[f'pred{horizon}_{suffix}'], places=5)
        self.assertAlmostEqual(result['gating']['clinical'] + result['gating']['imaging'], 100, places=3)
        self.assertAlmostEqual(sum(v['attention'] for v in result['history']), 100, places=3)
        self.assertTrue(all(v['date'] <= result['patient']['last_visit'] for v in result['history']))
        for horizon in ('12m', '24m'):
            actual = result['predictions']['xgboost'][horizon]
            reference = expected['xgb_predictions'][horizon]
            self.assertEqual(actual is None, reference is None)
            if actual:
                self.assertAlmostEqual(actual['score'], reference['score'], places=5)
                self.assertEqual(actual['index_date'], reference['index_date'].strftime('%Y-%m-%d'))
        self.assertIsNone(result['predictions']['xgboost']['probability'])

    def test_missing_horizon_and_export(self):
        _, eligible, _ = service.xgb['24m']
        missing = next(p['id'] for p in service.patient_index if p['id'] not in eligible.index)
        payload = self.client.get(f'/api/patients/{missing}').json()
        self.assertIsNone(payload['predictions']['xgboost']['24m'])
        export = self.client.get(f'/api/patients/{missing}/export')
        self.assertEqual(export.status_code, 200)
        self.assertIn('text/csv', export.headers['content-type'])
        self.assertEqual(len(export.text.lstrip('\ufeff').splitlines()), 7)
        self.assertEqual(self.client.get('/api/patients/99999999/export').status_code, 404)

    def test_overview_and_benchmarks(self):
        overview = self.client.get('/api/overview').json()
        self.assertEqual(overview['patients'], 583)
        self.assertGreater(overview['visits'], 583)
        self.assertTrue(math.isfinite(overview['mean_score']))
        bench = self.client.get('/api/benchmarks?protocol=historical').json()
        self.assertTrue(bench['historical'])
        self.assertEqual(len(bench['items']), len(service.benchmarks))


if __name__ == '__main__':
    unittest.main(verbosity=2)
