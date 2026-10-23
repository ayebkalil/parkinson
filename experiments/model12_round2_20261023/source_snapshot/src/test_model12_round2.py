"""Round-two checks: patient isolation, preprocessing isolation and whole model."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import model12_experiments as e
import model12_round2 as r
from model12_data import Preprocessor
from test_model12_experiments import IntegrityTests, frame


class InternalValidationTests(unittest.TestCase):
    def make_cohort(self):
        rows = []
        for i in range(24):
            patient = frame()
            patient["PATNO"] = i
            patient["row_id"] += 3*i
            patient["prog_mcid_12m"] = patient.prog_mcid_12m * (i % 2)
            rows.append(patient)
        return pd.concat(rows, ignore_index=True)

    def test_patients_partitioned_once_and_preprocessing_not_pooled(self):
        cohort = self.make_cohort()
        assignments = r.fold_assignments(cohort, 3, 17)
        pd.testing.assert_frame_equal(assignments, r.fold_assignments(cohort, 3, 17))
        self.assertFalse(assignments.PATNO.duplicated().any())
        held = set(assignments.loc[assignments.fold == 0, "PATNO"])
        train, val = cohort[~cohort.PATNO.isin(held)].copy(), cohort[cohort.PATNO.isin(held)].copy()
        val.loc[:, "NP3TOT"] = 10000.
        prep = Preprocessor().fit(train)
        state = prep.state()
        prep.transform(val)
        self.assertEqual(prep.state(), state)
        self.assertLess(prep.scaler.mean_[0], 100.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            r.write_data(root, train, val)
            self.assertEqual(len(list((root / "data").glob("*.parquet"))), 8)
            self.assertFalse(list(root.rglob("test*.parquet")))
            tr = e.load_frame(root, "train")
            va = e.load_frame(root, "val")
            self.assertFalse(set(tr.PATNO) & set(va.PATNO))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                r.write_data(Path(directory), train, train)

    def test_round_two_preserves_every_architecture_parameter_shape(self):
        config = e.read_json("configs/model12_round2.json")
        signature = e.signature(e.model(config["lstm_defaults"]))
        self.assertEqual(signature["count"], 243651)
        for cfg in r.configs_lstm(config):
            self.assertEqual(signature, e.signature(e.model(cfg)))
        for task in e.TASKS:
            for cfg in r.configs_xgb(config, task):
                self.assertEqual(cfg["objective"], "binary:logistic" if task == "cls" else "reg:squarederror")

    def test_sensitivity_gain_cannot_override_balanced_accuracy_loss(self):
        base = dict(roc_auc=.7, average_precision=.5, balanced_accuracy=.67,
                    sensitivity=.65, specificity=.69)
        poor = dict(base, roc_auc=.72, average_precision=.52, sensitivity=.9,
                    specificity=.4, balanced_accuracy=.65)
        self.assertFalse(r.better_task(poor, base, "cls"))
        good = dict(base, roc_auc=.72, average_precision=.52, sensitivity=.7,
                    specificity=.7, balanced_accuracy=.7)
        self.assertTrue(r.better_task(good, base, "cls"))
        self.assertFalse(r.better_task({"mae":4.51}, {"mae":4.5}, "12m"))


if __name__ == "__main__":
    unittest.main()
