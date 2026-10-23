"""Regression checks for leakage, residual units, padding and missing labels."""
import unittest
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import train_model2_lstm as old
from model12_data import FEATURES, STATIC_FEATURES, Preprocessor, make_dataset, latest_rows, reconstruct_targets
from model12_experiments import (model, signature, masked_reg, masked_bce,
                                 choose_threshold, classification, read_json)


def frame():
    data = pd.DataFrame({c: [1., 2., 100.] for c in FEATURES})
    data["NP3TOT"] = [10., 15., 30.]
    data["PATNO"] = [1, 1, 1]
    data["visit_date"] = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
    data["row_id"] = [0, 1, 2]
    data["target_np3tot_12m"] = [15., 30., np.nan]
    data["target_np3tot_24m"] = [30., np.nan, np.nan]
    data["prog_mcid_12m"] = [1., 1., np.nan]
    data["prog_mcid_24m"] = [1., np.nan, np.nan]
    return data


class IntegrityTests(unittest.TestCase):
    def test_original_feature_lists_are_unchanged(self):
        tree = ast.parse(Path('src/train_model1_xgboost.py').read_text(encoding='utf-8'))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'FEATURE_COLS' for t in node.targets))
        self.assertEqual(STATIC_FEATURES, ast.literal_eval(assignment.value))
        self.assertEqual(FEATURES, old.FEATURE_COLS)

    def test_rows_preserve_missing_24m(self):
        df = frame()
        self.assertTrue(np.isnan(latest_rows(df, 12).target_np3tot_24m.iloc[0]))
        self.assertEqual(latest_rows(df, 24).row_id.iloc[0], 0)
        self.assertEqual(df.dropna(subset=["target_np3tot_12m"]).groupby("PATNO").last().target_np3tot_24m.iloc[0], 30.)

    def test_targets_and_labels_reconstructed(self):
        out = reconstruct_targets(frame())
        self.assertEqual(out.target_date_24m.iloc[0], pd.Timestamp("2022-01-01"))
        broken = frame()
        broken.loc[1, "target_np3tot_24m"] = 30.
        with self.assertRaises(ValueError):
            reconstruct_targets(broken)

    def test_raw_anchor_and_future_prefix_invariance(self):
        df = frame()
        prep = Preprocessor().fit(df)
        state = prep.state()
        ds = make_dataset(df, prep, df.iloc[[1]])
        self.assertEqual(ds.cur_scores, [15.])
        self.assertNotEqual(float(ds.sequences[0][-1, 0]), 15.)
        future = df.copy()
        future.loc[2, FEATURES] = 99999.
        later = make_dataset(future, prep, df.iloc[[1]])
        np.testing.assert_array_equal(ds.sequences[0], later.sequences[0])
        self.assertEqual(prep.state(), state)
        restored = Preprocessor.restore(state)
        np.testing.assert_array_equal(prep.transform(df)[1], restored.transform(df)[1])

    def test_padding_attention_and_architecture(self):
        torch.manual_seed(42)
        cfg = read_json("configs/model12_experiments.json")
        original = old.LongitudinalBiLSTM(len(FEATURES), 64, 2, .2)
        for trial in cfg["lstm_trials"]:
            self.assertEqual(signature(original), signature(model(dict(cfg["lstm_defaults"], **trial))))
        original.eval()
        x = torch.randn(2, 6, len(FEATURES))
        lengths = torch.tensor([2, 6])
        mask = torch.arange(6)[None, :] < lengths[:, None]
        cur = torch.tensor([17., 40.])
        with torch.no_grad():
            batch = original(x, mask, lengths, cur)
            single = original(x[:1, :2], mask[:1, :2], lengths[:1], cur[:1])
            polluted = x.clone()
            polluted[0, 2:] = 10000
            changed = original(polluted, mask, lengths, cur)
        for i in range(3):
            torch.testing.assert_close(batch[i][0], single[i][0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(batch[i], changed[i])
        self.assertEqual(float(batch[3][0, 2:].sum()), 0.)
        # Zero residual heads: output must equal raw NP3TOT exactly.
        for head in [original.head_delta_12m, original.head_delta_24m]:
            for param in head.parameters():
                param.data.zero_()
        with torch.no_grad():
            a, b, _, _ = original(x, mask, lengths, cur)
        torch.testing.assert_close(a, cur)
        torch.testing.assert_close(b, cur)

    def test_missing_targets_do_not_generate_nan_gradients(self):
        for loss in ["huber", "mae"]:
            p = torch.tensor([1., 2.], requires_grad=True)
            t = torch.tensor([float("nan"), float("nan")])
            value = masked_reg(p, t, {"loss": loss, "beta": 1.}) + masked_bce(p, t, torch.tensor(2.))
            value.backward()
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertEqual(float(value.detach()), 0.)

    def test_threshold_recovers_discrimination_below_point_five(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([.1, .2, .3, .4])
        self.assertEqual(classification(y, p)["sensitivity"], 0.)
        threshold, feasible = choose_threshold(y, p)
        self.assertTrue(feasible)
        metrics = classification(y, p, threshold)
        self.assertEqual(metrics["balanced_accuracy"], 1.)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])


if __name__ == "__main__":
    unittest.main()
