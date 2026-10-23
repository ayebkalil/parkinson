"""Focused regression checks; synthetic fixtures are never performance results."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

import fusion_experiments as f


class FusionChecks(unittest.TestCase):
    def setUp(self):
        f.seed_all(42, 2)
        plan = json.loads(Path("configs/fusion_experiments.json").read_text())
        self.cfg = {**plan["defaults"], "dropout": 0.0}

    def batch(self):
        x = torch.randn(2, 5, 45)
        x[..., -1] = 1
        x[0, 2:] = 0
        lengths = torch.tensor([2, 5])
        mask = torch.arange(5)[None] < lengths[:, None]
        return x, mask, lengths, torch.tensor([20., 30.])

    def fixture(self):
        df = pd.DataFrame({c: [1., 2., 99.] for c in f.FEATURES})
        df["PATNO"] = 1
        df["visit_date"] = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
        df["NP3TOT"] = [10., 20., 90.]
        for c in f.TARGETS:
            df[c] = [25., 35., np.nan] if c != "prog_mcid_12m" else [1., 1., np.nan]
        return df

    def test_raw_anchor_and_prefix_exclude_future(self):
        df = self.fixture()
        prep = f.Preprocessor().fit(df)
        tab = df.iloc[[1]]
        ds = f.make_dataset(df, prep, tab)
        self.assertEqual(ds.cur_scores[0].item(), 20.)
        self.assertEqual(len(ds.sequences[0]), 2)
        mutated = df.copy()
        mutated.loc[2, f.FEATURES] = 99999
        later = f.make_dataset(mutated, prep, tab)
        np.testing.assert_array_equal(ds.sequences[0], later.sequences[0])

    def test_train_only_imputation_and_availability(self):
        train = self.fixture()
        val = train.copy()
        val[f.old.IMAGING_COLS] = np.nan
        prep = f.Preprocessor(missing=True).fit(train)
        before = copy.deepcopy(prep.state())
        _, values = prep.transform(val)
        self.assertEqual(before, prep.state())
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values[:, -1] == 0).all())
        self.assertTrue((values[:, 74:85] == 1).all())

    def test_no_cross_patient_forward_fill(self):
        train = self.fixture()
        val = train.iloc[:2].copy()
        val["PATNO"] = [1, 2]
        val.loc[1, f.old.IMAGING_COLS] = np.nan
        _, values = f.Preprocessor().fit(train).transform(val)
        self.assertEqual(values[1, -1], 0)

    def test_padding_invariance_both_models(self):
        batch = self.batch()
        for arch in ["early", "late"]:
            model = f.build_model(arch, self.cfg).eval()
            with torch.no_grad():
                together = model(*batch)
                alone = model(batch[0][:1, :2], batch[1][:1, :2], batch[2][:1], batch[3][:1])
            for h in [0, 1]:
                torch.testing.assert_close(together[h][:1], alone[h], atol=2e-5, rtol=1e-5)

    def test_absent_imaging_uses_clinical_branch(self):
        x, mask, lengths, cur = self.batch()
        x[0, :, -1] = 0
        model = f.LateFusion(self.cfg).eval()
        p12, p24, _, gates = model(x, mask, lengths, cur)
        torch.testing.assert_close(gates[0], torch.tensor([1., 0.]))
        torch.testing.assert_close(p12[0], model.branch_outputs[0][0, 0])
        torch.testing.assert_close(p24[0], model.branch_outputs[1][0, 0])

    def test_branch_and_gate_gradients(self):
        model = f.LateFusion(self.cfg)
        a, b, _, gates = model(*self.batch())
        (a.square().mean() + b.square().mean()).backward()
        for prefix in ["clin_proj", "clin_lstm", "clin_head_12", "img_encoder", "img_head_12", "gate_net"]:
            norm = sum(p.grad.abs().sum().item() for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)
            self.assertGreater(norm, 0, prefix)
        torch.testing.assert_close(gates.sum(1), torch.ones(2))

    def test_missing_targets_zero_gradient(self):
        for kind in ["mae", "huber"]:
            pred = torch.tensor([1., 8.], requires_grad=True)
            loss = f.masked_loss(pred, torch.tensor([2., float("nan")]), kind)
            loss.backward()
            self.assertEqual(pred.grad[1].item(), 0.)
            empty = f.masked_loss(pred, torch.full((2,), float("nan")), kind)
            self.assertEqual(empty.item(), 0.)

    def test_checkpoint_snapshot_does_not_change(self):
        model = f.EarlyFusion(self.cfg)
        snapshot = copy.deepcopy(model.state_dict())
        name = next(iter(snapshot))
        expected = snapshot[name].clone()
        with torch.no_grad():
            next(model.parameters()).add_(10)
        torch.testing.assert_close(snapshot[name], expected)
        self.assertFalse(torch.equal(model.state_dict()[name], expected))

    def test_preprocessor_roundtrip(self):
        frame = self.fixture()
        prep = f.Preprocessor(missing=True).fit(frame)
        restored = f.Preprocessor.restore(prep.state())
        np.testing.assert_array_equal(prep.transform(frame)[1], restored.transform(frame)[1])

    def test_zero_residual_heads_return_np3tot_points(self):
        for arch in ["early", "late"]:
            model = f.build_model(arch, self.cfg)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if ("head_12" in name or "head_24" in name):
                        p.zero_()
            batch = self.batch()
            a, b, _, _ = model(*batch)
            torch.testing.assert_close(a, batch[3])
            torch.testing.assert_close(b, batch[3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
