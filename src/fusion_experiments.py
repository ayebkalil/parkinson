"""Isolated, validation-only experiments preserving the team's fusion networks.

The historical module and all its checkpoints remain untouched. Run from repo root.
No test features or targets are loaded by this module (PATNO only for split audit).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Matplotlib is imported by the historical module, even for numerical runs.
# Keep its cache in the writable temp directory on managed Windows machines.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "parkinson-matplotlib"))
import train_fusion_models as old
from evaluate import evaluate_regression, evaluate_classification

FEATURES = old.ALL_FEATURE_COLS
TARGETS = ["target_np3tot_12m", "target_np3tot_24m", "prog_mcid_12m"]
DATA = Path("data/processed")


def write_json(path, value):
    # Strict JSON, including undefined metrics on small subgroups.
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (np.integer, np.floating)):
            v = v.item()
        return None if isinstance(v, float) and not np.isfinite(v) else v
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def ordered(df):
    return df.sort_values(["PATNO", "visit_date"], kind="stable").reset_index(drop=True)


class Preprocessor:
    """Train-only medians/scaling; causal LOCF; optional missingness indicators."""
    def __init__(self, missing=False, legacy=False):
        self.missing = missing
        self.legacy = legacy

    def fit(self, train):
        train = ordered(train)
        self.medians = train[FEATURES].median().fillna(0.0)
        values = train[FEATURES] if self.legacy else train.groupby("PATNO")[FEATURES].ffill()
        self.scaler = StandardScaler().fit(values.fillna(self.medians))
        return self

    def transform(self, frame):
        df = ordered(frame)
        observed = df[FEATURES].notna().astype(np.float32).to_numpy()
        filled = df.groupby("PATNO")[FEATURES].ffill()
        available = filled[old.IMAGING_COLS].notna().any(axis=1).to_numpy(np.float32)
        values = self.scaler.transform(filled.fillna(self.medians)).astype(np.float32)
        if self.missing:
            values = np.concatenate([values, 1.0 - observed], axis=1)
        # Availability is bookkeeping, not a new biological modality.
        values = np.column_stack([values, available]).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite model inputs")
        return df, values

    def state(self):
        return dict(missing=self.missing, legacy=self.legacy,
                    feature_cols=FEATURES, medians=self.medians.to_dict(),
                    mean=self.scaler.mean_.tolist(), scale=self.scaler.scale_.tolist())

    @classmethod
    def restore(cls, state):
        if state["feature_cols"] != FEATURES:
            raise ValueError("Feature order differs from checkpoint")
        obj = cls(state["missing"], state["legacy"])
        obj.medians = pd.Series(state["medians"])
        obj.scaler = StandardScaler()
        obj.scaler.mean_ = np.asarray(state["mean"])
        obj.scaler.scale_ = np.asarray(state["scale"])
        obj.scaler.var_ = obj.scaler.scale_ ** 2
        obj.scaler.n_features_in_ = len(FEATURES)
        obj.scaler.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        return obj


def make_dataset(frame, prep, latest=None):
    df, values = prep.transform(frame)
    seqs, curs, t12, t24, cls, pats = [], [], [], [], [], []
    groups = {p: g for p, g in df.groupby("PATNO", sort=False)}
    rows = []
    if latest is None:
        for pat, group in groups.items():
            for ix in group.index[group[TARGETS[0]].notna()]:
                rows.append((pat, group.loc[:ix], df.loc[ix]))
    else:
        if latest.PATNO.duplicated().any():
            raise ValueError("Duplicate validation patients")
        for _, row in latest.iterrows():
            group = groups[row.PATNO]
            prefix = group[group.visit_date <= row.visit_date]
            if prefix.empty or prefix.iloc[-1].visit_date != row.visit_date:
                raise ValueError("Missing exact prediction visit")
            if not np.isclose(prefix.iloc[-1].NP3TOT, row.NP3TOT):
                raise ValueError("Current motor score disagrees with canonical visit")
            rows.append((row.PATNO, prefix, row))
    for pat, prefix, row in rows:
        seqs.append(values[prefix.index])
        # Preserve raw motor score, separate from standardized features.
        curs.append(float(prefix.iloc[-1].NP3TOT))
        t12.append(row[TARGETS[0]])
        t24.append(row[TARGETS[1]])
        cls.append(row[TARGETS[2]])
        pats.append(pat)
    return old.SequenceDataset(seqs, curs, t12, t24, cls, pats)


class EarlyFusion(old.EarlyFusionModel):
    def __init__(self, cfg):
        super().__init__(input_dim=44 * (2 if cfg["missing"] else 1),
                         hidden_dim=cfg["early_hidden"], dropout=cfg["dropout"])
        h = cfg["early_hidden"]
        if cfg["layers"] != 2:
            self.bilstm = nn.LSTM(h, h, num_layers=cfg["layers"], batch_first=True,
                                 bidirectional=True, dropout=cfg["dropout"] if cfg["layers"] > 1 else 0)

    def forward(self, x, mask, lengths, cur_scores):
        return super().forward(x[..., :-1], mask, lengths, cur_scores)


class LateFusion(old.LateFusionModel):
    def __init__(self, cfg):
        super().__init__(hidden_dim=cfg["late_hidden"], dropout=cfg["dropout"])
        self.clin_idx = list(range(30)) + list(range(41, 44))
        self.img_idx = list(range(30, 44))
        if cfg["missing"]:
            self.clin_idx += [i + 44 for i in self.clin_idx.copy()]
            self.img_idx += [i + 44 for i in self.img_idx.copy()]
            self.clin_proj[0] = nn.Linear(len(self.clin_idx), cfg["late_hidden"])
            self.img_encoder[0] = nn.Linear(len(self.img_idx), cfg["late_hidden"])
        if cfg["layers"] != 2:
            h = cfg["late_hidden"]
            self.clin_lstm = nn.LSTM(h, h, num_layers=cfg["layers"], batch_first=True,
                                    bidirectional=True, dropout=cfg["dropout"] if cfg["layers"] > 1 else 0)
        self.mask_absent = cfg["mask_absent"]

    def forward(self, x, mask, lengths, cur_scores):
        c_proj = self.clin_proj(x[..., self.clin_idx])
        packed = nn.utils.rnn.pack_padded_sequence(c_proj, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.clin_lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        hc, _ = self.clin_attn(out, mask)
        # Index each patient's last REAL observation, never the batch padding.
        last = x[torch.arange(len(x), device=x.device), lengths - 1]
        hi = self.img_encoder(last[:, self.img_idx])
        joint = torch.cat([hc, hi], dim=-1)
        gates = self.gate_net(joint)
        available = last[:, -1] > 0
        if self.mask_absent:
            gates = torch.where(available[:, None], gates, gates.new_tensor([1.0, 0.0]))
        p12 = torch.stack([cur_scores + self.clin_head_12(hc).squeeze(-1),
                           cur_scores + self.img_head_12(hi).squeeze(-1)], dim=1)
        p24 = torch.stack([cur_scores + self.clin_head_24(hc).squeeze(-1),
                           cur_scores + self.img_head_24(hi).squeeze(-1)], dim=1)
        # Live tensors enable optional branch auxiliary supervision (no detach).
        self.branch_outputs = (p12, p24)
        self.available = available
        return (p12 * gates).sum(1), (p24 * gates).sum(1), self.head_cls(joint).squeeze(-1), gates


def build_model(architecture, cfg):
    return (EarlyFusion if architecture == "early" else LateFusion)(cfg)


def masked_loss(pred, target, kind="huber", beta=1.0):
    valid = torch.isfinite(target)
    if not valid.any():
        return pred.sum() * 0.0
    if kind == "mae":
        return (pred[valid] - target[valid]).abs().mean()
    return nn.functional.smooth_l1_loss(pred[valid], target[valid], beta=beta)


def predict(model, loader, tab):
    model.eval()
    parts = []
    with torch.no_grad():
        for x, mask, lengths, cur, t12, t24, cls, pats in loader:
            a, b, logits, extra = model(x, mask, lengths, cur)
            part = pd.DataFrame(dict(PATNO=pats, pred_12m=a.numpy(), pred_24m=b.numpy(),
                                     true_12m=t12.numpy(), true_24m=t24.numpy(),
                                     true_mcid=cls.numpy(), prob_mcid=logits.sigmoid().numpy(),
                                     sequence_length=lengths.numpy()))
            if isinstance(model, LateFusion):
                for h, branch in zip([12, 24], model.branch_outputs):
                    part[f"clinical_{h}m"] = branch[:, 0].numpy()
                    part[f"imaging_{h}m"] = branch[:, 1].numpy()
                    part[f"uniform_{h}m"] = branch.mean(1).numpy()
                part["weight_imaging"] = extra[:, 1].numpy()
                part["imaging_available"] = model.available.numpy()
            parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    if result.PATNO.tolist() != tab.PATNO.tolist():
        raise ValueError("Patient order mismatch")
    result["visit_date"] = tab.visit_date.to_numpy()
    for h in [12, 24]:
        result[f"aligned_{h}m"] = tab[f"aligned_{h}m"].to_numpy()
        if not np.allclose(result[f"true_{h}m"], tab[f"target_np3tot_{h}m"], equal_nan=True):
            raise ValueError("Target mismatch")
        if not np.isfinite(result[f"pred_{h}m"]).all():
            raise ValueError("Non-finite predictions")
    return result


def metrics(pred):
    result = []
    predictors = ["pred"] + (["clinical", "imaging", "uniform"] if "clinical_12m" in pred else [])
    for protocol in ["canonical", "aligned"]:
        for h in [12, 24]:
            subset = pred if protocol == "canonical" else pred[pred[f"aligned_{h}m"]]
            for predictor in predictors:
                m = evaluate_regression(subset[f"true_{h}m"].to_numpy(), subset[f"{predictor}_{h}m"].to_numpy())
                result.append(dict(protocol=protocol, horizon=h, predictor=predictor, **m))
    return result


def selection_score(pred):
    # Do not tune against stale 24m labels. Equal horizon weights, patient-level MAE.
    return float(np.mean([np.mean(np.abs(pred.loc[pred[f"aligned_{h}m"], f"pred_{h}m"] -
                                       pred.loc[pred[f"aligned_{h}m"], f"true_{h}m"])) for h in [12, 24]]))


def load_audit(output):
    train = pd.read_parquet(DATA / "train_longitudinal.parquet")
    val = pd.read_parquet(DATA / "val_longitudinal.parquet")
    tab = pd.read_parquet(DATA / "val_tabular_latest.parquet").reset_index(drop=True)
    splits = pd.read_csv(DATA / "patient_split.csv")
    if splits.PATNO.duplicated().any():
        raise ValueError("PATNO assigned multiple splits")
    mapping = splits.set_index("PATNO").SPLIT.str.lower()
    audit = {}
    for name, df in [("train", train), ("val", val)]:
        assert df.PATNO.map(mapping).eq(name).all()
        assert df.visit_date.notna().all()
        assert np.isfinite(df.NP3TOT).all()
        assert (df.delta_t_days >= 0).all()
        audit[name] = dict(rows=len(df), patients=df.PATNO.nunique(),
                           duplicate_patient_dates=int(df.duplicated(["PATNO", "visit_date"]).sum()),
                           max_length=int(df.groupby("PATNO").size().max()),
                           missing_fraction=df[FEATURES].isna().mean().to_dict())
    assert not set(train.PATNO) & set(val.PATNO)
    # Only PATNO is read from test, exclusively to verify the canonical partition.
    test_ids = pd.read_parquet(DATA / "test_longitudinal.parquet", columns=["PATNO"])
    assert test_ids.PATNO.map(mapping).eq("test").all()
    assert not set(test_ids.PATNO) & (set(train.PATNO) | set(val.PATNO))
    groups = {p: g for p, g in ordered(val).groupby("PATNO")}
    for h in [12, 24]:
        k = f"target_np3tot_{h}m"
        endpoint = np.array([groups[r.PATNO].loc[lambda g: g.visit_date <= r.visit_date].iloc[-1][k]
                             for r in tab.itertuples()])
        tab[f"endpoint_target_{h}m"] = endpoint
        tab[f"aligned_{h}m"] = np.isfinite(endpoint) & np.isclose(endpoint, tab[k])
        audit[f"validation_{h}m"] = dict(canonical_valid=int(tab[k].notna().sum()),
                                        aligned_valid=int(tab[f"aligned_{h}m"].sum()),
                                        mismatch=int((~np.isclose(endpoint, tab[k], equal_nan=True)).sum()))
    # Recompute longitudinal targets from strictly future dated motor observations.
    for name, df in [("train", train), ("val", val)]:
        mismatch = {12: 0, 24: 0}
        datescores = df.groupby(["PATNO", "visit_date"]).NP3TOT.mean().reset_index()
        for pat, group in datescores.groupby("PATNO"):
            original = df[df.PATNO == pat]
            dates = group.visit_date.to_numpy()
            scores = group.NP3TOT.to_numpy()
            for row in group.itertuples():
                days = (dates - np.datetime64(row.visit_date)) / np.timedelta64(1, "D")
                for h, lo, hi, center in [(12,250,480,365),(24,550,910,730)]:
                    ix = np.flatnonzero((days >= lo) & (days <= hi))
                    target = scores[ix[np.argmin(abs(days[ix]-center))]] if len(ix) else np.nan
                    actual = original.loc[original.visit_date == row.visit_date, f"target_np3tot_{h}m"]
                    mismatch[h] += int((~np.isclose(actual, target, equal_nan=True)).sum())
        audit[name]["longitudinal_target_mismatches"] = mismatch
        assert not any(mismatch.values())
    audit["test_policy"] = "PATNO only for split audit; no test inputs, labels, predictions or metrics"
    audit["upstream_limit"] = "Modalities joined on PATNO/EVENT_ID, source acquisition dates absent from processed data; within-event availability not certified."
    tab[["PATNO", "visit_date", TARGETS[0], TARGETS[1], "endpoint_target_12m",
         "endpoint_target_24m", "aligned_12m", "aligned_24m"]].to_csv(
             output / "validation_target_audit.csv", index=False)
    write_json(output / "audit.json", audit)
    return train, val, tab


def historical(val, train, tab, output):
    medians = train[FEATURES].median().fillna(0)
    scaler = StandardScaler().fit(train[FEATURES].fillna(medians))
    scaled = old.build_trajectories(val, FEATURES, scaler, medians)
    ds = old.SequenceDataset(*old.extract_labeled_sequences(scaled, FEATURES, "latest_only", tab))
    loader = DataLoader(ds, batch_size=64, collate_fn=old.pad_collate_fn)
    for name, model in [("early", old.EarlyFusionModel()), ("late", old.LateFusionModel())]:
        model.load_state_dict(torch.load(f"models/{name}_fusion.pt", weights_only=True, map_location="cpu"))
        pred = predict(model, loader, tab)
        pred.to_csv(output / f"historical_{name}_predictions.csv", index=False)
        write_json(output / f"historical_{name}_metrics.json", metrics(pred))
        print("HISTORICAL", name, metrics(pred)[:2], flush=True)


def train_run(architecture, cfg, seed, train, val, tab, output, threads):
    dest = output / f"{architecture}_{cfg['name']}_seed{seed}"
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite experiment {dest}")
    dest.mkdir()
    seed_all(seed, threads)
    prep = Preprocessor(cfg["missing"]).fit(train)
    ds = make_dataset(train, prep)
    vd = make_dataset(val, prep, tab)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), collate_fn=old.pad_collate_fn)
    vl = DataLoader(vd, batch_size=64, collate_fn=old.pad_collate_fn)
    model = build_model(architecture, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    best, best_epoch, bad = float("inf"), 0, 0
    history = []
    start = time.perf_counter()
    write_json(dest / "config.json", dict(architecture=architecture, config=cfg, seed=seed, threads=threads))
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, count = 0.0, 0
        for x, mask, lengths, cur, t12, t24, cls, _ in loader:
            optimizer.zero_grad()
            a, b, logits, _ = model(x, mask, lengths, cur)
            loss = masked_loss(a, t12, cfg["loss"], cfg["beta"]) + cfg["weight24"] * masked_loss(b, t24, cfg["loss"], cfg["beta"])
            valid = torch.isfinite(cls)
            if valid.any():
                loss = loss + cfg["classification_weight"] * nn.functional.binary_cross_entropy_with_logits(logits[valid], cls[valid])
            if isinstance(model, LateFusion) and cfg["auxiliary_weight"]:
                for branch, target, weight in zip(model.branch_outputs, [t12,t24], [1.,cfg["weight24"]]):
                    loss = loss + cfg["auxiliary_weight"] * weight * masked_loss(branch[:, 0], target, cfg["loss"], cfg["beta"])
                    available_target = torch.where(model.available, target, torch.nan)
                    loss = loss + cfg["auxiliary_weight"] * weight * masked_loss(branch[:, 1], available_target, cfg["loss"], cfg["beta"])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"], error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        pred = predict(model, vl, tab)
        score = selection_score(pred)
        scheduler.step(score)
        row = dict(epoch=epoch, train_loss=total/count, selection_mae=score,
                   mae12=float(abs(pred.pred_12m-pred.true_12m).mean()),
                   mae24_aligned=float(abs(pred.loc[pred.aligned_24m,"pred_24m"]-pred.loc[pred.aligned_24m,"true_24m"]).mean()),
                   lr=optimizer.param_groups[0]["lr"])
        history.append(row)
        if score < best:
            best, best_epoch, bad = score, epoch, 0
            # Clone tensors: state_dict().copy() alone aliases live model weights.
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        print(dest.name, row, flush=True)
        if bad >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    checkpoint = dict(model_state_dict=best_state, architecture=architecture, config=cfg,
                      seed=seed, epoch=best_epoch, selection_mae=best,
                      preprocessor=prep.state(), selection_protocol="mean aligned MAE12/MAE24",
                      data_sha256={p.name:sha(p) for p in DATA.glob('*.parquet') if p.name.startswith(('train_','val_'))})
    torch.save(checkpoint, dest / "checkpoint.pt")
    pd.DataFrame(history).to_csv(dest / "history.csv", index=False)
    pred = predict(model, vl, tab)
    pred.to_csv(dest / "validation_predictions.csv", index=False)
    write_json(dest / "metrics.json", metrics(pred))
    write_json(dest / "classification.json", evaluate_classification(pred.true_mcid.to_numpy(), pred.prob_mcid.to_numpy()))
    summary = dict(architecture=architecture, config=cfg["name"], seed=seed,
                   epoch=best_epoch, selection_mae=best, seconds=time.perf_counter()-start,
                   parameters=sum(p.numel() for p in model.parameters()))
    write_json(dest / "summary.json", summary)
    return summary


def reload_validation(path, val, tab):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    prep = Preprocessor.restore(ck["preprocessor"])
    model = build_model(ck["architecture"], ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    loader = DataLoader(make_dataset(val, prep, tab), batch_size=64, collate_fn=old.pad_collate_fn)
    return predict(model, loader, tab)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fusion_experiments.json")
    parser.add_argument("--output", default="experiments/fusion_20261013")
    parser.add_argument("--stage", choices=["audit", "screen", "repeat", "run", "verify"], default="screen")
    parser.add_argument("--architecture", choices=["early", "late", "both"],
                        help="Fusion to train when --stage run is used")
    parser.add_argument("--experiment", default="compact_mae",
                        help="Named configuration from --config for --stage run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --stage run")
    args = parser.parse_args(argv)
    plan = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "run":
        if args.architecture is None:
            parser.error("--stage run requires --architecture early, late or both")
        if args.experiment not in {c["name"] for c in plan["experiments"]}:
            parser.error(f"Unknown experiment: {args.experiment}")
        run_architectures = ["early", "late"] if args.architecture == "both" else [args.architecture]
        for architecture in run_architectures:
            run_dir = output / f"{architecture}_{args.experiment}_seed{args.seed}"
            if run_dir.exists():
                raise FileExistsError(f"Refusing to overwrite experiment {run_dir}; use a new --output")
    if args.stage == "screen" and any(output.glob("*_seed*")):
        raise FileExistsError("Use a new --output directory to preserve previous runs")
    if args.stage == "repeat":
        saved_plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        if plan != saved_plan:
            raise ValueError("Repeat configuration differs from the frozen screening plan")
    seed_all(42, plan["threads"])
    train, val, tab = load_audit(output)
    if args.stage in ["audit", "screen", "run"]:
        (output / "requirements-lock.txt").write_text(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
        source_paths = list(Path("src").glob("*.py")) + [Path(args.config)]
        write_json(output / "source_sha256.json", {str(p):sha(p) for p in source_paths})
        write_json(output / "environment.json", dict(python=platform.python_version(), platform=platform.platform(),
                   cpu=os.environ.get("PROCESSOR_IDENTIFIER"), logical_cpus=os.cpu_count(),
                   cuda=torch.cuda.is_available(), threads=plan["threads"],
                   packages={p:importlib.metadata.version(p) for p in ["torch","numpy","pandas","scikit-learn","scipy","pyarrow"]}))
        protected = list(Path("models").glob('*')) + list(Path("report").glob('*')) + list(DATA.glob('*'))
        manifest = {str(p):sha(p) for p in protected if p.is_file()}
        if not (output / "original_artifacts_sha256.json").exists():
            write_json(output / "original_artifacts_sha256.json", manifest)
        if args.stage != "run":
            historical(val, train, tab, output)
    if args.stage == "audit":
        return
    configs = [{**plan["defaults"], **c} for c in plan["experiments"]]
    if args.stage == "run":
        cfg = next(c for c in configs if c["name"] == args.experiment)
        for architecture in run_architectures:
            summary = train_run(architecture, cfg, args.seed, train, val, tab, output, plan["threads"])
            run_dir = output / f"{architecture}_{args.experiment}_seed{args.seed}"
            print(f"Saved {architecture} validation run to {run_dir}")
            print(summary)
        if args.architecture == "both":
            write_json(output / "selection.json", dict(
                criterion="configuration fixed before this run; no search performed",
                configurations={a: args.experiment for a in run_architectures}, seed=args.seed,
                warning="24m validation has only 14 correctly aligned labels"))
    if args.stage == "screen":
        write_json(output / "plan.json", plan)
        summaries = []
        for cfg in configs:
            for architecture in ["early", "late"]:
                summaries.append(train_run(architecture,cfg,42,train,val,tab,output,plan["threads"]))
        chosen = {a:min([s for s in summaries if s["architecture"]==a], key=lambda s:s["selection_mae"])["config"] for a in ["early","late"]}
        write_json(output / "selection.json", dict(criterion="mean aligned MAE12/MAE24; seed42 screening", configurations=chosen,
                   warning="24m selection uses only 14 aligned validation labels; high uncertainty"))
    if args.stage == "repeat":
        chosen = json.loads((output / "selection.json").read_text())["configurations"]
        for seed in plan["repeat_seeds"]:
            for architecture in ["early", "late"]:
                for name in dict.fromkeys(["corrected", chosen[architecture]]):
                    cfg = next(c for c in configs if c["name"] == name)
                    train_run(architecture,cfg,seed,train,val,tab,output,plan["threads"])
    if args.stage == "verify":
        verified = []
        for path in output.glob("*/checkpoint.pt"):
            pred = reload_validation(path, val, tab)
            saved = pd.read_csv(path.parent / "validation_predictions.csv")
            np.testing.assert_allclose(pred[["pred_12m","pred_24m"]], saved[["pred_12m","pred_24m"]], rtol=1e-6, atol=1e-6)
            verified.append(str(path))
        manifest = json.loads((output / "original_artifacts_sha256.json").read_text())
        assert all(sha(p)==digest for p,digest in manifest.items()), "An original artifact changed"
        write_json(output / "verification.json", dict(checkpoints_reloaded=verified,
                   original_artifacts_unchanged=len(manifest), prediction_rtol=1e-6,
                   prediction_atol=1e-6, passed=True))
        print("All saved predictions reproduced; all original artifacts unchanged.")


if __name__ == "__main__":
    main()
