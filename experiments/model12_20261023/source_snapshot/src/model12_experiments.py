"""Reproducible, isolated XGBoost/BiLSTM tuning. Run from the repository root.

Stages: prepare -> screen -> repeat -> select -> test -> verify.
Only test/verify load test features after a persisted validation-only selection.
Preparation audits test target integrity without calculating model performance.
The original BiLSTM class is instantiated unchanged (44 inputs, 64 units, 2
bidirectional layers); no fusion module is imported or changed.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import xgboost as xgb
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, r2_score, roc_curve)

import train_model2_lstm as old
from model12_data import (FEATURES, STATIC_FEATURES, Preprocessor,
                          make_dataset, prepare, sha, write_json)

TASKS = ["12m", "24m", "cls"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def regression(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    valid = np.isfinite(y)
    if not np.isfinite(p[valid]).all():
        raise ValueError("Nonfinite predictions must not be silently excluded")
    y, p = y[valid], p[valid]
    return dict(n=len(y), mae=float(np.mean(abs(y-p))) if len(y) else np.nan,
                rmse=float(np.sqrt(np.mean((y-p)**2))) if len(y) else np.nan,
                r2=float(r2_score(y, p)) if len(y) > 1 else np.nan,
                bias=float(np.mean(p-y)) if len(y) else np.nan)


def classification(y, p, threshold=0.5):
    y, p = np.asarray(y, float), np.asarray(p, float)
    valid = np.isfinite(y)
    y, p = y[valid].astype(int), p[valid]
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probability")
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    sens = tp / (tp+fn) if tp+fn else np.nan
    spec = tn / (tn+fp) if tn+fp else np.nan
    both = len(np.unique(y)) == 2
    return dict(n=len(y), accuracy=float((tp+tn)/len(y)) if len(y) else np.nan,
                balanced_accuracy=float((sens+spec)/2), sensitivity=float(sens), specificity=float(spec),
                roc_auc=float(roc_auc_score(y, p)) if both else np.nan,
                average_precision=float(average_precision_score(y, p)) if both else np.nan,
                brier_score=float(np.mean((p-y)**2)) if len(y) else np.nan,
                threshold=float(threshold), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
                probability_quantiles=np.quantile(p, [0, .1, .25, .5, .75, .9, 1]).tolist() if len(y) else [])


def choose_threshold(y, p):
    """Validation only. Avoid sensitivity-only selection; equal class utility."""
    y, p = np.asarray(y), np.asarray(p)
    valid = np.isfinite(y)
    y, p = y[valid], p[valid]
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both validation classes")
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    eligible = np.flatnonzero((tpr >= .5) & (1-fpr >= .5) & np.isfinite(thresholds))
    if not len(eligible):
        # No viable discrimination: make the failed constraint explicit.
        return .5, False
    best = min(eligible, key=lambda i: (-(tpr[i]+1-fpr[i])/2, abs(thresholds[i]-.5)))
    return float(thresholds[best]), True


def rank_class(m):
    return (m["roc_auc"] + m["average_precision"]) / 2


def model(cfg):
    return old.LongitudinalBiLSTM(input_dim=len(FEATURES), hidden_dim=64,
                                 num_layers=2, dropout=cfg["dropout"])


def signature(net):
    # Dropout probabilities are permitted training hyperparameters; module
    # topology, tensor dimensions, directions and forward source stay fixed.
    return dict(modules=[(n, type(m).__name__) for n, m in net.named_modules()],
                parameters={n: list(p.shape) for n, p in net.state_dict().items()},
                count=sum(p.numel() for p in net.parameters()),
                input_dim=net.input_dim, hidden_dim=net.hidden_dim,
                layers=net.lstm.num_layers, bidirectional=net.lstm.bidirectional)


def seed_all(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def load_frame(root, split, name="longitudinal"):
    return pd.read_parquet(root / "data" / f"{split}_{name}.parquet")


def loader(dataset, batch_size, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=old.collate_sequences, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def predict_lstm(net, data_loader, rows):
    res = old.evaluate_dataset(net, data_loader, torch.device("cpu"))
    if res["patnos"] != rows.PATNO.tolist():
        raise ValueError("Prediction/target patient order mismatch")
    result = rows[["PATNO", "row_id", "visit_date", "eval12", "eval24"]].copy()
    for col in ["pred_12m", "pred_24m", "prob_mcid", "true_12m", "true_24m", "true_mcid"]:
        result[col] = res[col]
    for h in [12, 24]:
        if not np.allclose(result[f"true_{h}m"], rows[f"target_np3tot_{h}m"], equal_nan=True):
            raise ValueError("Mismatched target")
    return result


def metrics_lstm(pred, threshold):
    a, b = pred[pred.eval12], pred[pred.eval24]
    return {"12m": regression(a.true_12m, a.pred_12m),
            "24m": regression(b.true_24m, b.pred_24m),
            "cls": classification(a.true_mcid, a.prob_mcid, threshold),
            "cls_at_05": classification(a.true_mcid, a.prob_mcid, .5)}


def masked_reg(pred, target, cfg):
    valid = torch.isfinite(target)
    if not valid.any():
        return pred.sum() * 0
    if cfg["loss"] == "mae":
        return (pred[valid]-target[valid]).abs().mean()
    return nn.functional.smooth_l1_loss(pred[valid], target[valid], beta=cfg["beta"])


def masked_bce(logits, target, pos_weight):
    valid = torch.isfinite(target)
    if not valid.any():
        return logits.sum() * 0
    return nn.functional.binary_cross_entropy_with_logits(logits[valid], target[valid], pos_weight=pos_weight)


def fit_lstm(root, cfg, seed, config, prepared):
    folder = root / "runs" / "bilstm" / cfg["name"] / str(seed)
    if (folder / "metrics.json").exists():
        return read_json(folder / "metrics.json")
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(seed, config["threads"])
    train_ds, val_ds, val_rows, prep = prepared
    train_loader = loader(train_ds, cfg["batch_size"], True, seed)
    val_loader = loader(val_ds, 256)
    net = model(cfg)
    assert signature(net) == signature(model(config["lstm_defaults"]))
    # Re-seed after the architecture check instantiated an unused model.
    seed_all(seed, config["threads"])
    net = model(cfg)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg["scheduler_factor"], patience=cfg["scheduler_patience"])
    labels = np.asarray(train_ds.targets_mcid)
    labels = labels[np.isfinite(labels)]
    ratio = float((len(labels)-labels.sum()) / labels.sum())
    pos_weight = torch.tensor(ratio if cfg["balanced_loss"] else 1.0)
    best_score, best_epoch, best_weights, history = np.inf, 0, None, []
    start = time.time()
    for epoch in range(1, cfg["epochs"]+1):
        ep_start = time.time()
        net.train()
        losses = []
        for x, mask, lengths, cur, t12, t24, cls, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            a, b, logits, _ = net(x, mask, lengths, cur)
            loss = (masked_reg(a, t12, cfg) + cfg["weight24"]*masked_reg(b, t24, cfg)
                    + cfg["classification_weight"]*masked_bce(logits, cls, pos_weight))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg["clip"], error_if_nonfinite=True)
            optimizer.step()
            losses.append((float(loss.detach()), len(x)))
        pred = predict_lstm(net, val_loader, val_rows)
        m = metrics_lstm(pred, .5)
        score = m["12m"]["mae"]
        if cfg["selection"] == "mean_mae":
            score = (score + m["24m"]["mae"])/2
        elif cfg["selection"] == "classification":
            score = -rank_class(m["cls"])
        improved = score < best_score - 1e-7
        if improved:
            best_score, best_epoch = score, epoch
            best_weights = copy.deepcopy(net.state_dict())
            # Persist actual best weights immediately, never a live reference.
            torch.save(dict(model_state_dict=best_weights, epoch=epoch, score=score,
                            config=cfg, seed=seed, preprocessor=prep.state(), architecture=signature(net)), folder / "best.pt")
        scheduler.step(score)
        history.append(dict(epoch=epoch, score=score, mae12=m["12m"]["mae"], mae24=m["24m"]["mae"],
                            roc_auc=m["cls"]["roc_auc"], average_precision=m["cls"]["average_precision"],
                            train_loss=sum(l*n for l,n in losses)/sum(n for _,n in losses),
                            lr=optimizer.param_groups[0]["lr"], seconds=time.time()-ep_start, best=improved))
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        if epoch == 1 or epoch % 5 == 0 or improved:
            print(f"BiLSTM {cfg['name']} seed={seed} epoch={epoch} MAE={m['12m']['mae']:.4f}/{m['24m']['mae']:.4f} AUC={m['cls']['roc_auc']:.4f} {time.time()-ep_start:.1f}s {'BEST' if improved else ''}", flush=True)
        if epoch-best_epoch >= cfg["patience"]:
            break
    checkpoint = torch.load(folder / "best.pt", weights_only=False, map_location="cpu")
    for key in best_weights:
        assert torch.equal(checkpoint["model_state_dict"][key], best_weights[key])
    net.load_state_dict(checkpoint["model_state_dict"])
    pred = predict_lstm(net, val_loader, val_rows)
    a = pred[pred.eval12]
    threshold, feasible = choose_threshold(a.true_mcid, a.prob_mcid)
    m = metrics_lstm(pred, threshold)
    result = dict(model="bilstm", name=cfg["name"], seed=seed, config=cfg, epoch=best_epoch,
                  elapsed_seconds=time.time()-start, pos_weight=float(pos_weight),
                  train_prevalence=float(labels.mean()), threshold_feasible=feasible,
                  threshold=threshold, metrics=m)
    pred.to_csv(folder / "validation_predictions.csv", index=False, float_format="%.17g")
    write_json(folder / "metrics.json", result)
    return result


def xgb_configs(config, task):
    baseline = dict(name="baseline", n_estimators=600, learning_rate=.03, max_depth=4,
                    min_child_weight=1, subsample=.8, colsample_bytree=.8,
                    reg_alpha=.5, reg_lambda=1.5, gamma=0, early_stopping_rounds=40,
                    eval_metric="rmse", objective="reg:squarederror")
    if task == "cls":
        baseline.update(n_estimators=500, reg_alpha=0., reg_lambda=1., early_stopping_rounds=35,
                        eval_metric="auc", objective="binary:logistic")
    configs = [baseline]
    rng = np.random.default_rng(config["xgb_search_seed"])
    for i in range(config["xgb_trials"]):
        cfg = dict(baseline, name=f"trial_{i:02d}", n_estimators=int(rng.choice([700, 1100, 1600])),
                   learning_rate=float(rng.choice([.015, .03, .05, .08])),
                   max_depth=int(rng.choice([2, 3, 4, 5])), min_child_weight=int(rng.choice([1, 4, 10, 20])),
                   subsample=float(rng.choice([.65, .8, 1.])), colsample_bytree=float(rng.choice([.65, .8, 1.])),
                   reg_alpha=float(rng.choice([0., .5, 2., 5.])), reg_lambda=float(rng.choice([1., 5., 15., 30.])),
                   gamma=float(rng.choice([0., .1, .5, 2.])), early_stopping_rounds=60,
                   eval_metric="aucpr" if task == "cls" else "mae")
        configs.append(cfg)
    return configs


def fit_xgb(root, cfg, task, seed, config):
    folder = root / "runs" / "xgboost" / task / cfg["name"] / str(seed)
    if (folder / "metrics.json").exists():
        return read_json(folder / "metrics.json")
    folder.mkdir(parents=True, exist_ok=True)
    horizon = 12 if task == "cls" else int(task[:-1])
    train, val = [load_frame(root, s, f"tabular_{horizon}m") for s in ["train", "val"]]
    target = "prog_mcid_12m" if task == "cls" else f"target_np3tot_{task}"
    train, val = [df[df[target].notna()] for df in [train, val]]
    params = {k:v for k,v in cfg.items() if k != "name"}
    params.update(n_jobs=config["threads"], random_state=seed, tree_method="hist")
    if task == "cls":
        params["scale_pos_weight"] = float((len(train)-train[target].sum()) / train[target].sum())
    net = (xgb.XGBClassifier if task == "cls" else xgb.XGBRegressor)(**params)
    start = time.time()
    net.fit(train[STATIC_FEATURES], train[target], eval_set=[(val[STATIC_FEATURES], val[target])], verbose=False)
    pred = net.predict_proba(val[STATIC_FEATURES])[:,1] if task == "cls" else net.predict(val[STATIC_FEATURES])
    threshold, feasible = choose_threshold(val[target], pred) if task == "cls" else (.5, True)
    m = classification(val[target], pred, threshold) if task == "cls" else regression(val[target], pred)
    result = dict(model="xgboost", task=task, name=cfg["name"], seed=seed, config=cfg,
                  fitted_params=params, best_iteration=net.best_iteration, threshold=threshold,
                  threshold_feasible=feasible, metrics=m, elapsed_seconds=time.time()-start)
    if task == "cls":
        result["metrics_at_05"] = classification(val[target], pred, .5)
    net.save_model(folder / "best.json")
    output = val[["PATNO", "row_id", "visit_date"]].copy()
    output["truth"], output["prediction"] = val[target].to_numpy(), pred
    output.to_csv(folder / "validation_predictions.csv", index=False, float_format="%.17g")
    write_json(folder / "metrics.json", result)
    print(f"XGB {task} {cfg['name']} seed={seed} best={net.best_iteration} metric={rank_class(m) if task == 'cls' else m['mae']:.5f}", flush=True)
    return result


def prepare_lstm(root):
    train, val = [load_frame(root, s) for s in ["train", "val"]]
    rows = load_frame(root, "val", "evaluation")
    prep = Preprocessor().fit(train)
    return make_dataset(train, prep), make_dataset(val, prep, rows), rows, prep


def expanded_lstm(config):
    return [dict(config["lstm_defaults"], **cfg) for cfg in config["lstm_trials"]]


def screen(root, config, family):
    seed = config["screen_seed"]
    if family in ["all", "xgboost"]:
        for task in TASKS:
            for cfg in xgb_configs(config, task):
                fit_xgb(root, cfg, task, seed, config)
    if family in ["all", "bilstm"]:
        prepared = prepare_lstm(root)
        for cfg in expanded_lstm(config):
            fit_lstm(root, cfg, seed, config, prepared)


def load_result(root, family, task, name, seed):
    folder = root / "runs" / family
    if family == "xgboost":
        folder /= task
    return read_json(folder / name / str(seed) / "metrics.json")


def task_metric(result, family, task):
    return result["metrics"][task] if family == "bilstm" else result["metrics"]


def loss_metric(m, task):
    return -rank_class(m) if task == "cls" else m["mae"]


def shortlist(root, config):
    chosen = {}
    for family in ["xgboost", "bilstm"]:
        chosen[family] = {}
        for task in TASKS:
            cfgs = xgb_configs(config, task) if family == "xgboost" else expanded_lstm(config)
            scored = [(loss_metric(task_metric(load_result(root, family, task, c["name"], config["screen_seed"]), family, task), task), c["name"]) for c in cfgs]
            chosen[family][task] = min(scored)[1]
    write_json(root / "shortlist.json", chosen)
    return chosen


def repeat(root, config, family):
    chosen = shortlist(root, config)
    if family in ["all", "xgboost"]:
        for task in TASKS:
            for cfg in xgb_configs(config, task):
                if cfg["name"] in ["baseline", chosen["xgboost"][task]]:
                    for seed in config["repeat_seeds"]:
                        fit_xgb(root, cfg, task, seed, config)
    if family in ["all", "bilstm"]:
        prepared = prepare_lstm(root)
        # The MAE and classification-Huber screens have almost tied class
        # ranking. Repeat all three small-budget candidates, never a best seed.
        names = {cfg["name"] for cfg in expanded_lstm(config)}
        for cfg in expanded_lstm(config):
            if cfg["name"] in names:
                for seed in config["repeat_seeds"]:
                    fit_lstm(root, cfg, seed, config, prepared)


def select(root, config):
    chosen = read_json(root / "shortlist.json")
    seeds = [config["screen_seed"], *config["repeat_seeds"]]
    decisions = {}
    for family in ["xgboost", "bilstm"]:
        decisions[family] = {}
        for task in TASKS:
            names = (["baseline", chosen[family][task]] if family == "xgboost"
                     else [cfg["name"] for cfg in expanded_lstm(config)])
            summaries = {}
            for name in sorted(set(names)):
                values = [task_metric(load_result(root, family, task, name, seed), family, task) for seed in seeds]
                summaries[name] = {k:float(np.mean([m[k] for m in values])) for k in
                                   (["roc_auc", "average_precision", "balanced_accuracy", "sensitivity", "specificity"] if task == "cls" else ["mae", "rmse", "r2", "bias"])}
            base = summaries["baseline"]
            eligible = [name for name in summaries if task != "cls"
                        or summaries[name]["balanced_accuracy"] >= base["balanced_accuracy"]]
            candidate = min(eligible, key=lambda name: loss_metric(summaries[name], task))
            cand = summaries[candidate]
            gain = loss_metric(cand, task) < loss_metric(base, task) - 1e-6
            if task == "cls":
                gain = gain and cand["balanced_accuracy"] >= base["balanced_accuracy"]
            decisions[family][task] = dict(selected=candidate if gain else "baseline", candidate=candidate,
                                           validation_mean=summaries, gained=bool(gain))
    payload = dict(seeds=seeds, decisions=decisions, selection_data="validation only",
                   checkpoint_policy="one original-architecture network per selected task/seed; no ensemble; all seeds reported",
                   config_sha256=sha(root / "config.json"))
    write_json(root / "selection.json", payload)
    print(json.dumps(decisions, indent=2), flush=True)


def folder_for(root, family, task, name, seed):
    folder = root / "runs" / family
    if family == "xgboost":
        folder /= task
    return folder / name / str(seed)


def predict_saved(root, family, task, name, seed, split, cache):
    folder = folder_for(root, family, task, name, seed)
    result = read_json(folder / "metrics.json")
    if family == "xgboost":
        horizon = 12 if task == "cls" else int(task[:-1])
        rows = load_frame(root, split, f"tabular_{horizon}m")
        net = (xgb.XGBClassifier if task == "cls" else xgb.XGBRegressor)()
        net.load_model(folder / "best.json")
        target = "prog_mcid_12m" if task == "cls" else f"target_np3tot_{task}"
        out = rows[["PATNO", "row_id", "visit_date"]].copy()
        out["truth"] = rows[target].to_numpy()
        out["prediction"] = net.predict_proba(rows[STATIC_FEATURES])[:,1] if task == "cls" else net.predict(rows[STATIC_FEATURES])
    else:
        key = (name, seed, split)
        if key not in cache:
            ckpt = torch.load(folder / "best.pt", weights_only=False, map_location="cpu")
            net = model(ckpt["config"])
            net.load_state_dict(ckpt["model_state_dict"], strict=True)
            assert signature(net) == ckpt["architecture"]
            prep = Preprocessor.restore(ckpt["preprocessor"])
            frame, rows = load_frame(root, split), load_frame(root, split, "evaluation")
            ds = make_dataset(frame, prep, rows)
            cache[key] = predict_lstm(net, loader(ds, 256), rows)
        pred = cache[key]
        sub = pred[pred.eval24 if task == "24m" else pred.eval12]
        out = sub[["PATNO", "row_id", "visit_date"]].copy()
        out["truth"] = sub["true_mcid" if task == "cls" else f"true_{task}"].to_numpy()
        out["prediction"] = sub["prob_mcid" if task == "cls" else f"pred_{task}"].to_numpy()
    return out.reset_index(drop=True), result["threshold"]


def test(root, config):
    selection = read_json(root / "selection.json")
    if (root / "test_metrics.csv").exists():
        raise ValueError("Test already evaluated; use verify to replay frozen checkpoints")
    # Persist a freeze marker BEFORE any test model evaluation; further fitting
    # or selection in this output directory is refused, even after interruption.
    if (root / "test_freeze.json").exists():
        raise ValueError("Interrupted test stage: inspect freeze marker; no retuning permitted")
    frozen = {str(p.relative_to(root)): sha(p) for p in (root / "runs").rglob("best.*")}
    write_json(root / "test_freeze.json", dict(selection_sha256=sha(root / "selection.json"), checkpoints=frozen))
    rows, cache = [], {}
    pred_dir = root / "test_predictions"
    pred_dir.mkdir(exist_ok=True)
    for family in ["xgboost", "bilstm"]:
        for task in TASKS:
            name = selection["decisions"][family][task]["selected"]
            for seed in selection["seeds"]:
                outputs = {}
                for role, cfg_name in [("baseline", "baseline"), ("selected", name)]:
                    pred, threshold = predict_saved(root, family, task, cfg_name, seed, "test", cache)
                    outputs[role] = pred
                    pred.to_csv(pred_dir / f"{family}_{task}_{role}_{seed}.csv", index=False, float_format="%.17g")
                    thresholds = [("validation_threshold", threshold), ("fixed_0.5", .5)] if task == "cls" else [("na", .5)]
                    for policy, value in thresholds:
                        metric = classification(pred.truth, pred.prediction, value) if task == "cls" else regression(pred.truth, pred.prediction)
                        rows.append(dict(model=family, task=task, role=role, config=cfg_name, seed=seed, policy=policy,
                                         **{k:v for k,v in metric.items() if not isinstance(v, list)}))
                assert outputs["baseline"][["PATNO", "row_id", "truth"]].equals(outputs["selected"][["PATNO", "row_id", "truth"]])
    pd.DataFrame(rows).to_csv(root / "test_metrics.csv", index=False)
    # Show the multitask trade-off for each complete selected network as well;
    # task-specific recommendations must not hide its other output heads.
    full = []
    names = {"baseline", *[d["selected"] for d in selection["decisions"]["bilstm"].values()]}
    for name in sorted(names):
        for seed in selection["seeds"]:
            for task in TASKS:
                pred, threshold = predict_saved(root, "bilstm", task, name, seed, "test", cache)
                metric = classification(pred.truth, pred.prediction, threshold) if task == "cls" else regression(pred.truth, pred.prediction)
                full.append(dict(config=name, seed=seed, task=task,
                                 **{k:v for k,v in metric.items() if not isinstance(v, list)}))
    pd.DataFrame(full).to_csv(root / "bilstm_full_checkpoint_test_metrics.csv", index=False)


def verify(root, config):
    protected = read_json(root / "protected_sha256.json")
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError(f"Protected/shared file changed: {path}")
    for path, expected in read_json(root / "prepared_sha256.json").items():
        if sha(root / path) != expected:
            raise ValueError(f"Prepared data changed: {path}")
    frozen = read_json(root / "test_freeze.json")
    assert sha(root / "selection.json") == frozen["selection_sha256"]
    for path, expected in frozen["checkpoints"].items():
        assert sha(root / path) == expected
    selection = read_json(root / "selection.json")
    historical_checkpoint = torch.load("models/model2_bilstm.pt", weights_only=False, map_location="cpu")
    assert {k: list(v.shape) for k, v in historical_checkpoint["model_state_dict"].items()} == signature(model(config["lstm_defaults"]))["parameters"]
    cache, max_error = {}, 0.
    for family in ["xgboost", "bilstm"]:
        for task in TASKS:
            for seed in selection["seeds"]:
                for role, name in [("baseline", "baseline"), ("selected", selection["decisions"][family][task]["selected"])]:
                    pred, threshold = predict_saved(root, family, task, name, seed, "test", cache)
                    saved = pd.read_csv(root / "test_predictions" / f"{family}_{task}_{role}_{seed}.csv")
                    assert pred.PATNO.tolist() == saved.PATNO.tolist()
                    error = float(np.max(abs(pred.prediction.to_numpy()-saved.prediction.to_numpy())))
                    assert error < 1e-6
                    max_error = max(max_error, error)
                    if task == "cls":
                        validation, _ = predict_saved(root, family, task, name, seed, "val", cache)
                        replay_threshold, _ = choose_threshold(validation.truth, validation.prediction)
                        assert abs(threshold-replay_threshold) < 1e-10
    write_json(root / "verification.json", dict(protected_files_unchanged=True, prepared_data_unchanged=True,
                selection_frozen=True, checkpoints_unchanged=True, replay_max_abs_error=max_error,
                architecture_identical=True, architecture=signature(model(config["lstm_defaults"]))))
    print("Verification passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "screen", "repeat", "select", "test", "verify"], required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/model12_20261023"))
    parser.add_argument("--config", type=Path, default=Path("configs/model12_experiments.json"))
    parser.add_argument("--family", choices=["all", "xgboost", "bilstm"], default="all")
    args = parser.parse_args()
    root = args.output
    if args.stage == "prepare":
        root.mkdir(parents=True, exist_ok=False)
        config = read_json(args.config)
        write_json(root / "config.json", config)
        protected = [p for folder in ["models", "data/processed", "report", "src"] for p in Path(folder).glob("*")
                     if p.is_file() and "model12" not in p.name]
        protected += [Path("README.md"), Path("configs/fusion_experiments.json")]
        write_json(root / "protected_sha256.json", {str(p):sha(p) for p in protected})
        write_json(root / "architecture.json", signature(model(config["lstm_defaults"])))
        write_json(root / "environment.json", dict(python=sys.version, platform=platform.platform(),
            versions={p:importlib.metadata.version(p) for p in ["torch", "xgboost", "pandas", "numpy", "scikit-learn", "pyarrow"]},
            cuda=torch.cuda.is_available(), threads=config["threads"]))
        (root / "requirements-lock.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
        print(prepare(root), flush=True)
        return
    config = read_json(root / "config.json")
    seed_all(config["screen_seed"], config["threads"])
    if args.stage in ["screen", "repeat", "select"] and (root / "test_freeze.json").exists():
        raise ValueError("Test has been opened: tuning and selection are frozen")
    if args.stage == "screen":
        screen(root, config, args.family)
    elif args.stage == "repeat":
        repeat(root, config, args.family)
    elif args.stage == "select":
        select(root, config)
    elif args.stage == "test":
        test(root, config)
    else:
        verify(root, config)


if __name__ == "__main__":
    main()
