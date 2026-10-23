"""Second bounded search, original architectures, internal PATNO validation only.

Reuses the audited first-round training functions without changing them. There is
no test stage and no test Parquet in this experiment. A BiLSTM configuration is
selected as one complete three-head network. Original artifacts remain immutable.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

import model12_experiments as e
from model12_data import (FEATURES, Preprocessor, evaluation_rows, latest_rows,
                          make_dataset, reconstruct_targets, sha, write_json)


DEFAULT_OUTPUT = Path("experiments/model12_round2_20261023")


def fold_assignments(frame, n_folds, seed):
    patients = frame[["PATNO"]].drop_duplicates().sort_values("PATNO").reset_index(drop=True)
    labels = latest_rows(frame, 12).set_index("PATNO").prog_mcid_12m
    strata = patients.PATNO.map(labels).fillna(2).astype(int)
    assignments = np.empty(len(patients), dtype=int)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, held) in enumerate(splitter.split(patients, strata)):
        assignments[held] = fold
    patients["fold"] = assignments
    patients["stratum"] = strata
    return patients


def write_data(root, train, val):
    if set(train.PATNO) & set(val.PATNO):
        raise ValueError("Patient overlap in internal data")
    directory = root / "data"
    directory.mkdir(parents=True, exist_ok=False)
    for split, frame in [("train", train), ("val", val)]:
        frame.to_parquet(directory / f"{split}_longitudinal.parquet", index=False)
        evaluation_rows(frame).to_parquet(directory / f"{split}_evaluation.parquet", index=False)
        for h in [12, 24]:
            latest_rows(frame, h).to_parquet(directory / f"{split}_tabular_{h}m.parquet", index=False)


def prepare(root, config_path):
    config = e.read_json(config_path)
    source = Path(config["source_experiment"])
    if root.resolve() == source.resolve():
        raise ValueError("A distinct experiment directory is required")
    # Verify the existing audit, without loading or calculating any test targets.
    for name, expected in e.read_json(source / "protected_sha256.json").items():
        if sha(name) != expected:
            raise ValueError(f"Historical file changed since first audit: {name}")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config)
    write_json(root / "configuration_lock.json", {"sha256":sha(root / "config.json")})
    files = [p for p in source.rglob("*") if p.is_file()]
    files += [Path(p) for p in e.read_json(source / "protected_sha256.json")]
    files += [Path("src/model12_data.py"), Path("src/model12_experiments.py")]
    write_json(root / "protected_sha256.json", {str(p): sha(p) for p in files})
    # Read ONLY the corrected train and val; test remains unavailable to fitting.
    frames = {}
    for split in ["train", "val"]:
        path = source / "data" / f"{split}_longitudinal.parquet"
        expected = e.read_json(source / "prepared_sha256.json")[str(path.relative_to(source))]
        if sha(path) != expected:
            raise ValueError("Corrected source data changed")
        frames[split] = reconstruct_targets(pd.read_parquet(path))
    train, val = frames["train"], frames["val"]
    write_data(root / "final", train, val)
    assignments = fold_assignments(train, config["cv_folds"], config["cv_split_seed"])
    assignments.to_csv(root / "patient_folds.csv", index=False)
    counts = []
    for fold in range(config["cv_folds"]):
        held = set(assignments.loc[assignments.fold == fold, "PATNO"])
        tr, va = train[~train.PATNO.isin(held)].copy(), train[train.PATNO.isin(held)].copy()
        write_data(root / "cv" / str(fold), tr, va)
        counts.append(dict(fold=fold, train_patients=tr.PATNO.nunique(), val_patients=va.PATNO.nunique(),
                           train_prefixes=int(tr.target_np3tot_12m.notna().sum()),
                           val12=len(latest_rows(va, 12)), val24=len(latest_rows(va, 24))))
    pd.DataFrame(counts).to_csv(root / "fold_counts.csv", index=False)
    write_json(root / "data_sha256.json", {str(p.relative_to(root)):sha(p) for p in root.rglob("*.parquet")})
    write_json(root / "architecture.json", e.signature(e.model(config["lstm_defaults"])))
    for name in ["environment.json", "requirements-lock.txt"]:
        shutil.copyfile(source / name, root / name)
    write_json(root / "protocol.json", dict(test_evaluation=False, test_data_present=False,
        patient_partition="original train/val/test unchanged; inner folds use original train PATNO only",
        preprocessing="separately fitted inside each training fold",
        single_multitask_bilstm=True, baseline_correction="same aligned targets and raw residual as first round",
        selection=config["selection_policy"], raw_acquisition_dates_certified=False))
    print(pd.DataFrame(counts).to_string(index=False), flush=True)


def configs_xgb(config, task):
    baseline = e.xgb_configs({"xgb_trials": 0, "xgb_search_seed": 0}, task)[0]
    return [baseline, *[dict(baseline, **trial, early_stopping_rounds=60,
                             eval_metric="aucpr" if task == "cls" else "mae")
                        for trial in config["xgb_trials"]]]


def configs_lstm(config):
    return [dict(config["lstm_defaults"], **c) for c in config["lstm_trials"]]


def cv(root, config, family, fold):
    if (root / "cv_selection.json").exists():
        raise ValueError("Inner search frozen; do not retune after selection")
    folds = range(config["cv_folds"]) if fold is None else [fold]
    for i in folds:
        if i not in range(config["cv_folds"]):
            raise ValueError("Unknown fold")
        folder = root / "cv" / str(i)
        print(f"Internal fold {i}", flush=True)
        if family in ["all", "xgboost"]:
            for task in e.TASKS:
                for cfg in configs_xgb(config, task):
                    e.fit_xgb(folder, cfg, task, config["cv_training_seed"], config)
        if family in ["all", "bilstm"]:
            prepared = e.prepare_lstm(folder)
            for cfg in configs_lstm(config):
                e.fit_lstm(folder, cfg, config["cv_training_seed"], config, prepared)


def flatten(result, task):
    metric = e.task_metric(result, result["model"], task)
    return {k:v for k,v in metric.items() if not isinstance(v, list)}


def mean_metrics(values):
    return {key:float(np.mean([v[key] for v in values])) for key in values[0]}


def cv_results(root, config):
    rows, means = [], {"xgboost": {}, "bilstm": {}}
    for family in means:
        for task in e.TASKS:
            means[family][task] = {}
            configs = configs_xgb(config, task) if family == "xgboost" else configs_lstm(config)
            for cfg in configs:
                values = []
                for fold in range(config["cv_folds"]):
                    result = e.load_result(root / "cv" / str(fold), family, task, cfg["name"], config["cv_training_seed"])
                    m = flatten(result, task)
                    values.append(m)
                    rows.append(dict(model=family, task=task, config=cfg["name"], fold=fold, **m))
                means[family][task][cfg["name"]] = mean_metrics(values)
    pd.DataFrame(rows).to_csv(root / "cv_metrics.csv", index=False)
    write_json(root / "cv_mean_metrics.json", means)
    return means


def better_task(candidate, baseline, task):
    if task != "cls":
        return bool(candidate["mae"] < baseline["mae"] - 1e-6)
    return bool(e.rank_class(candidate) > e.rank_class(baseline) + 1e-6
                and candidate["balanced_accuracy"] >= baseline["balanced_accuracy"]
                and candidate["sensitivity"] >= .5 and candidate["specificity"] >= .5)


def joint_score(metrics, baseline):
    # Same output heads and objective; this only ranks whole-network runs.
    ratios = [metrics[h]["mae"] / baseline[h]["mae"] for h in ["12m", "24m"]]
    ratios.append((1-e.rank_class(metrics["cls"]))/(1-e.rank_class(baseline["cls"])))
    return float(np.mean(ratios))


def select_cv(root, config):
    if (root / "cv_selection.json").exists():
        raise ValueError("Selection already frozen")
    means = cv_results(root, config)
    choices = {"xgboost": {}, "bilstm": {}}
    for task in e.TASKS:
        values = means["xgboost"][task]
        names = [n for n in values if n != "baseline"]
        name = min(names, key=lambda n:e.loss_metric(values[n], task))
        choices["xgboost"][task] = dict(challenger=name,
            passed_cv=better_task(values[name], values["baseline"], task))
    values = means["bilstm"]
    baseline = {h:values[h]["baseline"] for h in e.TASKS}
    scores = {cfg["name"]:joint_score({h:values[h][cfg["name"]] for h in e.TASKS}, baseline)
              for cfg in configs_lstm(config) if cfg["name"] != "baseline"}
    name = min(scores, key=scores.get)
    choices["bilstm"] = dict(challenger=name, joint_scores=scores,
        passed_cv=all(better_task(values[h][name], baseline[h], h) for h in e.TASKS),
        passed_cv_by_task={h:better_task(values[h][name], baseline[h], h) for h in e.TASKS})
    write_json(root / "cv_selection.json", choices)
    write_json(root / "cv_freeze.json", dict(selection_sha256=sha(root / "cv_selection.json"),
        config_sha256=sha(root / "config.json"),
        checkpoints={str(p.relative_to(root)):sha(p) for p in (root / "cv").rglob("best.*")}))
    print(choices, flush=True)


def copy_reference(root, config, family, task, seed):
    source = Path(config["source_experiment"])
    src = e.folder_for(source, family, task, "baseline", seed)
    dest = e.folder_for(root / "final", family, task, "baseline", seed)
    if dest.exists():
        for p in src.glob("*"):
            if p.is_file() and (not (dest / p.name).exists() or sha(p) != sha(dest / p.name)):
                raise ValueError("Reused reference differs")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)


def finalize(root, config, family):
    if (root / "final_selection.json").exists():
        raise ValueError("Final results frozen; use verify")
    selection = e.read_json(root / "cv_selection.json")
    frozen = e.read_json(root / "cv_freeze.json")
    assert sha(root / "cv_selection.json") == frozen["selection_sha256"]
    assert sha(root / "config.json") == frozen["config_sha256"]
    final_root = root / "final"
    if family in ["all", "xgboost"]:
        for task in e.TASKS:
            cfg = next(c for c in configs_xgb(config, task) if c["name"] == selection["xgboost"][task]["challenger"])
            for seed in config["final_seeds"]:
                copy_reference(root, config, "xgboost", task, seed)
                e.fit_xgb(final_root, cfg, task, seed, config)
    if family in ["all", "bilstm"]:
        cfg = next(c for c in configs_lstm(config) if c["name"] == selection["bilstm"]["challenger"])
        prepared = e.prepare_lstm(final_root)
        for seed in config["final_seeds"]:
            copy_reference(root, config, "bilstm", "12m", seed)
            e.fit_lstm(final_root, cfg, seed, config, prepared)


def validation_intervals(root, config, choices):
    """Descriptive paired patient bootstrap, conditional on fitted models.

    Does not undo reuse of validation for epochs, thresholds or selection.
    """
    rng = np.random.default_rng(314159)
    intervals, cache = [], {}
    for family in ["xgboost", "bilstm"]:
        for task in e.TASKS:
            name = choices[family]["challenger"] if family == "bilstm" else choices[family][task]["challenger"]
            differences, expected_gains = [], []
            for seed in config["final_seeds"]:
                predictions, thresholds = [], []
                for cfg_name in ["baseline", name]:
                    # Replay full precision: a CSV rounding error at the decision
                    # threshold can change a binary prediction.
                    pred, threshold = e.predict_saved(root / "final", family, task, cfg_name, seed, "val", cache)
                    predictions.append(pred.sort_values(["PATNO", "row_id"]).reset_index(drop=True))
                    thresholds.append(threshold)
                base, candidate = predictions
                assert base[["PATNO", "row_id"]].equals(candidate[["PATNO", "row_id"]])
                # Require identical numeric values, regardless of storage dtype.
                assert np.array_equal(base.truth.to_numpy(), candidate.truth.to_numpy())
                truth = base.truth.to_numpy(dtype=float)
                base_pred, candidate_pred = [p.prediction.to_numpy(dtype=float) for p in predictions]
                if task != "cls":
                    differences.append(abs(base_pred-truth)-abs(candidate_pred-truth))
                else:
                    a = ((base_pred >= thresholds[0]) == truth).astype(float)
                    b = ((candidate_pred >= thresholds[1]) == truth).astype(float)
                    differences.append(b-a)
                metrics = [e.task_metric(e.load_result(root / "final", family, task, n, seed), family, task)
                           for n in ["baseline", name]]
                expected_gains.append(metrics[1]["balanced_accuracy"]-metrics[0]["balanced_accuracy"] if task == "cls"
                                      else metrics[0]["mae"]-metrics[1]["mae"])
            delta = np.mean(differences, axis=0)
            if task == "cls":
                neg, pos = delta[base.truth == 0], delta[base.truth == 1]
                samples = (neg[rng.integers(0, len(neg), (2000, len(neg)))].mean(1)
                           + pos[rng.integers(0, len(pos), (2000, len(pos)))].mean(1))/2
                gain = (neg.mean()+pos.mean())/2
            else:
                samples = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
                gain = delta.mean()
            assert np.isclose(gain, np.mean(expected_gains), atol=1e-10, rtol=0)
            lo, hi = np.quantile(samples, [.025, .975])
            intervals.append(dict(model=family, task=task, n=len(delta), gain=gain, ci95_low=lo, ci95_high=hi,
                                  metric="MAE reduction" if task != "cls" else "balanced accuracy increase"))
    result = pd.DataFrame(intervals)
    result.to_csv(root / "validation_paired_intervals.csv", index=False)
    return result


def report(root, config):
    cv_selection = e.read_json(root / "cv_selection.json")
    rows, decisions = [], {"xgboost": {}, "bilstm": {}}
    for family in decisions:
        challenger = cv_selection[family].get("challenger")
        for task in e.TASKS:
            name = challenger if family == "bilstm" else cv_selection[family][task]["challenger"]
            means = {}
            for role, cfg_name in [("baseline", "baseline"), ("challenger", name)]:
                values = []
                for seed in config["final_seeds"]:
                    result = e.load_result(root / "final", family, task, cfg_name, seed)
                    m = flatten(result, task)
                    values.append(m)
                    rows.append(dict(model=family, task=task, role=role, config=cfg_name, seed=seed,
                                     policy="validation_threshold" if task == "cls" else "na", **m))
                    if task == "cls":
                        fixed = result["metrics"]["cls_at_05"] if family == "bilstm" else result["metrics_at_05"]
                        rows.append(dict(model=family, task=task, role=role, config=cfg_name, seed=seed,
                                         policy="fixed_0.5", **{k:v for k,v in fixed.items() if not isinstance(v,list)}))
                means[role] = mean_metrics(values)
            decisions[family][task] = dict(challenger=name, means=means,
                passed_original_validation=better_task(means["challenger"], means["baseline"], task))
    # A single BiLSTM is retained or rejected as a whole; never mix its heads.
    bilstm_pass = cv_selection["bilstm"]["passed_cv"] and all(d["passed_original_validation"] for d in decisions["bilstm"].values())
    retained = {"bilstm": cv_selection["bilstm"]["challenger"] if bilstm_pass else "baseline", "xgboost": {}}
    for task in e.TASKS:
        d = decisions["xgboost"][task]
        retained["xgboost"][task] = d["challenger"] if cv_selection["xgboost"][task]["passed_cv"] and d["passed_original_validation"] else "baseline"
    selection = dict(decisions=decisions, retained=retained, seeds=config["final_seeds"],
                     test_evaluated=False, one_complete_bilstm=True)
    if (root / "final_selection.json").exists():
        assert e.read_json(root / "final_selection.json") == selection
    else:
        write_json(root / "final_selection.json", selection)
    full = pd.DataFrame(rows)
    full.to_csv(root / "validation_metrics_by_seed.csv", index=False)
    keys = ["model", "task", "role", "config", "policy"]
    numeric = full.select_dtypes(include="number").columns.difference(["seed"])
    means = full.groupby(keys)[numeric].mean().reset_index()
    means.to_csv(root / "validation_before_after_mean.csv", index=False)
    full.groupby(keys)[numeric].std().to_csv(root / "validation_before_after_std.csv")
    intervals = validation_intervals(root, config, cv_selection)
    # Checkpoint manifests contain actual thresholds, never mean thresholds.
    manifests = []
    for family in ["xgboost", "bilstm"]:
        for seed in config["final_seeds"]:
            for task in (["all_three_heads"] if family == "bilstm" else e.TASKS):
                name = retained[family] if family == "bilstm" else retained[family][task]
                result = e.load_result(root / "final", family, "12m" if family == "bilstm" else task, name, seed)
                folder = e.folder_for(root / "final", family, task, name, seed)
                manifests.append(dict(model=family, task=task, config=name, seed=seed,
                    checkpoint=str(folder / ("best.pt" if family == "bilstm" else "best.json")),
                    classification_threshold=result["threshold"] if family == "bilstm" or task == "cls" else None))
    write_json(root / "retained_checkpoints.json", manifests)
    final_freeze = dict(selection_sha256=sha(root / "final_selection.json"),
        checkpoints={str(p.relative_to(root)):sha(p) for p in (root / "final" / "runs").rglob("best.*")})
    if (root / "final_freeze.json").exists():
        assert e.read_json(root / "final_freeze.json") == final_freeze
    else:
        write_json(root / "final_freeze.json", final_freeze)
    sources = [Path(p) for p in ["src/model12_round2.py", "src/test_model12_round2.py", "src/model12_data.py", "src/model12_experiments.py", "src/train_model1_xgboost.py", "src/train_model2_lstm.py", "src/evaluate.py", "configs/model12_round2.json"]]
    for p in sources:
        target = root / "source_snapshot" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    write_json(root / "source_sha256.json", {str(p):sha(p) for p in sources})
    cvtable = pd.read_csv(root / "cv_metrics.csv")
    cvsummary = cvtable.groupby(["model", "task", "config"])[["mae", "roc_auc", "average_precision", "balanced_accuracy"]].mean().reset_index()
    cvsummary.to_csv(root / "cv_before_after_mean.csv", index=False)
    reg = means[means.task != "cls"][["model", "task", "role", "config", "n", "mae", "rmse", "r2", "bias"]]
    cls = means[(means.task == "cls") & (means.policy == "validation_threshold")][["model", "role", "config", "n", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc", "average_precision", "brier_score"]]
    conf = full[full.task == "cls"][["model", "role", "seed", "policy", "threshold", "tn", "fp", "fn", "tp"]]
    counts = len(list((root / "cv").rglob("metrics.json"))) + len(list((root / "final" / "runs").rglob("metrics.json")))
    reused = 4 * len(config["final_seeds"])
    interruptions = list(root.rglob("interruption.json"))
    interruption_note = (f"{len(interruptions)} essai(s) interrompu(s) par une erreur d'écriture Windows, "
                         "repris à partir de l'époque 1 avec les mêmes réglages et graine. "
                         "Les traces interruption.json et poids partiels sont archivés; "
                         "seuls les essais terminés sont comparés." if interruptions else "")
    summary = []
    for family in ["xgboost", "bilstm"]:
        for task in e.TASKS:
            d = decisions[family][task]
            before, after = d["means"]["baseline"], d["means"]["challenger"]
            kept = retained[family] if family == "bilstm" else retained[family][task]
            if task == "cls":
                scores = f"balanced accuracy {before['balanced_accuracy']:.4f} -> {after['balanced_accuracy']:.4f}; ROC-AUC {before['roc_auc']:.4f} -> {after['roc_auc']:.4f}; AP {before['average_precision']:.4f} -> {after['average_precision']:.4f}"
            else:
                scores = f"MAE {before['mae']:.4f} -> {after['mae']:.4f}"
            summary.append(f"- {family} {task} : {scores}. Configuration conservée : {kept}.")
    text = f"""DEUXIÈME SÉRIE XGBOOST / BILSTM — 23 OCTOBRE 2026

Poursuite du prompt initial, mêmes architectures et mêmes variables.
{counts-reused} nouveaux entraînements terminés; {reused} références de la première
série réutilisées à l'identique (3 régressions/classification XGB + 1 BiLSTM,
pour chacune des trois graines). Les résultats de TEST ne sont pas recalculés.
Les résultats ci-dessous sont de développement, et non une validation externe.
{interruption_note}

RÉSULTATS PRINCIPAUX SUR VALIDATION CANONIQUE (moyenne des trois graines) :
{chr(10).join(summary)}
Pour le BiLSTM, une amélioration isolée d'une tête ne suffit pas : la configuration
est conservée ou rejetée pour les trois sorties ensemble. Les références restent
disponibles, tous les essais infructueux sont archivés. Un gain numérique de
validation n'est pas présenté comme une preuve de progrès sur un test indépendant.

1. Ce qui reste strictement identique

XGBoost : 42 entrées, régressions NP3TOT 12/24m, classification aggravation >=3,5.
Objectifs reg:squarederror / binary:logistic, aucun changement de cible ni de
variable, pas de nouvelle modalité. Réglages des arbres uniquement.
BiLSTM : 44 entrées, projection 44->64; LSTM bidirectionnel, 2 couches de largeur64;
attention additive, concaténation contexte/dernière visite, trois têtes 128->64->1,
deux résidus NP3TOT brut. 243 651 paramètres; classe et forward originaux importés
sans modification. Un checkpoint COMPLET pour les trois sorties, par graine.
Dropout déjà présent réglé, aucune couche ajoutée/retirée et aucune fusion modifiée.

L'audit initial a confirmé le mélange de visites avec groupby(PATNO).last() :
2 075 cibles 24m sur 2 155 dans train et 435 sur 449 dans validation provenaient
d'une autre visite que la ligne index. La sélection de lignes exactes corrige ce
problème dans des fichiers distincts, partagés par référence et candidats.
Le résidu BiLSTM utilisait aussi le score standardisé avec une cible en points
bruts : le NP3TOT brut est désormais conservé séparément des entrées standardisées.
Ces deux corrections précèdent la comparaison des hyperparamètres.

Le protocole audité est conservé : dernières lignes exactes par horizon, cibles
non propagées entre visites, préfixes bornés à la ligne index, padding exclu,
score brut séparé de la standardisation, masques des cibles absentes, sauvegarde
puis rechargement des meilleurs poids. Le packing, le masque d'attention et la
sauvegarde immédiate des poids du code d'origine étaient déjà corrects.
Les cibles longitudinales sont
à nouveau reconstruites et vérifiées avant préparation. À 24m, visite index plus
ancienne si nécessaire; mêmes patients ET mêmes cibles pour toutes les comparaisons.

2. Protocole de cette poursuite

Trois plis stratifiés par PATNO, tirés uniquement dans l'entraînement canonique.
Strates : aggravation 12m à la dernière visite éligible, absence de label à part.
PATNO disjoints entre apprentissage et validation de chaque pli; aucun patient
de validation canonique ou de test dans les plis. Chaque patient train est une
fois dans une validation interne. patient_folds.csv permet de refaire les plis.
Les médianes, scaler et poids de classe sont recalculés dans le train de CHAQUE
pli, jamais sur l'ensemble avant partition. La LOCF demeure causale.

{pd.read_csv(root / 'fold_counts.csv').to_string(index=False)}

La validation interne sert à l'early stopping, au seuil et au choix du candidat.
Ce n'est pas une validation croisée imbriquée : les scores internes sont donc
optimistes pour le réglage choisi. Ils mesurent la stabilité du développement.
La validation canonique, déjà utilisée par la première série, sert ensuite à
l'early stopping et au seuil des trois graines finales; elle n'est PAS un test
vierge ni une confirmation indépendante. Aucun ancien score de test n'entre dans
une fonction de sélection. La reprise après connaissance des anciens tests peut
néanmoins induire un biais adaptatif global : de nouveaux patients restent requis.

XGB : 6 candidats + référence, 3 tâches, 3 plis; candidats de profondeur2/3/4,
min_child_weight4–20, learning_rate .015–.05, alpha/lambda/gamma et échantillonnage
contrôlés. Deux configurations issues des anciennes sélections de VALIDATION
sont réexaminées. Plafonds 700–1800 arbres, arrêt anticipé validation.
BiLSTM : référence corrigée + 3 candidats (slow_huber, slow_mae, moderate_balanced).
Learning rates plus faibles, weight decay .003–.005, dropout .30–.35, batch64/128,
clipping1/1.5, Huber beta2 ou MAE; poids des tâches 1/1/.4–.6, BCE pondérée ou non.
Scheduler ReduceLROnPlateau patience2, maximum28 époques, arrêt patience6 pour les
candidats; référence conserve ses réglages initiaux. Voir config.json pour tout.

Le challenger XGB est choisi par tâche selon MAE ou moyenne ROC-AUC/AP des plis.
Le challenger BiLSTM est UN réseau choisi sur la moyenne de trois pertes
normalisées par la référence : MAE12, MAE24, 1-moyenne(ROC-AUC, AP).
La sélection est gelée dans cv_selection.json AVANT les réentraînements finaux.
Graines finales : {config['final_seeds']}. Aucune sélection de la meilleure graine.
Les matrices, seuils et métriques individuels sont sauvegardés; les tableaux
présentent des moyennes de métriques, jamais un ensemble de prédictions.

Règle conservatrice de conservation : XGB doit progresser à la fois en validation
interne et en moyenne des graines de validation canonique. Pour la classification,
ROC-AUC/AP et balanced accuracy doivent progresser/ne pas baisser et sensibilité/
spécificité rester >=.5. BiLSTM doit satisfaire ces règles pour LES TROIS sorties
avec la même configuration, et réduire les deux MAE; sinon référence conservée.
Les essais sans gain restent disponibles. Aucun remplacement des anciens modèles.

3. Moyennes des trois plis internes

{cvsummary.round(5).to_string(index=False)}

Choix avant validation canonique :
{cv_selection}

4. Avant/après sur la validation canonique — trois graines, mêmes patients

Régression (574 patients à 12m, 449 à 24m; scores en points NP3TOT) :
{reg.round(5).to_string(index=False)}

Classification, seuil choisi en validation pour chaque graine :
{cls.round(5).to_string(index=False)}

Matrices [[TN,FP],[FN,TP]], avec seuil validation et contrôle fixe .5 :
{conf.round(6).to_string(index=False)}

Configurations conservées selon la règle fixée avant les essais :
{retained}

Les quantiles des probabilités et les poids positifs sont dans metrics.json de
chaque run; les écarts types entre graines sont dans validation_before_after_std.csv.
Les effets du seuil se distinguent des effets de l'apprentissage : le seuil ne
modifie ni ROC-AUC, ni AP, ni Brier. La pondération peut détériorer la calibration.
Le Brier et le compromis sensibilité/spécificité sont toujours montrés.

Différences appariées sur les mêmes patients; bootstrap 2 000 tirages, graine314159.
Positif = gain (baisse MAE ou hausse balanced accuracy), négatif = régression :
{intervals.round(5).to_string(index=False)}
Ces intervalles sont descriptifs, conditionnels aux modèles entraînés : ils ne
corrigent PAS l'optimisme dû aux époques/seuils/sélections appris sur validation.
Ils ne servent pas à changer rétroactivement les règles de sélection.

5. Limites et confirmation indépendante

Pas de nouveau résultat de test dans cette série. Le test historique déjà consulté
est conservé hors de ce pipeline. Ni les gains de validation ni les plis internes
ne prouvent un gain sur de nouveaux patients. Pour le confirmer : protocole,
configurations et seuils figés, nouvelle extraction/cohorte avec PATNO jamais
utilisés; éventuellement une validation croisée imbriquée en complément interne.
Les cohortes d'origine sont conservées; ne pas assimiler toute la cohorte à PD.
data/processed suffit aux expériences; sans raw ni dates d'acquisition des modalités,
la provenance temporelle des jointures EVENT_ID/démographie reste non certifiable.
Les sources et expériences précédentes restent inchangées, vérifiées par SHA256.

6. Commandes PowerShell (Windows / terminal PowerShell de VS Code)

Depuis C:\\Users\\MSI\\Desktop\\parkinson :
.\\.venv\\Scripts\\python.exe -m unittest discover -s src -p test_model12_round2.py -v
.\\.venv\\Scripts\\python.exe src/model12_round2.py --stage verify --output {root.as_posix()}

Nouvelle exécution complète (choisir un dossier inexistant) :
$run = 'experiments/model12_round2_reproduction'
.\\.venv\\Scripts\\python.exe src/model12_round2.py --stage prepare --output $run
.\\.venv\\Scripts\\python.exe -u src/model12_round2.py --stage cv --output $run
.\\.venv\\Scripts\\python.exe src/model12_round2.py --stage select --output $run
.\\.venv\\Scripts\\python.exe -u src/model12_round2.py --stage final --output $run
.\\.venv\\Scripts\\python.exe src/model12_round2.py --stage report --output $run
.\\.venv\\Scripts\\python.exe src/model12_round2.py --stage verify --output $run

--family xgboost/bilstm et --fold 0/1/2 permettent d'exécuter les plis séparément.
Ne jamais lancer simultanément deux processus sur le même pli et la même famille.
cv/final reprennent les expériences terminées; les sélections gelées bloquent le
réglage ultérieur dans ce dossier. Les références finales sont copiées à l'identique
depuis la première série; celle-ci doit être présente pour reproduire cette reprise.
Les références internes sont bien réentraînées uniquement sur leur propre pli.

Livrables : config.json, architecture.json, patient_folds.csv, cv_metrics.csv,
cv_mean_metrics.json, cv_selection.json, validation_before_after_mean.csv,
validation_before_after_std.csv, validation_metrics_by_seed.csv, final_selection.json,
retained_checkpoints.json, cv/ et final/ (poids/configs/prédictions/courbes),
verification.json, source_snapshot/, requirements-lock.txt et environment.json.
"""
    (root / "RAPPORT_FR.txt").write_text(text, encoding="utf-8")
    print(reg.round(5).to_string(index=False))
    print(cls.round(5).to_string(index=False))
    print("Retained:", retained)


def verify(root, config):
    for path, expected in e.read_json(root / "protected_sha256.json").items():
        if sha(path) != expected:
            raise ValueError(f"Protected artifact changed: {path}")
    for path, expected in e.read_json(root / "data_sha256.json").items():
        assert sha(root / path) == expected
    for path, expected in e.read_json(root / "source_sha256.json").items():
        assert sha(path) == expected
        assert sha(root / "source_snapshot" / path) == expected
    assert not list(root.rglob("test*.parquet"))
    for freeze_name, selection_name in [("cv_freeze.json", "cv_selection.json"), ("final_freeze.json", "final_selection.json")]:
        frozen = e.read_json(root / freeze_name)
        assert sha(root / selection_name) == frozen["selection_sha256"]
        for path, expected in frozen["checkpoints"].items():
            assert sha(root / path) == expected
    assignment = pd.read_csv(root / "patient_folds.csv")
    assert not assignment.PATNO.duplicated().any()
    for fold in range(config["cv_folds"]):
        tr, va = [e.load_frame(root / "cv" / str(fold), split) for split in ["train", "val"]]
        assert not set(tr.PATNO) & set(va.PATNO)
        assert set(va.PATNO) == set(assignment.loc[assignment.fold == fold, "PATNO"])
        # Recompute a fold preprocessor, verify ALL saved fold checkpoints used it.
        expected_prep = Preprocessor().fit(tr).state()
        for p in (root / "cv" / str(fold)).rglob("best.pt"):
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            assert ckpt["preprocessor"] == expected_prep
            assert ckpt["architecture"] == e.signature(e.model(config["lstm_defaults"]))
    final_root, cache, max_error = root / "final", {}, 0.
    final_prep = Preprocessor().fit(e.load_frame(final_root, "train")).state()
    expected_architecture = e.signature(e.model(config["lstm_defaults"]))
    for path in (final_root / "runs" / "bilstm").rglob("best.pt"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["preprocessor"] == final_prep
        assert checkpoint["architecture"] == expected_architecture
    selection = e.read_json(root / "cv_selection.json")
    for family in ["xgboost", "bilstm"]:
        for task in e.TASKS:
            challenger = selection[family]["challenger"] if family == "bilstm" else selection[family][task]["challenger"]
            for name in ["baseline", challenger]:
                for seed in config["final_seeds"]:
                    pred, threshold = e.predict_saved(final_root, family, task, name, seed, "val", cache)
                    folder = e.folder_for(final_root, family, task, name, seed)
                    saved = pd.read_csv(folder / "validation_predictions.csv")
                    if family == "bilstm":
                        saved = saved[saved.eval24 if task == "24m" else saved.eval12]
                        column = "prob_mcid" if task == "cls" else f"pred_{task}"
                    else:
                        column = "prediction"
                    assert pred.PATNO.tolist() == saved.PATNO.tolist()
                    error = float(np.max(abs(pred.prediction.to_numpy()-saved[column].to_numpy())))
                    assert error < 1e-5  # Includes original float32 CSV serialization.
                    max_error = max(max_error, error)
                    recorded = e.task_metric(e.load_result(final_root, family, task, name, seed), family, task)
                    replayed = (e.classification(pred.truth, pred.prediction, threshold) if task == "cls"
                                else e.regression(pred.truth, pred.prediction))
                    for key, value in replayed.items():
                        if not isinstance(value, list):
                            assert np.isclose(value, recorded[key], atol=1e-10, rtol=0), (family, task, name, seed, key)
                    if task == "cls":
                        recalculated, _ = e.choose_threshold(pred.truth, pred.prediction)
                        assert abs(recalculated-threshold) < 1e-10
    write_json(root / "verification.json", dict(protected_artifacts_unchanged=True,
        source_snapshot_verified=True, final_preprocessing_fitted_on_train=True,
        saved_metrics_reproduced=True,
        patient_folds_disjoint=True, preprocessing_fitted_per_fold=True,
        architecture_identical=True, parameter_count=e.signature(e.model(config["lstm_defaults"]))["count"],
        single_complete_bilstm=True, test_evaluated=False, test_data_absent=True,
        replay_max_abs_error=max_error))
    print("Round 2 verification passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "cv", "select", "final", "report", "verify"], required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=Path("configs/model12_round2.json"))
    parser.add_argument("--family", choices=["all", "xgboost", "bilstm"], default="all")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output, args.config)
        return
    config = e.read_json(args.output / "config.json")
    assert sha(args.output / "config.json") == e.read_json(args.output / "configuration_lock.json")["sha256"]
    e.seed_all(config["cv_training_seed"], config["threads"])
    if args.stage == "cv":
        cv(args.output, config, args.family, args.fold)
    elif args.stage == "select":
        select_cv(args.output, config)
    elif args.stage == "final":
        finalize(args.output, config, args.family)
    elif args.stage == "report":
        report(args.output, config)
    else:
        verify(args.output, config)


if __name__ == "__main__":
    main()
