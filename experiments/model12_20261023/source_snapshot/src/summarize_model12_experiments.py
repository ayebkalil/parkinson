"""Export auditable tables, paired uncertainty and a French experiment report."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

import model12_experiments as e
from model12_data import STATIC_FEATURES, Preprocessor, make_dataset, sha, write_json


def historical_diagnostic(root):
    checkpoint = torch.load("models/model2_bilstm.pt", map_location="cpu", weights_only=False)
    prep = Preprocessor.restore(dict(feature_cols=checkpoint["feature_cols"], medians=checkpoint["train_medians"],
                                    mean=checkpoint["scaler_mean"], scale=checkpoint["scaler_scale"]))
    rows = e.load_frame(root, "val", "evaluation")
    ds = make_dataset(e.load_frame(root, "val"), prep, rows)
    net = e.model(dict(dropout=.2))
    net.load_state_dict(checkpoint["model_state_dict"])
    pred = e.predict_lstm(net, e.loader(ds, 256), rows)
    sub = pred[pred.eval12]
    threshold, _ = e.choose_threshold(sub.true_mcid, sub.prob_mcid)
    result = dict(checkpoint="models/model2_bilstm.pt", epoch=checkpoint["epoch"],
                  note="Validation only. Classification does not depend on the raw residual anchor. No regression comparison using old weights with a changed anchor.",
                  at_05=e.classification(sub.true_mcid, sub.prob_mcid, .5),
                  validation_threshold=e.classification(sub.true_mcid, sub.prob_mcid, threshold))
    write_json(root / "historical_classifier_diagnostic.json", result)
    return result


def validation_table(root):
    rows = []
    for path in (root / "runs").rglob("metrics.json"):
        result = e.read_json(path)
        for task in e.TASKS if result["model"] == "bilstm" else [result["task"]]:
            metric = e.task_metric(result, result["model"], task)
            rows.append(dict(model=result["model"], task=task, config=result["name"], seed=result["seed"],
                             epoch_or_tree=result.get("epoch", result.get("best_iteration")),
                             **{k:v for k,v in metric.items() if not isinstance(v, list)}))
    df = pd.DataFrame(rows).sort_values(["model", "task", "config", "seed"])
    df.to_csv(root / "validation_all_experiments.csv", index=False)
    return df


def paired_intervals(root, selection):
    """Patient bootstrap of mean per-seed errors, NOT an averaged predictor.

    Intervals condition on these three fitted seeds and a reused historical test.
    They are descriptive, not independent confirmatory statistical tests.
    """
    rng = np.random.default_rng(901)
    rows = []
    for family in ["xgboost", "bilstm"]:
        for task in e.TASKS:
            differences = []
            for seed in selection["seeds"]:
                a, b = [pd.read_csv(root / "test_predictions" / f"{family}_{task}_{role}_{seed}.csv").sort_values(["PATNO", "row_id"]).reset_index(drop=True) for role in ["baseline", "selected"]]
                assert a[["PATNO", "row_id", "truth"]].equals(b[["PATNO", "row_id", "truth"]])
                if task != "cls":
                    differences.append(abs(a.prediction-a.truth)-abs(b.prediction-b.truth))
                else:
                    base_result = e.load_result(root, family, task, "baseline", seed)
                    name = selection["decisions"][family][task]["selected"]
                    new_result = e.load_result(root, family, task, name, seed)
                    ca = (a.prediction >= base_result["threshold"]).to_numpy() == a.truth.to_numpy()
                    cb = (b.prediction >= new_result["threshold"]).to_numpy() == b.truth.to_numpy()
                    differences.append(cb.astype(float)-ca.astype(float))
            delta = np.mean(differences, axis=0)
            if task != "cls":
                simulations = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(axis=1)
                gain = delta.mean()
            else:
                neg, pos = delta[a.truth == 0], delta[a.truth == 1]
                simulations = (neg[rng.integers(0, len(neg), size=(2000, len(neg)))].mean(axis=1)
                               + pos[rng.integers(0, len(pos), size=(2000, len(pos)))].mean(axis=1))/2
                gain = (neg.mean()+pos.mean())/2
            low, high = np.quantile(simulations, [.025, .975])
            rows.append(dict(model=family, task=task, n=len(delta), gain=float(gain), ci95_low=low, ci95_high=high,
                             metric="MAE reduction" if task != "cls" else "balanced accuracy increase"))
    result = pd.DataFrame(rows)
    result.to_csv(root / "paired_patient_intervals.csv", index=False)
    return result


def same_index_diagnostic(root, selection):
    """Secondary 24m analysis at the exact historical 12m index, valid targets only."""
    metrics = []
    for split in ["val", "test"]:
        index_rows = e.load_frame(root, split, "tabular_12m").dropna(subset=["target_np3tot_24m"])
        for family in ["xgboost", "bilstm"]:
            for seed in selection["seeds"]:
                for role, name in [("baseline", "baseline"), ("selected", selection["decisions"][family]["24m"]["selected"])]:
                    folder = e.folder_for(root, family, "24m", name, seed)
                    if family == "xgboost":
                        net = xgb.XGBRegressor()
                        net.load_model(folder / "best.json")
                        pred = net.predict(index_rows[STATIC_FEATURES])
                    else:
                        checkpoint = torch.load(folder / "best.pt", weights_only=False, map_location="cpu")
                        net = e.model(checkpoint["config"])
                        net.load_state_dict(checkpoint["model_state_dict"])
                        prep = Preprocessor.restore(checkpoint["preprocessor"])
                        rows = index_rows.assign(eval12=True, eval24=True)
                        ds = make_dataset(e.load_frame(root, split), prep, rows)
                        pred = e.predict_lstm(net, e.loader(ds, 256), rows).pred_24m.to_numpy()
                    metrics.append(dict(split=split, model=family, seed=seed, role=role,
                                        **e.regression(index_rows.target_np3tot_24m, pred)))
    result = pd.DataFrame(metrics)
    result.to_csv(root / "same_historical_index_24m.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/model12_20261023"))
    args = parser.parse_args()
    root = args.output
    config, selection = e.read_json(root / "config.json"), e.read_json(root / "selection.json")
    e.seed_all(42, config["threads"])
    validation_table(root)
    old = historical_diagnostic(root)
    test = pd.read_csv(root / "test_metrics.csv")
    full_bilstm = pd.read_csv(root / "bilstm_full_checkpoint_test_metrics.csv")
    full_bilstm = full_bilstm.groupby(["config", "task"])[["mae", "roc_auc", "average_precision", "balanced_accuracy"]].mean().reset_index()
    persistence = []
    for split in ["val", "test"]:
        for h in [12, 24]:
            frame = e.load_frame(root, split, f"tabular_{h}m")
            persistence.append(dict(split=split, horizon=h,
                                    **e.regression(frame[f"target_np3tot_{h}m"], frame.NP3TOT)))
    persistence = pd.DataFrame(persistence)
    persistence.to_csv(root / "persistence_reference.csv", index=False)
    intervals = paired_intervals(root, selection)
    same_index_diagnostic(root, selection)
    numerical = [c for c in ["n", "mae", "rmse", "r2", "bias", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc", "average_precision", "brier_score", "threshold"] if c in test]
    means = test.groupby(["model", "task", "role", "config", "policy"], dropna=False)[numerical].mean().reset_index()
    means.to_csv(root / "before_after_test_mean.csv", index=False)
    deviations = test.groupby(["model", "task", "role", "config", "policy"], dropna=False)[numerical].std().reset_index()
    deviations.to_csv(root / "before_after_test_std.csv", index=False)
    # Persist all configurations, epochs, thresholds and paths, never best seed only.
    selected, references = [], []
    for family in ["xgboost", "bilstm"]:
        for task in e.TASKS:
            name = selection["decisions"][family][task]["selected"]
            for seed in selection["seeds"]:
                result = e.load_result(root, family, task, name, seed)
                folder = e.folder_for(root, family, task, name, seed)
                selected.append(dict(model=family, task=task, config=name, seed=seed,
                                     path=str(folder / ("best.pt" if family == "bilstm" else "best.json")),
                                     threshold=result["threshold"] if task == "cls" else None,
                                     selection_basis="validation only; not a deployment recommendation"))
                baseline = e.load_result(root, family, task, "baseline", seed)
                base_folder = e.folder_for(root, family, task, "baseline", seed)
                references.append(dict(model=family, task=task, seed=seed,
                                       path=str(base_folder / ("best.pt" if family == "bilstm" else "best.json")),
                                       threshold=baseline["threshold"] if task == "cls" else None))
    write_json(root / "selected_checkpoints.json", selected)
    write_json(root / "retained_reference_checkpoints.json", references)
    conclusion = []
    for row in intervals.itertuples(index=False):
        conclusion.append(dict(model=row.model, task=row.task, test_gain=row.gain,
                               ci95=[row.ci95_low, row.ci95_high],
                               test_gain_confirmed=bool(row.ci95_low > 0),
                               automatic_replacement_of_reference=False))
    write_json(root / "test_conclusions.json", conclusion)
    sources = [Path(p) for p in ["src/model12_data.py", "src/model12_experiments.py", "src/test_model12_experiments.py", "src/summarize_model12_experiments.py", "src/train_model1_xgboost.py", "src/train_model2_lstm.py", "src/evaluate.py", "src/build_multimodal_dataset.py", "src/create_patient_split.py", "configs/model12_experiments.json"]]
    snapshot = root / "source_snapshot"
    for path in sources:
        (snapshot / path.parent).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot / path)
    write_json(root / "source_sha256.json", {str(p):sha(p) for p in sources})
    audit = pd.read_csv(root / "target_audit.csv")
    regression = means[means.task != "cls"][["model", "task", "role", "config", "n", "mae", "rmse", "r2", "bias"]]
    cls = means[(means.task == "cls") & (means.policy == "validation_threshold")][["model", "role", "config", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc", "average_precision", "brier_score", "threshold"]]
    fixed = means[(means.task == "cls") & (means.policy == "fixed_0.5")][["model", "role", "accuracy", "balanced_accuracy", "sensitivity", "specificity"]]
    conf = test[test.task == "cls"][["model", "role", "seed", "policy", "threshold", "tn", "fp", "fn", "tp"]]
    val_final = []
    for family in ["xgboost", "bilstm"]:
        for task in e.TASKS:
            decision = selection["decisions"][family][task]
            for name, metrics in decision["validation_mean"].items():
                val_final.append(dict(model=family, task=task, config=name, selected=name == decision["selected"], **metrics))
    val_final = pd.DataFrame(val_final)
    val_final.to_csv(root / "validation_finalists_mean.csv", index=False)
    failure_lines = []
    for family in ["xgboost", "bilstm"]:
        for task in e.TASKS:
            dec = selection["decisions"][family][task]
            failure_lines.append(f"- {family} / {task} : candidat {dec['candidate']}, retenu {dec['selected']}; gain validation multigraines={dec['gained']}.")
    params = e.read_json(root / "architecture.json")["count"]
    elapsed = sum(e.read_json(p).get("elapsed_seconds", 0) for p in (root / "runs").rglob("metrics.json"))
    def test_mean(family, task, role, metric):
        part = means[(means.model == family) & (means.task == task) & (means.role == role)]
        if task == "cls":
            part = part[part.policy == "validation_threshold"]
        return float(part.iloc[0][metric])
    x24_before, x24_after = [test_mean("xgboost", "24m", role, "mae") for role in ["baseline", "selected"]]
    improvement = intervals[(intervals.model == "xgboost") & (intervals.task == "24m")].iloc[0]
    cls_before, cls_after = [test_mean("bilstm", "cls", role, "balanced_accuracy") for role in ["baseline", "selected"]]
    n_runs = len(list((root / "runs").rglob("metrics.json")))
    report = f"""XGBOOST ET BILSTM — EXPÉRIENCES DU 23 OCTOBRE 2026

Les résultats ci-dessous sont mesurés. Ils ne constituent pas une validation
clinique indépendante : le test historique a déjà été consulté dans ce projet.
Aucune donnée commune, aucun ancien checkpoint et aucun code de fusion n'a été
modifié par ces expériences. Les changements déjà présents au début sont conservés.

RÉSULTAT PRINCIPAL : {n_runs} entraînements exécutés, trois graines pour chaque
configuration finaliste. Aucun gain global robuste des hyperparamètres n'est
démontré sur ce test. XGB 24m passe de {x24_before:.3f} à {x24_after:.3f} de MAE,
mais le gain de {improvement['gain']:.3f} point a un intervalle patient à 95 %
[{improvement['ci95_low']:.3f}; {improvement['ci95_high']:.3f}] qui couvre zéro.
Les candidats BiLSTM ne battent pas la référence corrigée sur le test. Leur
classification passe de {cls_before:.4f} à {cls_after:.4f} de balanced accuracy.
Les gains de validation ne se sont donc pas généralisés.

Les références corrigées sont conservées dans retained_reference_checkpoints.json;
aucun candidat ne les remplace automatiquement. selected_checkpoints.json décrit
uniquement la sélection faite sur VALIDATION AVANT le test, conservée telle quelle
pour la traçabilité; ce n'est pas une recommandation de déploiement. Il n'y a eu
ni nouvel essai ni resélection après lecture du test. Les corrections de données,
d'unités et d'évaluation sont utiles même en l'absence de gain d'hyperparamètres.

1. Audit et correction des données

{audit.to_string(index=False)}

Le problème groupby(PATNO).last() est confirmé : chaque colonne prend sa dernière
valeur non nulle, et les cibles 24m peuvent provenir d'une autre visite. Les
cibles 12m des tables historiques concordent avec les lignes longitudinales.
Les lignes tabulaires historiques mélangent également des valeurs de variables
prises à des visites antérieures. Cette mémoire implicite n'est pas une preuve
de fuite future; elle est remplacée par une sélection de ligne exacte pour XGB.

Le nouveau protocole principal utilise la dernière ligne complète éligible par
horizon, avec une cible 12m disponible comme dans l'entraînement historique.
À 24m, la visite index peut donc être plus ancienne que celle de 12m. Les patients
24m sont ceux ayant les deux cibles à une visite. Les horizons, fenêtres (250–480
jours et 550–910 jours), moyenne des états moteurs au jour cible et seuil +3,5
sont inchangés. Les dates des cibles sont conservées pour audit, jamais en entrée.
Toutes les cibles longitudinales ont été recalculées et comparées; labels =
(cible brute - NP3TOT brut >= 3,5), avec NaN quand la cible manque. Le choix de
visite sur disponibilité d'un suivi est rétrospectif et peut induire un biais
de suivi; il est identique pour toutes les références et tous les candidats.

Référence et candidat sont réentraînés sur ces mêmes tables corrigées. Aucune
comparaison de gain n'utilise directement les anciens chiffres du README.
Le contrôle secondaire conserve la visite index historique 12m et n'évalue
24m que lorsqu'une vraie cible existe : 14 patients val et 18 test. Ce contrôle
n'a pas servi à sélectionner les modèles (same_historical_index_24m.csv).

Les ensembles PATNO sont disjoints et conformes au patient_split.csv original.
Les séquences s'arrêtent à la ligne index exacte, y compris en cas de plusieurs
états médicamenteux dans un même mois. LOCF seulement vers l'avant, médianes et
StandardScaler appris exclusivement sur train; paramètres sauvegardés et repris
à l'inférence. XGB conserve ses valeurs manquantes natives. Ni cibles ni dates
futures n'entrent dans les variables. Le choix déterministe intra-mois ne permet
pas de reconstituer l'ordre réel des examens, absent des sources préparées.

LIMITE DE PROVENANCE : data/raw est absent. Les jointures de modalités par
PATNO/EVENT_ID ne conservent pas leurs dates d'acquisition; l'ETL prend aussi les
dernières données démographiques/années d'études sans date par visite. Une fuite
à ce niveau ne peut être ni exclue ni corrigée rigoureusement avec les seuls
Parquet. Il faut vérifier les dates d'acquisition/disponibilité et faire des
jointures causales sur les fichiers bruts avant toute évaluation prospective.
Les valeurs utilisées sont laissées identiques pour références et candidats.

2. BiLSTM : fonctionnement et architecture

Le score moteur brut est conservé séparément des 44 variables standardisées.
L'ancien extracteur utilisait NP3TOT standardisé dans la connexion résiduelle;
la cible restait en points bruts. Les références corrigées incluent cette
réparation AVANT toute optimisation. La structure est strictement la classe
LongitudinalBiLSTM existante, sans sous-classe ni remplacement de forward.

44 entrées; projection 44->64 + LayerNorm/ReLU/Dropout; LSTM à 2 couches,
64 unités, bidirectionnel; attention additive sur 128; concaténation contexte
et vraie dernière visite (256); projection 256->128; trois têtes 128->64->1,
deux additions résiduelles NP3TOT brut et un logit. {params:,} paramètres.
Les noms, types, dimensions et directions sont vérifiés pour chaque réglage.
Les 42 variables XGB et les 44 variables BiLSTM sont celles des scripts initiaux.
Le dropout existant est réglable; aucune couche ou variable n'est ajoutée.

Le packing, le masque d'attention et lengths-1 de ce BiLSTM étaient déjà corrects.
La sauvegarde historique immédiate torch.save était déjà une vraie sauvegarde.
Le nouveau pipeline sauvegarde aussi immédiatement une copie profonde au meilleur
score de validation, puis recharge le fichier pour l'évaluation. Tests de padding,
invariance au futur, unités du résidu et gradients avec cibles absentes inclus.

3. Recherche et règles fixées avant le test

CPU, {config['threads']} threads, algorithmes PyTorch déterministes. Graines
{selection['seeds']} : aucune sélection de la meilleure graine, aucune moyenne
de prédictions présentée comme un modèle unique. Les tables donnent les moyennes
des MÉTRIQUES de trois entraînements et leurs écarts types séparément.
Temps cumulé d'entraînement des runs conservés : {elapsed/60:.1f} minutes.

XGB : référence historique (600 arbres, lr .03, profondeur 4, régularisation
.5/1.5; classifieur 500 arbres, alpha 0/lambda 1), puis 12 tirages déterministes
par tâche. Recherche de lr, plafond arbres, profondeur, min_child_weight,
subsample, colsample_bytree, alpha, lambda, gamma. Objectifs inchangés :
reg:squarederror / binary:logistic. Régression : arrêt anticipé MAE pour les
candidats, RMSE historique pour référence. Classification : AUC pour référence,
aire PR XGB (aucpr) pour les candidats; sélection finale selon la moyenne
ROC-AUC/AP calculée par sklearn. Le poids positif provient du train seulement.
Les paramètres et le numéro de la meilleure itération sont conservés par run.

BiLSTM : référence aux réglages initiaux corrigée (AdamW .0015, batch64,
wd .0001, dropout .2, clipping2, SmoothL1 beta1, poids 1/.6/.4, 45 époques,
patience12, scheduler facteur .5/patience4; checkpoint choisi sur MAE12).
Trois candidats : balanced_huber, balanced_mae, classification_huber. Ils testent
lr, batch128, scheduler/patience, wd, dropout, clipping, MAE/Huber et poids tâches.
Les deux premiers utilisent la moyenne MAE12/MAE24 pour le checkpoint; le dernier
la moyenne ROC-AUC/AP. Aucun réglage ni époque n'est choisi sur le test.
Pertes par horizon calculées sur cibles finies seulement. Le poids positif BCE
est négatifs/positifs des 14 850 préfixes train (prévalence ~30,6 %).

Après le criblage graine42, la meilleure configuration XGB par tâche et sa référence
sont répétées aux graines17/2026. Pour BiLSTM les trois candidats sont répétés :
MAE et classification-Huber étaient presque ex aequo en classement classification
(écart de moyenne ROC-AUC/AP < 0,0001), justifiant ces deux runs supplémentaires.
Cette extension a été décidée uniquement sur validation, avant l'ouverture du test.
Sélection sur moyenne validation des 3 graines, parmi les configurations répétées.
Régression : MAE plus faible. Classification : moyenne ROC-AUC/AP plus élevée
ET balanced accuracy au moins égale à la référence au seuil de validation.
Les sorties d'un multitâche peuvent donc être servies par différents checkpoints
complets de même architecture; il ne s'agit pas d'une modification des têtes.
Il n'existe pas de nouveau réseau combinant ces têtes : chaque checkpoint reste
un BiLSTM complet à trois sorties. Pour utiliser un seul réseau pour les trois
tâches, examiner ci-dessous ses performances complètes et le compromis entre
horizons. Aucun assemblage de poids ni fusion de prédictions n'est effectué.

{chr(10).join(failure_lines)}

Pour le classifieur XGB, trial_03 avait un classement ROC-AUC/AP légèrement meilleur,
mais une balanced accuracy validation moyenne inférieure et davantage de faux
positifs. Il est rejeté par la règle prévue; les paramètres de référence restent
retenus, avec le seuil de validation documenté.

Toutes les expériences, y compris celles sans gain, restent dans runs/ et
validation_all_experiments.csv. selection.json a été enregistré avant ouverture
du test; test_freeze.json interdit ensuite réglages et resélection dans ce dossier.
Les hashes protègent modèles/données sources; verify recharge les checkpoints
et reproduit les prédictions. Le code et les dépendances sont archivés.

4. Cause du classifieur historique « tout négatif »

Sur les 574 patients de validation, 189 positifs (32,9 %). Checkpoint historique
époque {old['epoch']}, probabilités maximum {old['at_05']['probability_quantiles'][-1]:.6f},
seuil .5 : sensibilité {old['at_05']['sensitivity']:.3f}, spécificité {old['at_05']['specificity']:.3f},
ROC-AUC {old['at_05']['roc_auc']:.4f}, AP {old['at_05']['average_precision']:.4f}.
La discrimination existe mais aucune probabilité ne dépasse .5. Le déséquilibre
modéré, la BCE non pondérée et son poids .4 accompagnent ce phénomène; on ne peut
pas attribuer causalement l'échec à chacun sans ablations supplémentaires. Le bug
d'unités du résidu obligeait en outre la régression à apprendre le niveau absolu.

Un simple seuil de validation {old['validation_threshold']['threshold']:.6f} sur
le checkpoint historique donne sensibilité {old['validation_threshold']['sensitivity']:.4f},
spécificité {old['validation_threshold']['specificity']:.4f}, balanced accuracy
{old['validation_threshold']['balanced_accuracy']:.4f}. Ce gain de seuil n'est pas
un gain de ROC-AUC/AP. Le diagnostic sauvegarde les quantiles de probabilité.

Nouveaux seuils : maximum balanced accuracy validation sous sensibilité >= .5
et spécificité >= .5; égalité départagée par proximité de .5. La référence reçoit
elle aussi ce réglage pour une comparaison équitable. Les contraintes sont de
développement, pas des exigences cliniques approuvées; elles ne sont pas garanties
sur le test. Pondération BCE peut dégrader la calibration : Brier est rapporté,
sans recalibration opportuniste sur le test.

5. Résultats de test — moyenne des trois graines

Régression : MAE/RMSE/biais en points NP3TOT; plus petit MAE = mieux.
{regression.round(5).to_string(index=False)}

Les trois sorties de chaque checkpoint BiLSTM retenu (moyenne 3 graines), pour
rendre visibles les compromis si l'on utilise UN SEUL réseau multitâche :
{full_bilstm.round(5).to_string(index=False)}

Classification, seuil fixé sur validation; 583 patients par graine. La colonne
threshold est une moyenne descriptive : utiliser le seuil individuel du run :
{cls.round(5).to_string(index=False)}

Contrôle au seuil fixe .5, mêmes prédictions :
{fixed.round(5).to_string(index=False)}

Matrices de confusion individuelles [[TN, FP], [FN, TP]] et seuils :
{conf.round(6).to_string(index=False)}

Gains appariés et intervalles bootstrap patients (2 000 tirages, graine901) :
positif = amélioration, négatif = régression.
{intervals.round(5).to_string(index=False)}

Ces intervalles conditionnent sur les trois entraînements retenus et n'intègrent
pas toute l'incertitude de sélection ni de provenance. Les graines sont peu
nombreuses. Un intervalle couvrant zéro ne démontre pas un gain stable. Un gain
validation qui ne se retrouve pas sur le test est un échec de généralisation;
on ne remplace pas le modèle après coup en sélectionnant sur ce même test.
Le checkpoint de référence reste conservé et utilisable.

Contrôle simple « aucun changement » : prédiction = score moteur courant. Il
ne participe pas à la sélection des architectures ou des configurations :
{persistence.round(5).to_string(index=False)}

L'étude inclut les cohortes originales (PD, prodromaux, contrôles), et ne doit pas
être interprétée comme une validation restreinte à des patients PD diagnostiqués.

6. Confirmation indépendante proposée

Geler le pipeline, les variables, les hyperparamètres et la règle de seuil.
Vérifier les acquisitions brutes et années d'études, puis tester sur de nouveaux
PATNO d'une cohorte externe ou d'une extraction ultérieure, jamais utilisés ici.
Définir prospectivement les visites index et le suivi 12/24m, gérer l'attrition,
conserver les états ON/OFF comparables, et rapporter les sous-groupes PD/traitement.
En complément, validation croisée imbriquée par PATNO avec ajustement complet
dans chaque pli; elle apporte une estimation interne, pas un nouveau test externe.

7. Reproduction dans PowerShell ou le terminal PowerShell de VS Code

Depuis C:\\Users\\MSI\\Desktop\\parkinson :

.\\.venv\\Scripts\\python.exe -m unittest discover -s src -p test_model12_experiments.py -v
.\\.venv\\Scripts\\python.exe src/model12_experiments.py --stage verify --output {root.as_posix()}
.\\.venv\\Scripts\\python.exe src/summarize_model12_experiments.py --output {root.as_posix()}

Nouvelle série complète, avec un dossier inexistant (ne pas écraser celle-ci) :

$run = 'experiments/model12_reproduction'
.\\.venv\\Scripts\\python.exe src/model12_experiments.py --stage prepare --output $run
.\\.venv\\Scripts\\python.exe -u src/model12_experiments.py --stage screen --output $run
.\\.venv\\Scripts\\python.exe -u src/model12_experiments.py --stage repeat --output $run
.\\.venv\\Scripts\\python.exe src/model12_experiments.py --stage select --output $run
.\\.venv\\Scripts\\python.exe src/model12_experiments.py --stage test --output $run
.\\.venv\\Scripts\\python.exe src/model12_experiments.py --stage verify --output $run
.\\.venv\\Scripts\\python.exe src/summarize_model12_experiments.py --output $run

screen/repeat peuvent reprendre les runs terminés avant ouverture du test. Ils
doivent utiliser la même configuration figée dans output/config.json. Chaque
run incomplet est réentraîné déterministement; aucun résultat n'est supprimé.
L'environnement installé est documenté dans environment.json et requirements-lock.txt.
Une égalité bit à bit sur d'autres versions/matériels n'est pas garantie.

Livrables : data/ (copies corrigées), target_audit.csv, architecture.json,
validation_all_experiments.csv, validation_finalists_mean.csv,
before_after_test_mean.csv, before_after_test_std.csv, test_metrics.csv,
paired_patient_intervals.csv, same_historical_index_24m.csv,
selected_checkpoints.json, runs/ (poids, config, courbes, prédictions),
retained_reference_checkpoints.json, test_conclusions.json, test_predictions/,
verification.json, source_snapshot/ et source_sha256.json.

Références techniques consultées :
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.typing.DataFrameGroupBy.last.html
https://xgboost.readthedocs.io/en/stable/python/python_api.html
"""
    (root / "RAPPORT_FR.txt").write_text(report, encoding="utf-8")
    print(regression.round(4).to_string(index=False))
    print(cls.round(4).to_string(index=False))
    print(f"Report: {root / 'RAPPORT_FR.txt'}")


if __name__ == "__main__":
    main()
