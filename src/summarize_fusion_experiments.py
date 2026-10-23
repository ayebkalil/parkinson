"""Generate traceable French report from actual saved validation predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_experiments import metrics, write_json


def paired_bootstrap(old, new, horizon, aligned, seed=31415):
    keys = ["PATNO", "visit_date", f"true_{horizon}m"]
    pd.testing.assert_frame_equal(old[keys], new[keys])
    valid = old[f"true_{horizon}m"].notna().to_numpy(copy=True)
    if aligned:
        valid &= old[f"aligned_{horizon}m"].to_numpy(bool)
    y = old.loc[valid, f"true_{horizon}m"].to_numpy()
    improvement = abs(old.loc[valid, f"pred_{horizon}m"].to_numpy()-y) - abs(new.loc[valid, f"pred_{horizon}m"].to_numpy()-y)
    rng = np.random.default_rng(seed)
    samples = np.array([rng.choice(improvement, len(improvement), replace=True).mean() for _ in range(2000)])
    return dict(n=len(y), improvement=float(improvement.mean()),
                low=float(np.quantile(samples,.025)), high=float(np.quantile(samples,.975)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/fusion_20261013")
    args = parser.parse_args()
    root = Path(args.output)
    selection = json.loads((root / "selection.json").read_text())["configurations"]
    write_json(root / "selected_checkpoints.json", {
        a: dict(path=str(root / f"{a}_{c}_seed42/checkpoint.pt"), config=c, seed=42,
                seed_policy="Fixed delivery seed 42; never choose the best repeat seed")
        for a,c in selection.items()})
    audit = json.loads((root / "audit.json").read_text())
    rows, histories, diagnostics, classifications = [], [], [], []
    for architecture in ["early", "late"]:
        pred = pd.read_csv(root / f"historical_{architecture}_predictions.csv")
        for row in metrics(pred):
            rows.append(dict(architecture=architecture, config="historical_checkpoint",seed="unknown",**row))
    for path in sorted(root.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        histories.append(summary)
        pred = pd.read_csv(path.parent / "validation_predictions.csv")
        for row in metrics(pred):
            rows.append(dict(architecture=summary["architecture"],config=summary["config"],seed=summary["seed"],**row))
        classifications.append({**{k:summary[k] for k in ["architecture","config","seed","epoch"]},
                                **json.loads((path.parent / "classification.json").read_text())})
        if "weight_imaging" in pred:
            for status in [True,False]:
                sub = pred[pred.imaging_available == status]
                base = dict(config=summary["config"], seed=summary["seed"], imaging_available=status,
                            patients=len(sub), mean_weight_imaging=sub.weight_imaging.mean(),
                            min_weight_imaging=sub.weight_imaging.min(), max_weight_imaging=sub.weight_imaging.max())
                for h in [12,24]:
                    for name in ["pred","clinical","imaging","uniform"]:
                        valid = sub[f"aligned_{h}m"] & sub[f"true_{h}m"].notna()
                        y, p = sub.loc[valid,f"true_{h}m"], sub.loc[valid,f"{name}_{h}m"]
                        diagnostics.append(dict(**base,horizon=h,predictor=name,n=int(valid.sum()),
                                                mae=abs(p-y).mean(),bias=(p-y).mean(),prediction_std=p.std()))
    df = pd.DataFrame(rows)
    df.to_csv(root / "all_metrics.csv",index=False)
    runs = pd.DataFrame(histories)
    runs.to_csv(root / "runs.csv",index=False)
    pd.DataFrame(classifications).to_csv(root / "classification_only.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(root / "late_diagnostics.csv",index=False)
    # Same metric column names as evaluate.py; no historical ledger is overwritten.
    export = df[df.predictor == "pred"].copy()
    export["model_name"] = export.architecture + ":" + export.config + ":seed" + export.seed.astype(str)
    export["target_horizon"] = "+" + export.horizon.astype(str) + " Months (NP3TOT)"
    export["split"] = "validation"
    export.to_csv(root / "benchmark_validation.csv",index=False)
    primary = df[(df.protocol=="aligned") & (df.predictor=="pred")]
    before_after, confidence = [], []
    for architecture in ["early","late"]:
        old = pd.read_csv(root / f"historical_{architecture}_predictions.csv")
        selected = selection[architecture]
        new = pd.read_csv(root / f"{architecture}_{selected}_seed42/validation_predictions.csv")
        for protocol in ["aligned","canonical"]:
            for h in [12,24]:
                subset = df[(df.architecture==architecture)&(df.protocol==protocol)&(df.horizon==h)&(df.predictor=="pred")]
                baseline = subset[subset.config=="historical_checkpoint"].iloc[0]
                after = subset[(subset.config==selected)&(subset.seed==42)].iloc[0]
                before_after.append(dict(architecture=architecture,protocol=protocol,horizon=h,n=int(after.n_samples),
                                         before=baseline.mae,after=after.mae,gain=round(baseline.mae-after.mae,3),config=selected))
                confidence.append(dict(architecture=architecture,protocol=protocol,horizon=h,
                                       **paired_bootstrap(old,new,h,protocol=="aligned")))
    pd.DataFrame(before_after).to_csv(root / "before_after.csv",index=False)
    write_json(root / "descriptive_bootstrap.json",confidence)
    stability = primary[primary.config != "historical_checkpoint"].groupby(["architecture","config","horizon"]).mae.agg(["count","mean","std","min","max"]).reset_index()
    stability.to_csv(root / "seed_stability.csv",index=False)
    paired_seeds = []
    for a, chosen in selection.items():
        group = primary[primary.architecture == a]
        joined = group[group.config == "corrected"].merge(
            group[group.config == chosen], on=["seed", "horizon"], suffixes=("_corrected", "_selected"))
        for row in joined.itertuples():
            paired_seeds.append(dict(architecture=a, seed=row.seed, horizon=row.horizon,
                                     gain_vs_corrected=round(row.mae_corrected-row.mae_selected, 3)))
    paired_seeds = pd.DataFrame(paired_seeds)
    paired_seeds.to_csv(root / "paired_seed_gains.csv",index=False)
    multi_seed_conclusion = []
    for a in ["early","late"]:
        g = stability[(stability.architecture==a)&(stability.horizon==12)]
        corrected = g[g.config=="corrected"].iloc[0]
        chosen = g[g.config==selection[a]].iloc[0]
        multi_seed_conclusion.append(
            f"{a} à 12m : corrections seules {corrected['mean']:.3f}; configuration retenue {chosen['mean']:.3f} (moyennes sur {int(chosen['count'])} graines).")
    total_seconds = runs.seconds.sum()
    screening = runs[runs.seed == 42][["architecture", "config", "epoch", "selection_mae", "parameters", "seconds"]].sort_values(["architecture", "selection_mae"])
    interpretation = []
    for architecture in ["early", "late"]:
        group = screening[screening.architecture == architecture]
        reference = float(group.loc[group.config == "corrected", "selection_mae"].iloc[0])
        for row in group.itertuples():
            if row.config != "corrected":
                change = reference - row.selection_mae
                interpretation.append(f"{architecture}, {row.config} : {'gain' if change > 0 else 'dégradation'} de {abs(change):.3f} sur le critère moyen aligné vs corrected (graine 42).")
    diagnostic_table = df[(df.architecture=="late") & (df.config==selection["late"]) & (df.seed==42) & (df.protocol=="aligned")]
    late_available = pd.DataFrame(diagnostics)
    late_available = late_available[(late_available.config==selection["late"]) & (late_available.seed==42) & (late_available.horizon==12) & (late_available.predictor=="pred")]
    persistence = pd.read_csv(root / "historical_early_predictions.csv")
    motor = pd.read_parquet("data/processed/val_tabular_latest.parquet")
    assert motor.PATNO.tolist() == persistence.PATNO.tolist()
    for h in [12,24]:
        persistence[f"pred_{h}m"] = motor.NP3TOT.to_numpy()
    persistence_metrics = metrics(persistence)
    write_json(root / "persistence_diagnostic.json", persistence_metrics)
    oral_results = []
    for architecture in ["early", "late"]:
        row = next(r for r in before_after if r['architecture']==architecture and r['protocol']=='aligned' and r['horizon']==12)
        oral_results.append(f"{architecture.capitalize()} Fusion : {row['before']:.3f} à {row['after']:.3f}")
    report = f"""CONTRIBUTION AUX FUSIONS — 13 OCTOBRE 2026

Résultats de validation uniquement. Aucun score de test recalculé. Les gains
ci-dessous sont des observations de développement, pas une validation clinique.

1. Fonctionnement initial vérifié dans le code

Early Fusion : 44 variables (30 cliniques, incluant les biomarqueurs et deux
délais temporels, 11 d'imagerie, 3 démographiques), projection Linear/LayerNorm/
ReLU/Dropout, BiLSTM à deux couches de largeur 64, attention temporelle, deux
têtes de régression résiduelle et une tête de classification.

Late Fusion : branche clinique+démographie (33 variables), projection puis
BiLSTM à deux couches de largeur 48 avec attention; branche MLP imagerie+
démographie (14 variables) à la dernière visite. Les deux branches prédisent
deux scores résiduels. Un réseau Linear(144,32)/ReLU/Linear(32,2)/Softmax pondère
leurs prédictions, avec les mêmes poids pour les deux horizons. Ce réseau est
appris conjointement avec les branches sur le train; aucun empilement appris
séparément sur des prédictions d'entraînement. La branche dite « imagerie »
utilise aussi démographie ET score moteur actuel via l'ancrage résiduel : elle
n'est donc pas un modèle d'imagerie pure.

Entraînement initial : tous les préfixes avec cible 12 mois; AdamW, lr=0,001,
weight_decay=0,001, batch=32, dropout=0,25, 35 époques; SmoothL1(beta=1),
poids 1 / 1,2 / 0,5 pour 12m / 24m / classification, clipping 1,5,
ReduceLROnPlateau. Validation sur la dernière visite canonique par patient.
La sélection utilisait une moyenne non pondérée des pertes de minibatches.
Pas d'arrêt anticipé. Le checkpoint dit « meilleur » finissait avec les poids
finaux à cause d'une copie superficielle.

Références examinées et laissées intactes : XGBoost tabulaire (arbres sur visite
unique), BiLSTM de référence (attention + état de dernière visite + têtes
résiduelles), imagerie XGBoost/MLP (13 variables), Cross-Attention (requêtes
cliniques et jetons DaTSCAN/MRI/démographie). Aucune nouvelle modalité ajoutée.

2. Audit et corrections

- Score résiduel : conserver NP3TOT brut séparément du tableau standardisé.
  L'ancien extracteur passait NP3TOT standardisé aux têtes prédisant des points.
- Late Fusion : index lengths-1 par patient, au lieu de x[:,-1] qui lisait le
  padding des séquences courtes; invariance au contenu du minibatch testée.
- Sauvegarde : copie profonde des poids au meilleur score de validation.
- Normalisation : médianes et StandardScaler appris uniquement sur train;
  scaler ajusté sur la même LOCF causale que celle appliquée à l'inférence.
- Imagerie jamais observée dans le préfixe : poids forcés à [1,0], pour ne pas
  confier une prédiction à une branche alimentée uniquement par l'imputation.
  En cas d'absence partielle, LOCF puis médiane du train. Les indicateurs de
  manque de l'essai missing_huber décrivent la visite courante avant LOCF.
- Pertes : cibles non finies exclues à chaque horizon; perte nulle connectée
  au graphe lorsque toutes les cibles d'un minibatch sont absentes.
- Prédictions exportées avec PATNO, date, cibles et masques; ordre et valeurs
  comparés explicitement. Régression et classification dans des fichiers séparés.

Les partitions sont disjointes et concordent avec patient_split.csv. Train :
{audit['train']['patients']} patients / {audit['train']['rows']} lignes; validation :
{audit['val']['patients']} patients / {audit['val']['rows']} lignes. Les cibles longitudinales
ont été recalculées pour audit sur train et validation : zéro discordance.
Les entrées sont des préfixes jusqu'à la visite de prédiction; les cibles et
leurs deltas sont exclus des 44 variables. Les délais delta_t_days et
elapsed_months_bl sont conservés. Packing du BiLSTM et masque d'attention
écartent le padding. Les visites d'un même mois/état médicamenteux ne sont pas
réagrégées : les partitions et le protocole longitudinal sont conservés.

PROBLÈME NON CORRIGÉ DANS LES DONNÉES COMMUNES : groupby(PATNO).last() construit
les tables tabulaires en prenant la dernière valeur non nulle par colonne.
À 24 mois, 435 des 449 cibles de validation sont héritées d'une visite passée
alors que la visite index ne possède pas de cible 24 mois. À 12 mois, les 574
cibles sont concordantes. Les fichiers originaux ne sont pas modifiés.
Le protocole « aligned » exclut seulement les cibles non rattachées à la date
index, sans inventer ni remplacer une cible; à 24 mois il reste 14 patients.
Le protocole « canonical » conserve exactement les étiquettes historiques :
son score 24 mois N=449 n'est PAS une mesure fiable de prévision à +24 mois.
Les 14 patients alignés comprennent seulement 6 patients diagnostiqués Parkinson,
7 prodromaux et 1 contrôle sain : cet effectif n'est pas représentatif à lui seul.
Tous les avant/après comparent les mêmes patients ET les mêmes cibles à
l'intérieur du protocole indiqué. Les deux effectifs ne sont pas confondus.

Autre anomalie trouvée : run_statistical_tests.py passe l'historique COMPLET
des patients aux modèles, au-delà de la visite index, et reconstruit des
prétraitements différents de ceux de certains checkpoints. Ses tests existants
ne prouvent pas les nouveaux gains. Ce script est laissé intact; les nouveaux
exports sont bornés à la visite index et le bootstrap est apparié sur PATNO/date.

Disponibilité des modalités : l'ETL les associe par PATNO/EVENT_ID, sans garder
leurs dates d'acquisition. data/raw est absent. L'absence de mesure acquise
après la visite à l'intérieur d'un même événement ne peut donc pas être
certifiée ici. Même limite pour l'ordre intra-mois ON/OFF. Les préfixes, LOCF
et données dérivées contrôlables ont été audités; la provenance brute reste à
vérifier avant toute conclusion prospective.

3. Protocole expérimental et budget

Environnement .venv existant, CPU, 2 threads PyTorch, déterminisme activé.
Versions et processeur dans environment.json; dépendances dans requirements-lock.txt.
Trois configurations, deux architectures, graine 42, maximum 24 époques et
patience 6. Scheduler facteur 0,5 / patience 3, clipping 1,5.
Critère fixé avant les essais : moyenne des MAE 12m et 24m alignées, calculées
sur tous les patients valides, sans moyenner les moyennes de minibatches.
ATTENTION : la moitié de ce critère repose sur seulement 14 patients à 24m.

corrected : corrections ci-dessus, dimensions et paramètres initiaux conservés
(sauf protocole de sélection, arrêt anticipé et budget).
compact_mae : une couche, largeur 32, dropout 0,15, batch 64, MAE,
weight_decay=0,0001, poids 24m=1 et classification=0,2.
missing_huber : même taille compacte, indicateurs de manque, Huber beta=2;
pour Late, supervision auxiliaire des branches (poids 0,1), branche imagerie
supervisée uniquement lorsqu'une imagerie existe dans le préfixe.
Ces essais sont des ensembles de réglages, pas des ablations permettant
d'attribuer un gain à un hyperparamètre isolé. lr=0,001 et clipping inchangés :
pas de grille exhaustive. Les médianes/standardisation sont déjà pertinentes;
aucune recherche supplémentaire de normalisation n'a été engagée.

Sélection graine 42 : Early={selection['early']}; Late={selection['late']}.
Répétitions prévues de corrected et de la configuration retenue avec graines
17 et 2026, sans resélection de configuration ni sélection de la meilleure
graine. Checkpoint de livraison : graine 42 fixée pour les deux architectures.
Entraînements effectivement terminés : {len(runs)}; temps cumulé entraînement,
préparation et validation : {total_seconds/60:.1f} minutes (hors audit et tests).
Les checkpoints initiaux sont réévalués, pas réentraînés : leur graine et leur
historique d'apprentissage complet ne sont pas enregistrés dans les anciens .pt.

4. Avant / après — MAE en points NP3TOT

{pd.DataFrame(before_after).to_string(index=False)}

« aligned » = cible correctement rattachée à la visite index.
« canonical » = comparaison technique sur les cibles historiques, invalide
comme mesure prospective à 24m pour les raisons décrites ci-dessus.

Toutes les expériences, validation alignée (RMSE, R², biais et effectifs) :

{primary[['architecture','config','seed','horizon','n_samples','mae','rmse','r2','bias']].to_string(index=False)}

Comparaison des réglages sur la graine de sélection :

{screening.to_string(index=False)}

{chr(10).join(interpretation)}

Stabilité entre graines (écart-type d'échantillon, NaN si une seule graine) :

{stability.to_string(index=False)}

Effet supplémentaire du réglage retenu vs corrections seules, à graine égale
(positif = gain; négatif = dégradation; ce tableau n'a pas servi à resélectionner) :

{paired_seeds.to_string(index=False)}

{chr(10).join(multi_seed_conclusion)}

Conclusion des répétitions : les configurations compactes retenues au premier
essai ne démontrent pas de gain supplémentaire stable sur les corrections
seules. Sur trois graines, les corrections seules ont une MAE 12m moyenne
légèrement meilleure pour les deux architectures. L'amélioration la mieux
étayée porte donc sur les corrections; on ne doit pas attribuer tout le gain
aux hyperparamètres. Les configurations de sélection restent figées, et les
checkpoints corrected des trois graines sont également livrés. Aucune graine
de répétition n'a été choisie après coup pour embellir le tableau avant/après.

La dispersion entre graines n'est pas un intervalle de confiance clinique.
Le diagnostic de persistance (prédire le NP3TOT actuel sans apprendre) donne
une MAE alignée de {persistence_metrics[2]['mae']:.3f} à 12 mois et {persistence_metrics[3]['mae']:.3f} à 24 mois.
Il permet de distinguer le gain sur les anciens checkpoints du gain au-delà
de la simple conservation du score actuel; ce diagnostic ne remplace aucune fusion.
descriptive_bootstrap.json contient 2000 rééchantillonnages appariés de patients,
graine 31415, sur la différence MAE historique moins nouvelle MAE graine 42.
Ces intervalles sont descriptifs : la validation a servi à choisir les époques
et configurations; aucune significativité confirmatoire n'est revendiquée.

5. Diagnostics Late Fusion

late_diagnostics.csv compare les deux branches seules, la moyenne uniforme et
la pondération finale, par disponibilité d'imagerie et horizon aligné : MAE,
biais, écart-type des prédictions, poids min/moyen/max. Il s'agit des branches
du modèle conjoint, pas de modèles autonomes réentraînés. La moyenne uniforme
inclut la branche imputée même sans imagerie, volontairement comme diagnostic.
Les deux branches ont les mêmes unités NP3TOT grâce au score moteur brut.
Le gating reste appris sur le train; aucune calibration ou pondération n'a été
ajustée aux cibles de test. Les gradients des deux branches et du gate sont
testés lorsqu'une imagerie est disponible. Un poids nul est intentionnel
lorsque l'imagerie est entièrement absente.

Branches et combinaison de la configuration retenue, graine 42, même cohorte :

{diagnostic_table[['horizon','predictor','n_samples','mae','rmse','bias']].to_string(index=False)}

Disponibilité et poids de l'imagerie, configuration retenue graine 42 :

{late_available[['imaging_available','patients','mean_weight_imaging','min_weight_imaging','max_weight_imaging','mae','bias']].to_string(index=False)}

La comparaison clinique/imagerie/uniforme permet de vérifier si le gate atténue
une branche moins informative; elle ne démontre pas une causalité biologique.

6. Fichiers et reproduction Windows / terminal VS Code

src/fusion_experiments.py : préparation isolée, modèles dérivés des classes
originales, audit, entraînement, export et rechargement des checkpoints.
src/test_fusion_experiments.py : dix tests ciblés.
src/summarize_fusion_experiments.py : tableaux/diagnostics/rapport issus des CSV.
configs/fusion_experiments.json : paramètres fixés et graines.
README.md : accès au nouveau pipeline et avertissement sur les anciens résultats.
{root} : résultats, checkpoints, prédictions, provenance et rapport.
Aucun code des modèles de référence, données communes, résultat ou checkpoint
historique n'est écrasé. Les empreintes SHA256 des artefacts initiaux sont
vérifiées à la fin. Les checkpoints nouveaux contiennent modèle, configuration,
prétraitement, graine, époque et empreintes des données train/validation.

Commandes depuis C:\\Users\\MSI\\Desktop\\parkinson :

.\\.venv\\Scripts\\python.exe -m unittest discover -s src -p test_fusion_experiments.py -v
.\\.venv\\Scripts\\python.exe src/fusion_experiments.py --stage verify
.\\.venv\\Scripts\\python.exe src/summarize_fusion_experiments.py

Pour une reproduction complète, utiliser un NOUVEAU dossier pour préserver
les résultats livrés (les entraînements refusent d'écraser un dossier existant) :

.\\.venv\\Scripts\\python.exe src/fusion_experiments.py --stage screen --output experiments/reproduction_01
.\\.venv\\Scripts\\python.exe src/fusion_experiments.py --stage repeat --output experiments/reproduction_01
.\\.venv\\Scripts\\python.exe src/fusion_experiments.py --stage verify --output experiments/reproduction_01
.\\.venv\\Scripts\\python.exe src/summarize_fusion_experiments.py --output experiments/reproduction_01

Compatibilité : réutilisation directe de evaluate_regression et
evaluate_classification de src/evaluate.py; benchmark_validation.csv conserve
ses noms de métriques. reload_validation() fournit l'adaptateur aux nouveaux
checkpoints enrichis. Ne pas les charger avec l'ancien script qui reconstruit
implicitement les entrées et n'accepte qu'un state_dict nu. Aucun résultat de
classification (probabilités et labels 0/1) n'est présenté comme une MAE NP3TOT.

7. Limites et suite

Les gains mesurés incluent surtout des corrections de fonctionnement; ce ne
sont pas uniquement des gains d'hyperparamètres. Le test a déjà été consulté
pendant le développement initial et ne peut plus être décrit comme totalement
indépendant. Il n'a pas été utilisé ici pour choisir quoi que ce soit.
Un travail d'équipe doit reconstruire les tables tabulaires en conservant des
lignes entières (tail(1), sans récupérer d'anciennes cibles), puis approuver
un protocole avec un vrai suivi 24m et réévaluer toutes les références de façon
identique. Les splits actuels ne doivent pas être changés silencieusement.
Confirmation conseillée : cohorte externe ou nouveaux patients ultérieurs
jamais consultés, protocole/paramètres figés et dates sources vérifiées.
Les préfixes multiples surpondèrent les patients très suivis pendant
l'apprentissage. La robustesse intercohortes/médications et la classification
n'ont pas fait l'objet d'une nouvelle optimisation indépendante.

Ma contribution a consisté à auditer les deux fusions existantes, corriger
l'unité du score moteur résiduel, la lecture du padding en Late Fusion et la
sauvegarde des meilleurs poids, puis exécuter des essais reproductibles sur
la validation en conservant le BiLSTM commun d'Early Fusion et les branches
avec pondération apprise de Late Fusion. J'ai aussi identifié des cibles
24 mois rattachées à d'anciennes visites, que j'ai signalées sans modifier
les données de l'équipe. Les tableaux ci-dessus décrivent les gains réellement
mesurés et leur dispersion. À 12 mois, sur les mêmes 574 patients, les MAE des
checkpoints retenus ont évolué ainsi : {('; '.join(oral_results))} points NP3TOT.
À 24 mois, l'effectif correctement rattaché est trop petit pour affirmer un
gain solide. Ces résultats ne constituent pas une preuve de performance sur
une cohorte indépendante.
"""
    (root / "RAPPORT_FR.txt").write_text(report, encoding="utf-8")
    print(pd.DataFrame(before_after).to_string(index=False))
    print(stability.to_string(index=False))


if __name__ == "__main__":
    main()
