# Optimisation des sept familles

Les expériences sont isolées dans `experiments/seven_models_20261030`. Les
checkpoints et les Parquet historiques sont conservés. Le tableau existant
est reproduit dans `historical_reproduction_comparison.csv`, sans modifier
le registre de résultats du dashboard.

Le diagnostic et les limites sont détaillés dans `AUDIT.md` du dossier
d'expérience. Le tableau fourni est à **24 mois**, cible NP3TOT futur en
points bruts. Il contient 462 patients ; le tableau 12 mois en contient 583.
Les cibles tabulaires anciennes sont mal rattachées à 24 mois pour 444 des
462 patients test. Les gains ne doivent pas être calculés en comparant
directement ces chiffres aux résultats du protocole corrigé.

## Commandes Windows

Depuis la racine, avec le venv existant :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s src -p test_seven_model_protocol.py -v
.\.venv\Scripts\python.exe -m unittest discover -s src -p test_seven_preprocessing.py -v
.\.venv\Scripts\python.exe -m unittest discover -s src -p test_retention_statistics.py -v
```

Pour un nouveau dossier d'expérience, utiliser un autre nom pour `--output`.
La préparation refuse de remplacer un dossier existant :

```powershell
.\.venv\Scripts\python.exe src/seven_model_experiments.py --stage prepare --output experiments/seven_models_new
.\.venv\Scripts\python.exe src/seven_model_experiments.py --stage reproduce --output experiments/seven_models_new
.\.venv\Scripts\python.exe src/seven_model_experiments.py --stage pilot --output experiments/seven_models_new
.\.venv\Scripts\python.exe src/seven_model_batch.py --output experiments/seven_models_new --workers 2
```

Le batch exécute `screen → shortlist → repeat → select → test → report → verify`,
avec une barrière entre les étapes. Au plus deux processus d'entraînement sont
actifs, chacun limité à deux threads par la configuration. Une reprise de la
commande réutilise les runs déjà terminés. La graine 42 du screening est
réutilisée sans réentraînement lorsque la configuration est inchangée.

Pour reproduire **tous les livrables de cette étude**, y compris les ablations
du remplissage et la comparaison aux anciens checkpoints, utiliser les étapes
explicites suivantes dans un nouveau dossier (le batch ci-dessus reproduit le
parcours principal) :

```powershell
$studyOutput = 'experiments/seven_models_replay'
.\.venv\Scripts\python.exe src/seven_model_experiments.py --stage prepare --output $studyOutput --config configs/seven_models.json
.\.venv\Scripts\python.exe src/seven_model_experiments.py --stage reproduce --output $studyOutput
.\.venv\Scripts\python.exe src/seven_model_experiments.py --stage pilot --output $studyOutput
.\.venv\Scripts\python.exe src/seven_model_experiments.py --stage screen --output $studyOutput
.\.venv\Scripts\python.exe src/seven_model_experiments.py --stage shortlist --output $studyOutput
.\.venv\Scripts\python.exe src/seven_model_experiments.py --stage repeat --output $studyOutput
.\.venv\Scripts\python.exe src/analyze_seven_model_runs.py --stage quality --output $studyOutput
.\.venv\Scripts\python.exe src/analyze_seven_model_runs.py --stage prior --output $studyOutput
.\.venv\Scripts\python.exe src/extend_seven_model_preprocessing.py --output $studyOutput
.\.venv\Scripts\python.exe src/seven_model_experiments.py --stage select --output $studyOutput
.\.venv\Scripts\python.exe src/seven_model_experiments.py --stage test --output $studyOutput
.\.venv\Scripts\python.exe src/seven_model_experiments.py --stage report --output $studyOutput
.\.venv\Scripts\python.exe src/seven_model_experiments.py --stage verify --output $studyOutput
.\.venv\Scripts\python.exe src/analyze_seven_model_runs.py --stage ablations --output $studyOutput
.\.venv\Scripts\python.exe src/report_seven_model_results.py --output $studyOutput
```

Pour exécuter une seule étape ou une seule famille :

```powershell
.\.venv\Scripts\python.exe src/seven_model_experiments.py --stage screen --family cross --output experiments/seven_models_new
```

Familles : `tabular_xgb`, `bilstm`, `late`, `early`, `cross`, `imaging_mlp`,
`imaging_xgb`. La configuration complète est `configs/seven_models.json`,
copiée dans le dossier d'expérience avant les essais.

Analyses complémentaires de développement, avec leurs propres sorties :

```powershell
.\.venv\Scripts\python.exe src/analyze_seven_model_runs.py --stage quality
.\.venv\Scripts\python.exe src/analyze_seven_model_runs.py --stage prior
.\.venv\Scripts\python.exe src/analyze_seven_model_runs.py --stage ablations
```

Les ablations de perte et d'indicateurs de manque réentraînent XGBoost/MLP sur
train et évaluent sur validation. Les ablations de masques des fusions gardent
les poids fixes : elles décrivent une sensibilité à l'inférence, pas le gain
qu'aurait apporté un autre entraînement. Elles ne changent aucune sélection
après consultation du test.

L'audit des valeurs manquantes a également motivé une extension de développement
**avant tout nouveau test** : `causal_locf` et `robust_causal_locf` pour les deux
XGBoost, sur trois graines. Elle est tracée dans
`preprocessing_extension_policy.json`. La configuration du dossier d'expérience
et `configs/seven_models.json` incluent ces essais pour leur reproduction.
Le remplissage causal conserve les mêmes colonnes et ne prend aucune mesure
postérieure à la ligne index. Le type de prétraitement est sauvegardé comme
attribut du checkpoint XGBoost.

Après le batch, produire les figures et les diagnostics de sous-groupes :

```powershell
.\.venv\Scripts\python.exe src/report_seven_model_results.py
```

## Sorties

- `target_audit.csv`, `data_audit.json` : partitions, index, cibles, limites de provenance.
- `historical_reproduced_metrics.csv` : reproduction réelle des checkpoints initiaux.
- `runs_validation_metrics.csv` : essais de screening.
- `final_validation_metrics.csv` : finalistes, trois graines prédéfinies.
- `selection.json`, `freeze.json`, `frozen_checkpoints.json` : décisions sur validation et SHA256 avant test.
- `best_available_checkpoints.json` : points de livraison à graine fixe 42, en conservant un ancien modèle lorsqu'il est meilleur ou équivalent sur la validation commune.
- `test_metrics.csv`, `test_classification_separate.csv` : métriques issues d'inférences réelles, régression et classification séparées.
- `before_after_all_metrics.csv`, `before_after_comparison.csv` : avant/après par modèle, horizon, split, moyenne et écart type des graines.
- `paired_patient_bootstrap.csv` : changement de MAE et intervalle bootstrap apparié par patient.
- `RAPPORT_FR.txt` : synthèse des résultats et limites.
- `validation_preprocessing_chain.csv` : effet du seul LOCF, puis du réglage robuste, séparé du gain global.
- `before_after_test.png`, `test_subgroups_diagnostic.csv` : figure et diagnostics, sans sélection de sous-groupes.
- `logs/`, `source_snapshot/`, `requirements-lock.txt` : journaux, code et environnement.

La sélection porte sur une configuration multitâche entière pour chaque réseau,
et sur une configuration par horizon pour XGBoost. Les graines 42, 17 et 2026
sont toutes conservées : aucune sélection de la meilleure graine, aucun nouvel
ensemble de prédictions. Les checkpoints sélectionnés restent dans le dossier
d'expérience ; aucun ancien modèle en production n'est remplacé silencieusement.

Un garde-fou supplémentaire a été ajouté **avant toute prédiction de test du
nouveau protocole** : le candidat n'est retenu que si la borne supérieure à
95 % du changement de MAE de validation, par bootstrap apparié des patients,
est strictement négative. Sinon la référence est conservée. L'amendement et
les manifestes précédents restent dans le dossier d'expérience. Ces intervalles
de développement sont conditionnels aux graines, exploratoires après tuning
et ne constituent pas une confirmation indépendante ni un contrôle de toutes
les comparaisons multiples.

L'API `/api/benchmarks?protocol=audited` lit automatiquement les CSV vérifiés.
Le front **Modèles & analyses → Audité** affiche l'avant/après et les écarts
types. `protocol=historical` conserve l'archive avec sa limite d'alignement.
Les prédictions des modèles déjà actifs restent distinctes de ces expériences.

Les scores de test ne sont pas entièrement indépendants : ce test a déjà été
consulté lors de travaux antérieurs. Les fichiers bruts sont absents, donc les
dates d'acquisition des modalités et certains ordres intra-mois restent à
vérifier sur les sources avant une interprétation prospective.

Références techniques utilisées :
[paramètres XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html),
[LSTM PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html),
[MultiheadAttention PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
