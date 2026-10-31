# Audit des sept familles

Le tableau fourni correspond exclusivement à `+24 Months (NP3TOT)` dans
`report/benchmark_results.csv`. La cible est le **score moteur futur brut**,
pas le delta. Les fenêtres conservées sont 250–480 jours (centre 365) et
550–910 jours (centre 730), à partir de la visite index. Les scores des états
médicamenteux disponibles à la date future sont moyennés. La classification
distincte porte sur une augmentation d'au moins 3,5 points à 12 mois.

## Chemins du protocole existant

| Fonction | Code |
|---|---|
| Assemblage et cibles | `src/build_multimodal_dataset.py` |
| Partitions fixes par PATNO | `src/create_patient_split.py`, `data/processed/patient_split.csv` |
| Séquences, BiLSTM et padding | `src/train_model2_lstm.py` |
| XGBoost tabulaire | `src/train_model1_xgboost.py` |
| MLP et XGBoost imagerie | `src/train_model3_imaging.py` |
| Early, Late et Cross-Attention | `src/train_fusion_models.py` |
| Métriques et registre historique | `src/evaluate.py` |
| Interface historique | `app.py` |
| API et interface React | `backend/service.py`, `backend/main.py`, `frontend/src/App.tsx` |
| Corrections déjà présentes | `src/model12_data.py`, `src/model12_experiments.py`, `src/fusion_experiments.py` |

## Constat et décision

1. Les identifiants patients des partitions sont disjoints. Le split existant
   est conservé : 3787 patients train, 827 validation, 812 test ; tous ne
   possèdent pas une cible future éligible.
2. `groupby(PATNO).last()` choisit une dernière valeur non nulle **par colonne**.
   Les lignes tabulaires peuvent donc mélanger plusieurs visites. Les cibles
   à 24 mois sont mal rattachées pour 2075/2155 lignes train, 435/449 validation
   et 444/462 test. Des milliers de cellules de features sont aussi héritées
   d'autres lignes. Ces résultats historiques ne sont pas une référence valide
   pour calculer un gain d'optimisation sur des cibles corrigées.
3. Le protocole corrigé reprend `model12_data.prepare` : ligne complète exacte,
   dernière visite éligible propre à chaque horizon, avec les deux cibles
   disponibles pour l'éligibilité 24 mois, comme dans la cohorte initiale.
   Les effectifs restent 2683/2155 train, 574/449 validation, 583/462 test pour
   12/24 mois. Les index peuvent changer à 24 mois. `target_audit.csv` détaille
   les différences. Les données partagées restent intactes.
4. Les extracteurs historiques ont utilisé le NP3TOT standardisé comme
   ancrage résiduel alors que la cible reste en points bruts. Le nouvel
   extracteur conserve le score brut séparément. Le score initial n'est ajouté
   qu'une fois ; les tests mettent à zéro les têtes pour contrôler ce point.
5. Late et Cross-Attention utilisent `x[:, -1]` pour l'imagerie, ce qui peut
   désigner le padding. Les nouvelles classes héritées indexent `lengths-1`
   par patient et sont testées en batch et individuellement.
6. Le BiLSTM historique fait déjà un packing et une attention masquée corrects.
   Il concatène l'attention avec l'état de la dernière visite réelle : ce choix
   est conservé. La partie backward de cet état décrit la visite index, tandis
   que le contexte d'attention porte l'historique bidirectionnel. Aucune visite
   au-delà de l'index n'entre dans les nouveaux exemples.
7. Une copie superficielle de `state_dict()` dans les fusions et le MLP garde
   des tenseurs liés à l'entraînement : les poids sélectionnés dérivent jusqu'à
   la fin. Les nouvelles expériences clonent immédiatement les tenseurs CPU,
   sauvegardent puis rechargent le meilleur checkpoint de validation.
8. LOCF est uniquement antérieure à l'index. Les médianes et scalers sont
   ajustés sur train uniquement, puis sauvegardés. La normalisation est ajustée
   sur les données remplies par la même règle que celle utilisée à l'inférence.
   La référence XGBoost conserve ses valeurs manquantes natives. Une extension
   motivée par l'audit de manque compare aussi le LOCF causal, sans ajouter de
   variable ou de patient. Son effet est isolé de celui de la perte robuste.
9. Late utilise un Softmax de deux poids pour combiner des scores à la même
   échelle, appris conjointement avec les branches. Aucun stacking sur des
   prédictions d'entraînement n'est ajouté. Si aucune imagerie n'a été observée
   dans le préfixe, les poids sont [1,0]. Avec une absence partielle, le bloc
   conserve LOCF puis médiane du train.
    Les deux branches conservent l'ancrage moteur actuel prévu par l'architecture :
    la branche dite « imagerie » de Late Fusion n'est donc pas un modèle
    monomodal indépendant. Les ablations de branches sont des diagnostics
    à poids fixes, pas de nouveaux modèles ni de nouveaux ensembles retenus.
10. Cross-Attention conserve ses requêtes cliniques, ses jetons DaTSCAN, IRM et
    démographie. Les jetons d'imagerie absents sont masqués. Le jeton
    démographique reste disponible, ce qui évite une attention entièrement
    masquée. Les deux modèles imagerie utilisent leurs 11 mesures et les
    contrôles normatifs âge/sexe déjà présents ; aucun score moteur, traitement
    ou ancrage résiduel clinique n'y est ajouté.
11. L'évaluation réunit toutes les prédictions, puis calcule les métriques
    globales : aucune moyenne de MAE par batch. Chaque patient compte une fois
    par horizon. Les préfixes train sont équipondérés, comme dans l'approche
    historique, pour toutes les configurations de séquences. Les patients avec
    davantage de préfixes ont donc davantage de poids pendant l'entraînement,
    ce qui est explicitement distinct de la pondération d'évaluation.
12. Les cibles manquantes sont masquées séparément. Une prédiction non finie
    pour une cible connue déclenche une erreur ; elle n'est jamais supprimée
    pour améliorer artificiellement une métrique.
13. Les barres de sous-domaines du front utilisaient des plafonds de 56 pour
    la bradykinésie et 16 pour l'axial. Le calcul réel dans l'ETL somme 11 items
    de bradykinésie et 7 items axiaux (0–4 chacun), soit 44 et 28. L'échelle
    visuelle est corrigée ; aucune valeur clinique ni métrique n'est éditée.

## Comparaisons et limites

- Après répétition des graines et avant toute nouvelle prédiction de test, un
  garde-fou de conservation a été ajouté : le candidat doit améliorer la MAE
  moyenne de validation et avoir une borne supérieure bootstrap appariée à
  95 % strictement négative. L'amendement est tracé dans
  `validation_safeguard_amendment.json`. Les anciens manifestes sont conservés.
  Il ne transforme pas le bootstrap sur validation après tuning en preuve
  indépendante et ne corrige pas toutes les comparaisons multiples.
- Le premier contrôle de sources a détecté deux écritures du même chemin
  Windows dans le manifeste, avant toute inférence de test. La normalisation
  des clés est désormais effectuée par `record_source` au gel. Aucun modèle,
  aucune cible et aucun score de test n'a été modifié pour résoudre ce point.

- `historical_reproduction_comparison.csv` confronte les prédictions réellement
  recalculées des anciens checkpoints au registre arrondi. Une différence
  signifie que le checkpoint et/ou le prétraitement actuellement disponibles
  ne reproduisent pas exactement le chiffre archivé. Les dates et versions
  historiques ne peuvent être réinventées.
- Les gains d'optimisation sont calculés **uniquement entre entraînements sur
  le protocole corrigé commun**, à patients, index, cibles et règles identiques.
  Une différence avec le tableau ancien ne peut pas être qualifiée de gain.
- Sélection par MAE moyenne des deux horizons pour un checkpoint multitâche
  entier ; XGBoost est sélectionné séparément par horizon. La RMSE, R² et
  Pearson sont rapportés séparément. Classification séparée, non sélectionnée
  comme objectif principal de cette demande de régression.
- Recherche structurée CPU ; Optuna n'est pas installé. Pilotes de deux
  époques avant la recherche complète. Trois graines prédéfinies, sans choix
  de la plus favorable ni ajout d'ensemble. Les métriques sont moyennées sur
  les graines ; les prédictions de plusieurs modèles ne sont pas fusionnées.
- Le test historique a déjà été consulté dans les travaux précédents, y
  compris ceux du 23 octobre. Il n'est plus entièrement indépendant. La
  sélection de cette expérience n'utilise cependant aucun score de test.
- `data/raw` est absent. Les jointures PATNO/EVENT_ID ne conservent pas toutes
  les dates d'acquisition. L'ordre intra-mois et la provenance temporelle des
  dernières informations démographiques ne sont pas certifiables ici.
  L'audit ne prétend donc pas établir une absence exhaustive de fuite.
- Les comparaisons n'excluent pas les patients sans imagerie ou les patients
  difficiles. La disponibilité est étudiée sans changer la cohorte évaluée.
- Les anciens checkpoints, Parquet et le registre historique sont protégés
  par SHA256. L'interface actuelle n'est pas alimentée par des chiffres édités.
