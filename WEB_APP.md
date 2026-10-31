# Neurovia — React + FastAPI

Interface française pour explorer les données PPMI et les modèles existants du
projet. Le tableau de bord Streamlit historique reste disponible dans `app.py`.

## Lancer l’application

Depuis la racine, dans un terminal Windows :

```bat
run_project.bat
```

Puis ouvrir **http://127.0.0.1:8000**. Le même serveur distribue le front React
compilé et l’API FastAPI. La documentation interactive est à `/docs`.

Première installation, ou après modification des dépendances :

```bat
run_project.bat -Install
```

Après une modification du front :

```bat
run_project.bat -Build
```

Options : `-NoBrowser` pour ne pas ouvrir un onglet, `-Port 8001` pour choisir un
autre port. Ctrl+C arrête le serveur. Le lanceur utilise `.venv` directement ;
il ne change pas la politique d’exécution Windows persistante. Python et Node.js
sont requis. Les dépendances Python sont dans `requirements.txt`, celles de React
dans `frontend/package.json`, avec les versions résolues dans `package-lock.json`.

## Développement avec rechargement

Terminal backend, depuis la racine :

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Terminal frontend :

```powershell
cd frontend
npm.cmd run dev
```

Ouvrir http://127.0.0.1:5173. Vite relaie `/api` vers FastAPI. En production
locale, `npm.cmd run build` produit `frontend/dist`, servi par FastAPI.

## Fonctionnement

- L’introduction lance la vidéo Pinterest fournie, en sourdine et en boucle.
  Elle appelle `POST /api/initialize` et interroge l’état réel du moteur via
  `GET /api/status` pendant la lecture. Les modèles sont chargés une seule fois
  dans un thread ; l’API reste disponible pendant cette préparation.
- La progression reflète des étapes réellement terminées (imports, données,
  normalisation, Fusion, BiLSTM, XGBoost). Les modèles sont déjà présents sur
  disque : il s’agit d’un chargement en mémoire, pas d’un téléchargement réseau.
- Une courte durée de présentation rend l’introduction lisible lors des
  démarrages à chaud, même si les modèles sont déjà en mémoire. Une transition
  mène au tableau de bord, où la même image neuronale est reprise.
- La vidéo et les images sont dans `frontend/public/media`. Les sources exactes
  sont documentées dans `SOURCE.txt`. Les polices sont locales. Aucune ressource
  externe n’est nécessaire pour afficher l’interface après installation.
- L’interface propose les profils patients, les prévisions à 12/24 mois, le
  changement de modèle, l’attention temporelle BiLSTM, les pondérations Late
  Fusion, les sous-scores moteurs, la recherche, les filtres, la pagination,
  les résultats historiques, l’export CSV et les préférences d’animation.
- Une erreur de connexion ou de chargement est affichée avec une action de
  réessai. Une image de secours reste disponible si la lecture vidéo échoue.
  La préférence système de réduction des animations est respectée.

## Modèles et données

`backend/service.py` reprend le prétraitement et l’inférence de `app.py`, sans
entraîner de modèle ni changer les checkpoints ou les partitions.

Les modèles servis sont `models/late_fusion.pt`, `models/model2_bilstm.pt` et les
deux régresseurs XGBoost retenus dans `experiments/model12_round2_20261023`.
Les données XGBoost alignées viennent de `experiments/model12_20261023/data`.
Les deux horizons XGBoost peuvent avoir des visites index différentes ; une
prévision sans visite éligible est renvoyée à `null` et affichée comme indisponible.
La probabilité de progression est propre au modèle sélectionné et n’est pas
affichée pour XGBoost, qui sert ici uniquement des régresseurs.

Les courbes ne présentent aucun intervalle de confiance fabriqué. Les
pondérations de fusion sont décrites comme des poids appris, pas comme une
explication causale. Le registre de benchmarks est identifié comme historique
et ne décrit pas les nouveaux checkpoints XGBoost retenus. Les cohortes ont
déjà été consultées au cours du projet ; une validation indépendante reste
nécessaire. L’application ne donne pas de recommandation thérapeutique.

Le serveur écoute uniquement sur `127.0.0.1`. Cette application de recherche
locale n’inclut pas de comptes utilisateurs ni de déploiement public.

## Vérification

```powershell
.\.venv\Scripts\python.exe -m unittest backend.test_api -v
cd frontend
npm.cmd run build
```

Les tests couvrent le chargement asynchrone et son idempotence, les filtres,
les erreurs 404/422/503, les horizons absents, l’export CSV et la parité des
prédictions avec le dashboard historique. Ils lisent les artefacts existants.
