"""Read completed, verified experiments; no hard-coded performance values."""
import json
from pathlib import Path

LABELS={'tabular_xgb':'Tabular XGBoost','bilstm':'Longitudinal BiLSTM',
    'late':'Late Fusion','early':'Early Fusion','cross':'Cross-Attention Transformer',
    'imaging_mlp':'Monomodal Neuroimaging (Deep MLP)','imaging_xgb':'Monomodal Neuroimaging (XGBoost)'}


def audited_benchmarks(root: Path):
    folder=root/'experiments/seven_models_20261030'
    required=['verification.json','before_after_all_metrics.csv','before_after_comparison.csv','freeze.json']
    if not all((folder/name).is_file() for name in required):return None
    verified=json.loads((folder/'verification.json').read_text(encoding='utf-8'))
    if not verified.get('protected_files_unchanged') or not verified.get('checkpoint_hashes_valid'):return None
    import pandas as pd
    metrics=pd.read_csv(folder/'before_after_all_metrics.csv')
    comparison=pd.read_csv(folder/'before_after_comparison.csv')
    rows=[]
    for item in metrics[(metrics.split=='test')&(metrics.role=='selected')].itertuples():
        match=comparison[(comparison.split=='test')&(comparison.family==item.family)&(comparison.horizon==item.horizon)].iloc[0]
        rows.append(dict(model_name=LABELS[item.family],target_horizon=f'+{item.horizon} Months (NP3TOT)',
            n_samples=int(item.n_patients),mae=float(item.mae),mae_std=float(item.mae_std),
            rmse=float(item.rmse),r2=float(item.r2),pearson_r=float(item.pearson_r),
            baseline_mae=float(match.baseline_mae),delta_mae=float(match.delta_mae),
            relative_delta_percent=float(match.relative_delta_percent),n_seeds=int(item.seeds),
            notes='Protocole corrigé commun ; configuration choisie sur validation ; moyenne des métriques de trois entraînements, sans ensemble de prédictions.'))
    if len(rows)!=14:raise ValueError('Incomplete audited benchmark: expected seven families and two horizons')
    return dict(items=rows,historical=False,protocol='audited',
        note='Test historique déjà consulté : validation indépendante encore nécessaire. Même cohorte et mêmes cibles pour chaque comparaison avant/après. Moyennes et écarts types de trois graines. Les checkpoints des expériences sont distincts des modèles actifs du dashboard.')
