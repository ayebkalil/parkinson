"""Generate figures and subgroup diagnostics from frozen, computed predictions."""
import argparse
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'parkinson-matplotlib'))
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seven_model_protocol as p

LABELS={'tabular_xgb':'XGBoost tabulaire','bilstm':'BiLSTM longitudinal','late':'Late Fusion',
        'early':'Early Fusion','cross':'Cross-Attention','imaging_mlp':'Imagerie · Deep MLP','imaging_xgb':'Imagerie · XGBoost'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('experiments/seven_models_20261030'))
    args=parser.parse_args();root=args.output
    comparison=pd.read_csv(root/'before_after_comparison.csv')
    intervals=pd.read_csv(root/'paired_patient_bootstrap.csv')
    data=comparison[comparison.split=='test'].copy()
    fig,axes=plt.subplots(1,2,figsize=(16,6.5),sharex=True)
    for ax,h in zip(axes,[12,24]):
        group=data[data.horizon==h].set_index('family').loc[p.FAMILIES]
        y=np.arange(len(group))
        ax.barh(y-.17,group.baseline_mae,height=.29,color='#9c84c7',label='Référence corrigée harmonisée',xerr=group.baseline_mae_std,error_kw={'ecolor':'#65537d','capsize':3,'linewidth':1})
        ax.barh(y+.17,group.selected_mae,height=.29,color='#64b7a5',label='Configuration retenue sur validation',xerr=group.selected_mae_std,error_kw={'ecolor':'#376d60','capsize':3,'linewidth':1})
        ax.set_yticks(y,[LABELS[k] for k in group.index]);ax.invert_yaxis()
        ax.set_title(f'+{h} mois · {int(group.n_patients.iloc[0])} patients',fontsize=13,pad=14)
        ax.set_xlabel('MAE (points MDS-UPDRS III) · moyenne ± écart type de 3 graines',fontsize=9)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#dddddd');ax.tick_params(axis='both',length=0,labelsize=9)
    axes[0].legend(loc='upper center',bbox_to_anchor=(1.12,-.14),ncol=2,frameon=False,fontsize=9)
    fig.suptitle('Sept familles conservées · comparaison sur un protocole corrigé commun',fontsize=16,y=.98)
    fig.text(.5,.925,'Test historique déjà consulté : ces résultats ne constituent pas une validation indépendante.',ha='center',fontsize=10,color='#7c687e')
    fig.tight_layout(rect=[0,.06,1,.88]);fig.savefig(root/'before_after_test.png',dpi=200,bbox_inches='tight');plt.close(fig)

    # Subgroups are diagnostics after the frozen evaluation, never selection criteria.
    longitudinal=pd.read_parquet(root/'data/test_longitudinal.parquet').sort_values(['PATNO','visit_date','row_id'],kind='stable')
    filled=longitudinal.groupby('PATNO')[p.fusion.IMAGING_COLS].ffill()
    longitudinal['imaging_available']=filled.notna().any(axis=1)
    metadata=longitudinal[['PATNO','row_id','COHORT_DEFINITION','pd_state_clean','imaging_available']]
    subgroup=[]
    for file in (root/'test_predictions').glob('*/*/*/*m.csv'):
        family,role,seed,horizon=file.relative_to(root/'test_predictions').parts
        pred=pd.read_csv(file).merge(metadata,on=['PATNO','row_id'],validate='one_to_one')
        for feature in ['COHORT_DEFINITION','pd_state_clean','imaging_available']:
            for value,group in pred.groupby(feature,dropna=False):
                if len(group)<2:continue
                subgroup.append(dict(family=family,role=role,seed=int(seed),horizon=int(horizon[:-5]),
                    subgroup=feature,value=str(value),**p.regression(group.true,group.pred)))
    pd.DataFrame(subgroup).to_csv(root/'test_subgroups_diagnostic.csv',index=False)

    summary=data.merge(intervals,on=['family','horizon','n_patients'],suffixes=('','_bootstrap'))
    summary['paired_ci_supports_lower_mae']=(summary.ci95_high<0)
    summary.to_csv(root/'test_interpretation.csv',index=False)
    available=json.loads((root/'best_available_checkpoints.json').read_text(encoding='utf-8'))
    reused=sorted({a['family'] for a in available if a['origin']=='previously_retained'})
    counts={'test_comparisons':len(summary),'lower_mae_point_estimate':int((summary.delta_mae<0).sum()),
        'paired_ci_supports_lower_mae':int(summary.paired_ci_supports_lower_mae.sum()),
        'best_available_reuses_previous_families':reused,
        'test_is_independent':False,'production_checkpoints_replaced':False}
    p.write_json(root/'conclusions.json',counts)
    chain=[]
    for family in ['tabular_xgb','imaging_xgb']:
        for h in [12,24]:
            previous=None
            for name,label in [('baseline','native_missing'),('causal_locf','causal_locf_only'),('robust_causal_locf','regularized_robust_plus_locf')]:
                results=[json.loads(file.read_text(encoding='utf-8'))['metrics']
                    for file in (root/'final'/family/name).glob(f'*/{h}m/metrics.json')]
                if len(results)!=3:raise ValueError('Three seeds required for preprocessing attribution')
                row=dict(family=family,horizon=h,stage=label,seeds=3,n_patients=results[0]['n_patients'])
                for metric in ['mae','rmse','r2','pearson_r']:
                    values=[r[metric] for r in results];row[metric]=float(np.mean(values));row[metric+'_std']=float(np.std(values,ddof=1))
                row['delta_mae_previous_stage']=row['mae']-previous if previous is not None else None
                previous=row['mae'];chain.append(row)
    pd.DataFrame(chain).to_csv(root/'validation_preprocessing_chain.csv',index=False)
    addition=['','LECTURE DES RÉSULTATS',
        f"MAE test plus faible en estimation ponctuelle : {counts['lower_mae_point_estimate']} / {len(summary)} comparaisons.",
        f"Intervalle patient apparié entièrement sous zéro : {counts['paired_ci_supports_lower_mae']} / {len(summary)} comparaisons.",
        'Ces intervalles sont conditionnels aux trois graines et ne corrigent pas le fait que le test historique a déjà été consulté.',
        'Un delta MAE négatif représente une baisse. Les configurations restent celles choisies sur validation, même si leur résultat de test est moins bon.',
        'La référence corrigée harmonise aussi le budget et certains paramètres multitâches ; aucun effet isolé des seules corrections n’est revendiqué par rapport au tableau historique invalide.',
        'Pour XGBoost, validation_preprocessing_chain.csv sépare les valeurs manquantes natives, le seul remplissage causal et l’ajout de régularisation/perte robuste. Le gain total ne doit pas être attribué à la seule perte.',
        'Les ablations de développement sont séparées et ne peuvent pas modifier rétroactivement la sélection.',
        'best_available_checkpoints.json compare également les anciens checkpoints retenus sur la même validation. Un ancien checkpoint réutilisé ne constitue pas un nouveau gain.',
        'Familles où une version antérieure est conservée dans ce manifeste : '+(', '.join(reused) if reused else 'aucune'),
        'Aucun checkpoint historique de production ni résultat du dashboard n’a été remplacé manuellement.',
        '', 'SYNTHÈSE TEST :',summary[['family','horizon','n_patients','baseline_mae','selected_mae','delta_mae','ci95_low','ci95_high']].to_string(index=False)]
    with (root/'RAPPORT_FR.txt').open('a',encoding='utf-8') as stream:stream.write('\n'.join(addition)+'\n')


if __name__=='__main__':main()
