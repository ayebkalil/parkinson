"""Prespecified follow-up to missingness audit, before any new test evaluation.

Only XGBoost input imputation is changed. Includes an isolated LOCF ablation
and one robust-loss configuration with the same causal imputation. All three
seeds are measured on validation; no test metrics are read.
"""
import argparse
import json
from pathlib import Path
from seven_model_experiments import read,config_for,fit_xgb,gather,record_source
import seven_model_protocol as p


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('experiments/seven_models_20261030'))
    args=parser.parse_args();root=args.output
    if (root/'freeze.json').exists() or (root/'test_metrics.csv').exists():
        raise RuntimeError('This development extension must precede the final freeze and test')
    config=read(root/'config.json')
    for key in ['xgb_trials','imaging_xgb_trials']:
        config[key]['causal_locf']={**config[key]['baseline'],'locf':True}
        config[key]['robust_causal_locf']={**config[key]['pseudo_huber'],'locf':True}
    p.write_json(root/'config.json',config)
    p.write_json(root/'preprocessing_extension_policy.json',dict(reason='Development audit found ~76% missing imaging values at index visits',
        change='Causal LOCF before the exact index row; same feature columns, patients and targets',
        seeds=config['final_seeds'],selection='One candidate per horizon by screening seed 42, confirmed by mean of all three seeds',
        test_consulted=False))
    # The extension adds optional preprocessing only; default model training
    # remains unchanged. Snapshot the extended implementation before fitting.
    record_source(root)
    for family in ['tabular_xgb','imaging_xgb']:
        for name in ['causal_locf','robust_causal_locf']:
            cfg=config_for(config,family,name)
            fit_xgb(root,family,cfg,42,config,'runs')
            for seed in config['final_seeds']:fit_xgb(root,family,cfg,seed,config,'final')
    table=gather(root);gather(root,'final')
    shortlist=read(root/'shortlist.json')
    for family in ['tabular_xgb','imaging_xgb']:
        for h in [12,24]:
            ranked=table[(table.family==family)&(table.horizon==h)&(table.config!='baseline')].groupby('config').mae.mean().sort_values()
            shortlist['challengers'][family][f'{h}m']=ranked.index[0]
    p.write_json(root/'shortlist.json',shortlist)
    print('PREPROCESSING_EXTENSION_COMPLETE',json.dumps(shortlist['challengers'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
