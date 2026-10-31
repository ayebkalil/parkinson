"""Bounded subprocess orchestration; no agents, distributed training or ensembles.

At most two CPU jobs, each using the thread limit saved in the experiment.
Selection, freezing and test evaluation are barriers after all development jobs.
All completed checkpoints are reused on resume instead of training them again.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

FAMILIES=['tabular_xgb','bilstm','late','early','cross','imaging_mlp','imaging_xgb']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='experiments/seven_models_20261030')
    parser.add_argument('--workers',type=int,choices=[1,2],default=2);args=parser.parse_args()
    root=Path(args.output);logs=root/'logs';logs.mkdir(exist_ok=True)
    def execute(stage,family=None):
        name=stage+('_'+family if family else '')
        command=[sys.executable,'-u','src/seven_model_experiments.py','--stage',stage,'--output',str(root)]
        if family:command+=['--family',family]
        with (logs/f'{name}.log').open('ab') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            code=child.wait()
        if code:raise RuntimeError(f'{name} failed with code {code}; see {logs/name}.log')
        print('COMPLETE',name,flush=True)
    for stage in ['screen','repeat']:
        (root/'batch_state.json').write_text(json.dumps(dict(stage=stage,workers=args.workers,started=time.time())),encoding='utf-8')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending=[pool.submit(execute,stage,family) for family in FAMILIES]
            for future in as_completed(pending):future.result()
        if stage=='screen':execute('shortlist')
    for stage in ['select','test','report','verify']:execute(stage)
    (root/'batch_state.json').write_text(json.dumps(dict(stage='complete',finished=time.time())),encoding='utf-8')


if __name__=='__main__':main()
