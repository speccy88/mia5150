"""Estimate next-stage workload from observed pilot timings, not guessed GPU prices."""
from __future__ import annotations
import argparse,json,statistics,math
from pathlib import Path
from skillret_decider.core import read_jsonl,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path);p.add_argument('--data',required=True,type=Path)
    p.add_argument('--hourly-cad',required=True,type=float,help='User-confirmed current CAD equivalent of the quoted Pod rate')
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    if args.hourly_cad<=0: p.error('A positive current rate is required; do not use an invented rate')
    cfg=json.loads((args.run/'config.json').read_text())
    split=json.loads((args.data/'split.json').read_text())
    events=read_jsonl(args.run/'events.jsonl'); train=[e for e in events if e['event']=='train']
    # Drop first 10 timing samples when available, while retaining them in the evidence.
    steady=train[10:] if len(train)>10 else train
    if not steady: raise ValueError('No timed training steps')
    seconds_per_query=sum(e['step_s'] for e in steady)/sum(e['queries'] for e in steady)
    pred=read_jsonl(args.run/'validation.zero-shot.predictions.jsonl')
    val_seconds_per_query=statistics.mean(e['seconds_per_query_amortized'] for e in pred)
    ntrain=len(split['train_query_ids']); nval=len(split['validation_query_ids'])
    training_s=ntrain*seconds_per_query
    validation_s=2*nval*val_seconds_per_query # baseline + end-of-epoch for the proposed full schedule
    hours=(training_s+validation_s)/3600
    write_json(args.out,{'full_train_queries':ntrain,'full_validation_queries':nval,
        'observed_training_seconds_per_query':seconds_per_query,
        'observed_validation_seconds_per_query':val_seconds_per_query,
        'estimated_one_epoch_train_and_two_validation_hours':hours,
        'quoted_hourly_cad':args.hourly_cad,'estimated_compute_cad_excluding_setup_and_mining':hours*args.hourly_cad,
        'conservative_2x_allowance_cad_excluding_setup_and_mining':2*hours*args.hourly_cad,
        'NOT_included':['dataset/model downloads','BGE embedding generation','tokenization','reload verification',
                        'checkpoint export','final test inference','storage','tax','idle paid time'],
        'limitations':'Pilot subset sequence lengths and kernel behavior may differ. This is a planning estimate, not a guarantee or authorization.',
        'next_gate':'Approve an explicit CAD budget and Pod teardown supervision before the full run.'})
    print(args.out.read_text())

if __name__=='__main__': main()
