"""Strict external scorer: full qrel denominator, same candidates, paired baseline comparison."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from skillret_decider.core import (read_jsonl,write_json,write_jsonl,sha256_file,
    evaluate_predictions,baseline_predictions,paired_bootstrap)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,type=Path); p.add_argument('--split',choices=['validation','test'],required=True)
    p.add_argument('--predictions',required=True,type=Path); p.add_argument('--out',required=True,type=Path)
    p.add_argument('--zero-predictions',type=Path); p.add_argument('--approve-final-test',action='store_true')
    args=p.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    dm=json.loads((args.data/'manifest.json').read_text())
    if args.split=='test' and (not args.approve_final_test or not dm['is_test']): p.error('Final test scoring must be explicitly authorized')
    if args.split=='validation' and dm['is_test']: p.error('Cannot use test as validation')
    for name in (f'{args.split}.jsonl',f'{args.split}_labels.json'):
        if sha256_file(args.data/name)!=dm['files'][name]: raise ValueError(f'Changed input: {name}')
    examples=read_jsonl(args.data/f'{args.split}.jsonl')
    labels=json.loads((args.data/f'{args.split}_labels.json').read_text())
    expected={e['id']:e['retrieval_order'] for e in examples}
    def check(path):
        predictions=read_jsonl(path)
        for item in predictions:
            if item['query_id'] not in expected or item['candidate_skill_ids']!=expected[item['query_id']]:
                raise ValueError('Prediction candidate set/order changed or unknown query')
        return evaluate_predictions(predictions,labels,list(expected))
    adapted=check(args.predictions)
    baseline=evaluate_predictions(baseline_predictions(examples),labels,list(expected))
    report={'protocol':'metadata-only; BGE-M3 top15; NONE excluded for forced ranking; full qrel denominator',
        'dataset_revision':dm['dataset']['revision'],'is_official_paper_full_text_protocol':False,
        'bge_only':baseline['mean'],'selected_model':adapted['mean'],
        'paired_query_bootstrap_vs_bge':paired_bootstrap(baseline,adapted),
        'historical_CLM_comparison':'Separate full-catalog system. Historical score is context, not a controlled head-to-head.'}
    if args.zero_predictions:
        zero=check(args.zero_predictions)
        report['zero_shot']=zero['mean'];report['paired_query_bootstrap_vs_zero']=paired_bootstrap(zero,adapted)
    args.out.mkdir(parents=True)
    write_json(args.out/'summary.json',report)
    write_json(args.out/'selected.metrics.json',adapted)
    write_json(args.out/'bge.metrics.json',baseline)
    write_jsonl(args.out/'bge.predictions.jsonl',baseline_predictions(examples))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
