#!/usr/bin/env python3
"""Recompute quality and paired query bootstrap from saved ranking20 only."""
import argparse,csv,json,math,random,statistics
from collections import defaultdict
from pathlib import Path


def metric(r,k,name,contract='uuid_single_target'):
    targets={r['gold_id']} if contract=='uuid_single_target' else set(r['name_equivalent_ids'])
    rank=next((i+1 for i,x in enumerate(r['ranking20']) if x in targets),None)
    if not rank or rank>k:return 0.
    if name=='recall':return 1.
    if name=='mrr':return 1/rank
    if name=='ndcg':return 1/math.log2(rank+1)
    raise ValueError(name)


def percentile(x,q):
    x=sorted(x);pos=(len(x)-1)*q;i=int(pos);f=pos-i
    return x[i]*(1-f)+x[min(i+1,len(x)-1)]*f


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path(__file__).parent/'results');a=p.parse_args()
    rows=[json.loads(s) for s in (a.results/'per_case.jsonl').open()]
    by={(r['method'],r['size'],r['query_index']):r for r in rows}
    output=[]
    rng=random.Random(42)
    samples=[[rng.randrange(20) for _ in range(20)] for _ in range(10000)]
    for size in sorted({r['size'] for r in rows}):
        for treatment,control in [('vector','bm25'),('hybrid_rrf60','bm25'),('domain_vector_centroid','vector')]:
            for k in [5,10]:
                for name,contract in [(n,c) for n in ['recall','mrr','ndcg'] for c in ['uuid_single_target','name_equivalent_single_target']]:
                    diffs=[metric(by[treatment,size,i],k,name,contract)-metric(by[control,size,i],k,name,contract) for i in range(20)]
                    bootstrap=[sum(diffs[i] for i in s)/20 for s in samples]
                    output.append({'size':size,'treatment':treatment,'control':control,'k':k,'metric':name,
                                   'label_contract':contract,'n_paired_queries':20,'difference':statistics.mean(diffs),
                                   'bootstrap_lower':percentile(bootstrap,.025),'bootstrap_upper':percentile(bootstrap,.975),
                                   'wins':sum(d>0 for d in diffs),'losses':sum(d<0 for d in diffs),'ties':sum(d==0 for d in diffs),
                                   'scope':'supplement_new_router' if 'centroid' in treatment else 'primary_new_retrieval'})
    with (a.results/'paired_intervals.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]),lineterminator='\n');w.writeheader();w.writerows(output)
    ceilings=[]
    for method,size in sorted({(r['method'],r['size']) for r in rows}):
        sub=[r for r in rows if r['method']==method and r['size']==size]
        for contract,key in [('uuid_single_target','candidate_recall20'),('name_equivalent_single_target','name_candidate_recall20')]:
            value=statistics.mean(r[key] for r in sub)
            ceilings.append({'method':method,'size':size,'n':20,'label_contract':contract,'candidate_recall20':value,
                             'oracle_recall5':value,'oracle_mrr5':value,
                             'measured_jev_recall5':None,'measured_jev_latency_seconds':None,'measured_jev_cost_usd':None})
    with (a.results/'candidate_oracles.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ceilings[0]),lineterminator='\n');w.writeheader();w.writerows(ceilings)
    print(json.dumps({'paired_intervals':len(output),'candidate_oracle_cells':len(ceilings),
                      'bootstrap_seed':42,'bootstrap_replicates':10000,'unit':'paired query within size',
                      'caveat':'Exposed selected queries, exploratory intervals; no multiplicity correction or confirmatory generalization claim'}))

if __name__=='__main__':main()
