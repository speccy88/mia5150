#!/usr/bin/env python3
"""RunPod-only final diagnostics; complete-pool causal comparisons kept separate from operational missing-output metrics."""
import argparse,json
from pathlib import Path
import runner2 as r

def main(root):
    import numpy as np
    from metrics_existing import evaluate,paired_bootstrap,deterministic_subset
    data=r.load_data(root);labels=r.labels_for(data);ranks={x['query_id']:[h['skill_id'] for h in x['hits']] for x in r.lines(root/'results/full_6656_top100.jsonl')}
    def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default
    v2=root/'checkpoints/decisions_v2';old=root/'checkpoints/decisions';pairs={}
    # Old data imported first; explicit v2 per-pair checkpoints take precedence.
    for p in old.glob('*.json'):
        row=read(p,{})
        for x in row.get('candidates',[]):pairs[(row['query_id'],x['skill_id'])]={'query_id':row['query_id'],**x}
    for p in v2.glob('*.json'):
        x=read(p,{})
        if x.get('mode')=='pair':pairs[(x['query_id'],x['skill_id'])]=x
    protocol=read(root/'manifests/decision_v2_protocol.json',{})
    chosen=protocol.get('query_ids') or read(root/'manifests/decision_query_ids.json') or deterministic_subset(list(labels),200)
    matched=[];operational=[]
    for k in (10,20,50):
        complete=[]
        for q in chosen:
            if all((q,s) in pairs and pairs[(q,s)].get('status')=='ok' for s in ranks[q][:k]):complete.append(q)
        for mode in ('binary','ordinal'):
            scores={};ranked={}
            for q in chosen:
                rr=[pairs[(q,s)] for s in ranks[q][:k] if (q,s) in pairs and pairs[(q,s)].get('status')=='ok'];pos={s:i for i,s in enumerate(ranks[q])}
                score=lambda x:float(x['output']['questions'][mode]['probabilities']['true']) if mode=='binary' else float(x['output']['questions'][mode]['value'])
                ranked[q]=[x['skill_id'] for x in sorted(rr,key=lambda x:(-score(x),pos[x['skill_id']]))]
            op=evaluate({q:labels[q] for q in chosen},ranked)
            operational.append({'mode':mode,'candidate_count':k,'intended_query_count':len(chosen),'complete_query_count':len(complete),'metrics':op['aggregate'],'interpretation':'Zero-filled operational completion measurement; NOT an estimate of reranker model regression or improvement.'})
            entry={'mode':mode,'candidate_count':k,'complete_query_count':len(complete),'intended_query_count':len(chosen),'complete_query_ids':complete,'selection_warning':'Completed-query subset may be unrepresentative, especially when budget-limited.'}
            if not complete:entry.update(status='unavailable',reason='No queries with all k candidates successfully scored; no matched-pool effect computed.')
            else:
                gold={q:labels[q] for q in complete};base=evaluate(gold,{q:ranks[q][:k] for q in complete});treatment=evaluate(gold,{q:ranked[q] for q in complete})
                entry.update(status='available',cosine=base['aggregate'],decision=treatment['aggregate'],delta={m:treatment['aggregate'][m]-v for m,v in base['aggregate'].items()},per_query_cosine=base['per_query'],per_query_decision=treatment['per_query'],paired_bootstrap=paired_bootstrap(base,treatment,n_resamples=1000,seed=42))
            matched.append(entry)
    r.dump(root/'results/decision_matched_diagnostics.json',matched);r.dump(root/'results/decision_operational_diagnostics.json',operational)
    # Native API and manual encode use same model/instructions/corpus, different query batch shapes.
    native=read(root/'results/native_retrieve.json',{})
    rows=native.get('rows',[])
    r.dump(root/'results/native_equivalence_diagnostics.json',{'observed_query_count':len(rows),'exact_top100_order_rate':sum(x['same_top100_order'] for x in rows)/len(rows) if rows else None,'same_top100_set_rate':sum(x['same_top100_set'] for x in rows)/len(rows) if rows else None,'differing_query_ids':[x['query_id'] for x in rows if not x['same_top100_order']],'explanation':'Native retrieve re-encodes each query individually, while saved query vectors may use batches of four. BF16 kernels and padded batch shapes can change last-bit results; native torch.argsort tie behavior also differs from stable catalog-order NumPy sorting. These are possible causes, not proven explanations of every observed difference. Query truncation/chunk aggregation is another protocol difference if any query exceeded16384 tokens. Matching output is not an independent model improvement.','latency':'Native includes query embedding plus search, whereas cached cosine does not.'})
    telemetry=[]
    for kind in ('documents','queries'):
        rows=read(root/f'manifests/{kind}_token_audit.json',[])
        for basis in ('single_input','batch_elapsed_divided_by_input_count'):
            selected=[x for x in rows if x.get('status')=='ok' and x.get('latency_basis','single_input')==basis]
            # First8 completed rows per kind excluded conservatively from warm distribution.
            warm_ids={x['id'] for x in rows[8:]};warm=[x for x in selected if x['id'] in warm_ids];lat=[x['seconds'] for x in warm]
            elapsed=sum(x['seconds'] for x in selected);tokens=sum(sum(x.get('chunk_tokens',[x['tokens']])) for x in selected)
            telemetry.append({'kind':kind,'latency_basis':basis,'input_count':len(selected),'warm_input_count':len(warm),'warm_exclusion':'first8 completed inputs per kind','p50_seconds':float(np.quantile(lat,.5)) if lat else None,'p95_seconds':float(np.quantile(lat,.95)) if lat else None,'measured_encoding_seconds':elapsed,'formatted_tokens_processed':tokens,'tokens_per_second':tokens/elapsed if elapsed else None,'peak_gpu_allocated_bytes':max((x['peak_gpu_bytes'] for x in selected),default=None),'note':'Batch-amortized seconds are throughput allocations, not single-request latency. Tokenization/checkpoint I/O excluded from encoding timer; original-run timings may include small bookkeeping.'})
    r.dump(root/'results/performance_diagnostics.json',telemetry)
    dims=read(root/'results/retrieval_summary.json',[]);full=next((x for x in dims if x['configuration']=='full_6656'),None);eligible=[]
    if full:
        eligible=[x['dimension'] for x in dims if x['configuration'].startswith('full') and abs(x['NDCG@10']-full['NDCG@10'])<=.01 and abs(x['Recall@10']-full['Recall@10'])<=.01]
    r.dump(root/'results/dimension_tolerance.json',{'criterion':'absolute NDCG@10 AND Recall@10 difference from full6656 each <=0.01','tolerance':.01,'smallest_qualifying_dimension':min(eligible) if eligible else None,'qualifying_dimensions':eligible,'native_support':False,'mechanism':'Exploratory prefix truncation and L2 renormalization; full6656 retained.','criterion_frozen_before_result_review':True})
    # Replace figure4 with complete-pool effects only. Never plot incomplete scoring as model regression.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'figures';out.mkdir(exist_ok=True);available=[x for x in matched if x['status']=='available'];fig,ax=plt.subplots(figsize=(10,5.5))
    if available:
        ax.bar([f"{x['mode']} k{x['candidate_count']} n={x['complete_query_count']}" for x in available],[x['delta']['NDCG@10'] for x in available]);ax.axhline(0,color='black',linewidth=.8);ax.set(ylabel='Matched-complete-query ΔNDCG@10',title='Decision vs same-query cosine pool; completed subset may be biased');ax.tick_params(axis='x',rotation=15)
    else:ax.text(.5,.5,'Matched decision effect unavailable:\nno candidate pools completely scored.',ha='center',va='center');ax.set_axis_off()
    fig.tight_layout();fig.savefig(out/'04_decision_changes.png',dpi=180);plt.close(fig);r.dump(out/'04_decision_changes.json',matched)
    import csv
    with (out/'04_decision_changes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['mode','candidate_count','complete_query_count','status','delta_NDCG10']);w.writeheader();w.writerows({'mode':x['mode'],'candidate_count':x['candidate_count'],'complete_query_count':x['complete_query_count'],'status':x['status'],'delta_NDCG10':x.get('delta',{}).get('NDCG@10')} for x in matched)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
