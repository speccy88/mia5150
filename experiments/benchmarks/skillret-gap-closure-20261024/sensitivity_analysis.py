"""Frozen balanced24 uncertainty: query-clustered, seeds averaged within query."""
import csv,json,pathlib
import numpy as np
from ledger import canonical_attempts
HERE=pathlib.Path(__file__).resolve().parent
def csvout(name,rows):
    with (HERE/'report'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    protocol=json.loads((HERE/'protocol.json').read_text());qids=sorted(protocol['sensitivity_query_ids']);labels=json.loads((HERE/'evaluator-labels.json').read_text());data={}
    for model in ['qwen3_06_q8','qwen35_4b_q4']:
        with (HERE/f'results/{model}-cases.jsonl').open() as f:rows=canonical_attempts([json.loads(x) for x in f]).values()
        for r in rows:
            if r['query_id'] in qids:data[(model,r['query_id'],r['seed'],r['size'],r['method'])]=r
    strata={k:[q for q in qids if len(labels[q])==k] for k in [1,2,3]};assert all(len(q)==8 for q in strata.values())
    rng=np.random.default_rng(20261024);draws={k:rng.integers(0,8,(2000,8)) for k in strata};summary=[];contrasts=[]
    def estimate(values):
        observed=sum(float(np.mean([values[q] for q in strata[k]])) for k in strata)/3
        distribution=sum(np.array([values[q] for q in strata[k]])[draws[k]].mean(axis=1) for k in strata)/3
        return observed,float(np.quantile(distribution,.025)),float(np.quantile(distribution,.975))
    for n in [10,100,1000,6006]:
        for method in ['listing','bm25','vector','domain_vector','hybrid']:
            for metric in ['exact','f1']:
                means=[]
                for seed in [42,43,44]:means.append(np.mean([data[('qwen3_06_q8',q,seed,n,method)]['metrics'][metric] for q in qids]))
                values={q:np.mean([data[('qwen3_06_q8',q,seed,n,method)]['metrics'][metric] for seed in [42,43,44]]) for q in qids}
                point,lo,hi=estimate(values)
                summary.append({'size':n,'method':method,'metric':metric,'unique_queries':24,'catalog_seeds':3,'balanced_macro_seed_averaged':point,'query_bootstrap_ci95_low':lo,'query_bootstrap_ci95_high':hi,'seed42_mean':means[0],'seed43_mean':means[1],'seed44_mean':means[2],'seed_mean_sample_sd':float(np.std(means,ddof=1)),'scope':'2000 stratified paired-query resamples; average seeds within query; failed decisions zero; nominal exploratory intervals. Catalog/presentation seeds are not training replicates.'})
                differences={q:data[('qwen35_4b_q4',q,42,n,method)]['metrics'][metric]-data[('qwen3_06_q8',q,42,n,method)]['metrics'][metric] for q in qids}
                point,lo,hi=estimate(differences)
                contrasts.append({'size':n,'method':method,'metric':metric,'unique_queries':24,'seed':42,'medium_minus_small_balanced_macro':point,'paired_query_bootstrap_ci95_low':lo,'paired_query_bootstrap_ci95_high':hi,'scope':'2000 same-query stratified resamples; nominal descriptive model-deployment sensitivity; architecture/training/quantization confounded. Not multiple-comparison adjusted or a model-size causal claim.'})
    csvout('seed-query-bootstrap.csv',summary);csvout('model-paired-bootstrap.csv',contrasts);print('Balanced24 seed-averaged and paired-model uncertainty regenerated')
if __name__=='__main__':main()
