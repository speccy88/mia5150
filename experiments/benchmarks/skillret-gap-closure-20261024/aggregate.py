"""Independent saved-output audit, paired uncertainty, and regenerable tables."""
import collections,csv,gzip,json,pathlib,sys
import numpy as np
from core import METHODS,SIZES,alias,digest,render,select_prompt
from ledger import canonical_attempts
HERE=pathlib.Path(__file__).resolve().parent
OUT=HERE/'report'

def read(p):return json.loads(pathlib.Path(p).read_text())
def rows(p):
    with (gzip.open(p,'rt') if pathlib.Path(p).suffix=='.gz' else open(p)) as f:return [json.loads(x) for x in f if x.strip()]
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def csvout(name,data):
    if not data:return
    keys=list(dict.fromkeys(k for d in data for k in d))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
def mean(x):return float(np.mean(x)) if x else None
def pct(x,q):return float(np.quantile(x,q)) if x else None

def independent_score(row,gold):
    pred=set(row['selected_ids']) if row['status']=='ok' else set();g=set(gold)
    tp=len(pred&g);fp=len(pred-g);fn=len(g-pred)
    if row['status']!='ok':fp=row.get('invalid_return_count') or 0
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/len(g)
    return {'exact':int(row['status']=='ok' and pred==g),'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.,'tp':tp,'fp':fp,'fn':fn}

def summarize_raw(sub,scope,model,n,method):
    valid=[r for r in sub if r['status']=='ok'];exact=sum(r['metrics']['exact'] for r in sub)
    def values(key,items=sub):return [r[key] for r in items if r.get(key) is not None]
    tp=sum(r['metrics']['tp'] for r in sub);fp=sum(r['metrics']['fp'] for r in sub);fn=sum(r['metrics']['fn'] for r in sub)
    p=tp/(tp+fp) if tp+fp else 0.;rec=tp/(tp+fn) if tp+fn else 0.
    fresh=[r for r in valid if not r.get('routing_reused',False) and r.get('_latency_eligible',True)]
    return {'scope':scope,'model':model,'size':n,'method':method,'planned':len(sub),'unique_queries':len({r['query_id'] for r in sub}),'selector_calls':sum(r['selector']['physical_call'] for r in sub if r.get('selector')),'route_calls':sum(not r.get('routing_reused',False) and r['routing']['physical_call'] for r in sub if r.get('routing')),'valid':len(valid),'exact_successes':int(exact),'exact_all_planned':exact/len(sub),'exact_valid_only':exact/len(valid) if valid else None,'macro_precision':mean([r['metrics']['precision'] for r in sub]),'macro_recall':mean([r['metrics']['recall'] for r in sub]),'macro_f1':mean([r['metrics']['f1'] for r in sub]),'micro_precision':p,'micro_recall':rec,'micro_f1':2*p*rec/(p+rec) if p+rec else 0.,'candidate_recall':mean(values('candidate_recall')),'retrieval_recall5':mean([r['retrieval']['recall5'] for r in sub if r.get('retrieval')]),'retrieval_mrr5':mean([r['retrieval']['mrr5'] for r in sub if r.get('retrieval')]),'tokenized_logical_input_mean_including_preflight':mean(values('input_tokens')),'valid_logical_input_mean':mean(values('input_tokens',valid)),'valid_logical_output_mean':mean(values('output_tokens',valid)),'embedding_tokens_mean':mean(values('embedding_tokens')),'fresh_valid_e2e_count':len(fresh),'fresh_valid_e2e_mean_s':mean(values('end_to_end_seconds',fresh)),'fresh_valid_e2e_p50_s':pct(values('end_to_end_seconds',fresh),.5),'fresh_valid_e2e_p95_s':pct(values('end_to_end_seconds',fresh),.95),'failed_physical_wall_mean_s':mean(values('end_to_end_seconds',[r for r in sub if r['status']!='ok'])),'context_overflow':sum(r['status']=='context_overflow' for r in sub),'not_run':sum(r['status'].startswith('not_run') for r in sub),'invalid_selection':sum(r['status']=='invalid_selection' for r in sub),'empty_or_invalid_domain':sum(r['status']=='invalid_or_empty_domain' for r in sub),'abstentions':sum(r['status']=='ok' and not r['selected_ids'] for r in sub)}

def summarize(sub,scope,model,n,method):
    result=summarize_raw(sub,scope,model,n,method);unknown=[]
    result['candidate_recall_conditional_observed']=result['candidate_recall']
    result['retrieval_recall5_conditional_observed']=result['retrieval_recall5']
    result['retrieval_mrr5_conditional_observed']=result['retrieval_mrr5']
    result['retrieval_completed_cases']=sum(r.get('retrieval') is not None for r in sub)
    result['tokenized_input_known_cases']=sum(r.get('input_tokens') is not None for r in sub)
    if method=='listing':result['candidate_recall']=1.0
    else:
        result['candidate_recall']=sum(r.get('candidate_recall') or 0 for r in sub)/len(sub)
        result['retrieval_recall5']=sum((r.get('retrieval') or {}).get('recall5',0) for r in sub)/len(sub)
        result['retrieval_mrr5']=sum((r.get('retrieval') or {}).get('mrr5',0) for r in sub)/len(sub)
    for row in sub:
        if row.get('returned_cardinality_unknown'):unknown.append(row);continue
        answer=row.get('selector')
        if answer and answer.get('physical_call') and row['status']!='ok':
            if answer.get('status')!='ok' or (row['status']=='invalid_selection' and row.get('invalid_return_count') is None):unknown.append(row)
    unknown_ids={r['case_id'] for r in unknown};known=[r for r in sub if r['case_id'] not in unknown_ids]
    tp=sum(r['metrics']['tp'] for r in known);fp=sum(r['metrics']['fp'] for r in known);fn=sum(r['metrics']['fn'] for r in known)
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    result.pop('micro_precision');result.pop('micro_f1')
    result['micro_recall_decision_zero_on_failure']=result.pop('micro_recall')
    result.update(micro_precision_parseable_only=precision,micro_f1_parseable_only=2*precision*recall/(precision+recall) if precision+recall else 0.,parseable_or_known_no_output_cases=len(known),unknown_returned_cardinality_cases=len(unknown),micro_precision_all_planned=None if unknown else precision,micro_f1_all_planned=None if unknown else (2*precision*recall/(precision+recall) if precision+recall else 0.),micro_precision_scope='Parseable selections and known no-output cases; unknown cardinality excluded and counted. Zero when prediction denominator is zero.')
    result['unknown_model_usage_calls']=sum(bool(a.get('physical_call')) and (a.get('status')=='transport_or_guard_error' or a.get('input_tokens') is None or a.get('output_tokens') is None) for r in sub for a in [r.get('selector'),r.get('routing')] if a)
    return result

def bootstrap(primary,labels):
    qids=sorted({r['query_id'] for r in primary});lookup={(r['query_id'],r['size'],r['method']):r for r in primary}
    strata={k:[q for q in qids if len(labels[q])==k] for k in [1,2,3]};weights={1:2143/4392,2:1703/4392,3:546/4392}
    rng=np.random.default_rng(20261024);samples={k:rng.integers(0,len(v),size=(2000,len(v))) for k,v in strata.items()}
    result=[]
    for n in SIZES:
        for a,b in [('hybrid','bm25'),('domain_vector','vector')]:
            estimates={k:np.array([lookup[(q,n,a)]['metrics']['exact']-lookup[(q,n,b)]['metrics']['exact'] for q in v]) for k,v in strata.items()}
            distribution=sum(weights[k]*estimates[k][samples[k]].mean(axis=1) for k in strata)
            delta=sum(weights[k]*estimates[k].mean() for k in strata)
            result.append({'size':n,'contrast':a+' minus '+b,'unique_queries':len(qids),'weighted_exact_difference':float(delta),'ci95_low':float(np.quantile(distribution,.025)),'ci95_high':float(np.quantile(distribution,.975)),'bonferroni8_nominal_low':float(np.quantile(distribution,.05/16)),'bonferroni8_nominal_high':float(np.quantile(distribution,1-.05/16)),'resamples':2000,'clustering':'query paired within gold-count stratum; identical draws across sizes/contrasts','weights':'2143/4392,1703/4392,546/4392','interpretation':'Exposed exploratory evidence; percentile and nominal family-adjusted intervals; no guaranteed power.'})
    return result

def main():
    OUT.mkdir(exist_ok=True);protocol=read(HERE/'protocol.json');labels=read(HERE/'evaluator-labels.json');panel=read(HERE/'panel.json');sensitivity=set(protocol['sensitivity_query_ids'])
    allrows=[];validation={'status':'passed','case_rows':0,'unique_case_ids':True,'independent_metric_checks':0,'raw_selection_checks':0,'request_hash_checks':0,'protocol_sha256':protocol['protocol_sha256']}
    source=HERE.parents[1]/'scaling/skillret-seed5150-scaling/scaling5150/provenance-audit/public-inputs'
    docs={s['id']:render(s) for s in rows(source/'skills-metadata.jsonl')};qmap={q['id']:q['query'] for q in panel+read(HERE/'generated-panel.json')}
    ids=read(HERE/'catalog-document-ids.json');catalogs=json.loads(gzip.decompress((HERE/'catalogs.json.gz').read_bytes()));catalog={(c['query_id'],c['seed'],c['size']):[ids[i] for i in c['document_indices']] for c in catalogs}
    for path in sorted((HERE/'results').glob('*-cases.jsonl')):
        attempts=rows(path);canonical=canonical_attempts(attempts);seen=set()
        records=list(canonical.values());validation['historical_unattempted_rows']=validation.get('historical_unattempted_rows',0)+len(attempts)-len(records)
        for r in records:
            assert r['case_id'] not in seen;seen.add(r['case_id']);assert r['protocol_sha256']==protocol['protocol_sha256']
            expected=independent_score(r,labels[r['query_id']])
            for key,v in expected.items():assert abs(r['metrics'][key]-v)<1e-12,(r['case_id'],key)
            validation['independent_metric_checks']+=1
            if r.get('selector'):
                pool=catalog[(r['query_id'],r['seed'],r['size'])] if r['method']=='listing' else r['candidate_ids']
                assert set(pool)<=set(catalog[(r['query_id'],r['seed'],r['size'])])
                prompt,mapping=select_prompt(qmap[r['query_id']],pool,docs,r['query_id'],r['seed'])
                assert digest(prompt)==r['selector']['prompt_sha256'];validation['request_hash_checks']+=1
                if r['status']=='ok':
                    parsed=json.loads(r['selector']['content']);assert isinstance(parsed,list) and len(parsed)<=5 and len(set(parsed))==len(parsed) and set(parsed)<=mapping.keys()
                    assert [mapping[a] for a in parsed]==r['selected_ids'];validation['raw_selection_checks']+=1
            allrows.append(r)
    assert len({r['case_id'] for r in allrows})==len(allrows)
    summaries=[];strata=[];seed_results=[];errors=[]
    for model in ['qwen3_06_q8','qwen35_4b_q4']:
        group=[r for r in allrows if r['model']==model]
        dominant=collections.Counter(r['source_sha256'] for r in group if r['status']=='ok').most_common(1)
        for r in group:r['_latency_eligible']=bool(dominant) and r['source_sha256']==dominant[0][0]
        scopes={'primary200_seed42':[r for r in group if r['panel']=='official_exposed' and r['seed']==42] if model=='qwen3_06_q8' else [],'balanced24_seed42':[r for r in group if r['query_id'] in sensitivity and r['seed']==42],'balanced24_additional_seeds':[r for r in group if r['query_id'] in sensitivity and r['seed'] in [43,44]],'generated12_silver':[r for r in group if r['panel']=='generated_composition']}
        for scope,items in scopes.items():
            for n in SIZES:
                for method in METHODS:
                    sub=[r for r in items if r['size']==n and r['method']==method]
                    if not sub:continue
                    result=summarize(sub,scope,model,n,method);summaries.append(result)
                    base={(r['query_id'],r['seed']):r for r in items if r['size']==n and r['method']=='bm25' and r['status']=='ok' and r['_latency_eligible']}
                    pairs=[(r,base[(r['query_id'],r['seed'])]) for r in sub if r['status']=='ok' and r['_latency_eligible'] and not r.get('routing_reused',False) and (r['query_id'],r['seed']) in base]
                    result['latency_paired_query_cases']=len(pairs)
                    result['relative_fresh_valid_e2e_to_bm25']=mean([a['end_to_end_seconds'] for a,b in pairs])/mean([b['end_to_end_seconds'] for a,b in pairs]) if pairs else None
                    if scope=='primary200_seed42':
                        result['official_prevalence_weighted_exact']=sum((count/4392)*mean([r['metrics']['exact'] for r in sub if len(labels[r['query_id']])==k]) for k,count in [(1,2143),(2,1703),(3,546)])
                    for count in [1,2,3]:
                        part=[r for r in sub if len(labels[r['query_id']])==count]
                        if part:strata.append(summarize(part,scope+'_gold'+str(count),model,n,method))
        for seed in [42,43,44]:
            for n in SIZES:
                for method in METHODS:
                    part=[r for r in group if r['query_id'] in sensitivity and r['seed']==seed and r['size']==n and r['method']==method]
                    if part:
                        pending=sum(r['status'].startswith('not_run') for r in part)
                        attempted=[r for r in part if not r['status'].startswith('not_run')]
                        seed_results.append({'model':model,'seed':seed,'size':n,'method':method,'queries':len(part),'attempted':len(attempted),'valid':sum(r['status']=='ok' for r in part),'not_run':pending,'status':'incomplete' if pending else 'attempted_complete_with_exclusions','exact':None if pending else mean([r['metrics']['exact'] for r in part]),'f1':None if pending else mean([r['metrics']['f1'] for r in part]),'attempted_only_exact':mean([r['metrics']['exact'] for r in attempted])})
        for r in group:
            errors.append({'case_id':r['case_id'],'model':model,'query_id':r['query_id'],'panel':r['panel'],'size':r['size'],'seed':r['seed'],'method':r['method'],'status':r['status'],'gold_count':len(labels[r['query_id']]),'gold_domain_count':r['gold_domain_count'],'candidate_missing_gold':(r.get('candidate_recall') or 0)<1,'extra_selections':r['metrics']['fp'],'missing_selections':r['metrics']['fn'],'route_gold_retention':r.get('routing_gold_retention')})
    primary=[r for r in allrows if r['model']=='qwen3_06_q8' and r['panel']=='official_exposed' and r['seed']==42]
    primary_complete=len(primary)==4000 and not any(r['status'].startswith('not_run') for r in primary)
    paired=bootstrap(primary,labels) if primary_complete else []
    validation['case_rows']=len(allrows);validation['primary200_complete']=primary_complete;validation['primary200_unattempted']=sum(r['status'].startswith('not_run') for r in primary)
    csvout('selection-summary.csv',summaries);csvout('strata.csv',strata);csvout('seed-sensitivity.csv',seed_results);csvout('error-taxonomy.csv',errors);csvout('paired-primary.csv',paired)
    domain=[];major={s['id']:s['major'] for s in rows(source/'skills-metadata.jsonl')}
    for n in SIZES:
        part=[r for r in primary if r['method']=='domain_vector' and r['size']==n]
        if part:
            domain.append({
                'size':n,'queries':len(part),
                'any_positive_route_rate':mean([float((r.get('routing_gold_retention') or 0)>0) for r in part]),
                'all_positive_route_rate':mean([float(r.get('routing_gold_retention')==1) for r in part]),
                'oracle_single_domain_recall_ceiling':mean([max(collections.Counter(major[g] for g in labels[r['query_id']]).values())/len(labels[r['query_id']]) for r in part]),
                'all_gold_one_domain_feasible_rate':mean([float(len({major[g] for g in labels[r['query_id']]})==1) for r in part]),
                'empty_domain_count':sum(r.get('domain_candidate_count')==0 for r in part),
                'valid_route_count':sum((r.get('routing') or {}).get('status')=='ok' for r in part)})
    csvout('domain-diagnostics.csv',domain)
    retrieval_file=HERE/'results/retrieval.jsonl'
    if not retrieval_file.exists():retrieval_file=HERE/'results/retrieval.jsonl.gz'
    if retrieval_file.exists():
        retrieval=rows(retrieval_file);ret=[]
        for seed in [42,43,44]:
            for n in SIZES:
                for method in ['bm25','vector','hybrid']:
                    part=[r['methods'][method] for r in retrieval if r['seed']==seed and r['size']==n]
                    if part:ret.append({'seed':seed,'size':n,'method':method,'queries':len(part),'macro_recall5':mean([r['recall5'] for r in part]),'micro_positive_pair_recall5':sum(r['retrieved_hits'] for r in part)/sum(r['gold_count'] for r in part),'mrr5':mean([r['mrr5'] for r in part]),'search_seconds_mean':mean([r['search_seconds'] for r in part]),'scope':'search only; batch query embeddings separate'})
        csvout('full-retrieval.csv',ret);validation['retrieval_cases']=len(retrieval)
    write(OUT/'validation.json',validation);write(OUT/'summary.json',{'selection':summaries,'paired':paired,'domain':domain,'validation':validation})
    print(json.dumps(validation))

if __name__=='__main__':main()
