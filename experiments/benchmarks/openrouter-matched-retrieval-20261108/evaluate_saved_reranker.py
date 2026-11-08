#!/usr/bin/env python3
"""Validate and score user-supplied saved reranker outputs; never send a request.

Input JSONL rows must identify one prepared request by source_method, size,
query_index; copy all three provenance digests and query; supply model_id,
schema_version, status, ordered_ids, api_latency_seconds, reported_cost_usd.
Provenance digests must describe the actual request, not an after-the-fact guess.
"""
import argparse,csv,json,math
from pathlib import Path
from run_offline import single_target_metrics, write_csv, write_json

FIELDS=['query','catalog_sha256','candidate_order_sha256','candidate_text_sha256']

def validate_response(record,plan,model,schema):
    for field in FIELDS:
        if record.get(field)!=plan[field]:raise ValueError('Request provenance mismatch: '+field)
    if record.get('model_id')!=model or record.get('schema_version')!=schema:
        raise ValueError('Model/schema mismatch')
    if record.get('status') not in {'ok','failed','unknown'}:
        raise ValueError('Unrecognized execution status')
    for key in ['api_latency_seconds','reported_cost_usd']:
        value=record.get(key)
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0):
            raise ValueError('Invalid reported accounting: '+key)
    ranking=record.get('ordered_ids')
    if not isinstance(ranking,list) or any(not isinstance(x,str) for x in ranking):
        raise ValueError('ordered_ids must be strings')
    if len(ranking)!=len(set(ranking)) or len(ranking)>len(plan['candidate_ids']) or not set(ranking)<=set(plan['candidate_ids']):
        raise ValueError('Invalid, duplicate or out-of-pool ranking')
    if record['status']!='ok' and ranking:
        raise ValueError('Failed or unknown result cannot contribute quality')
    return ranking


def evaluate(records,plans,raw,model,schema):
    pmap={(x['source_method'],x['size'],x['query_index']):x for x in plans}
    gold={(x['method'],x['size'],x['query_index']):x for x in raw}
    seen=set();quality=[];attempts=[]
    for r in records:
        key=(r['source_method'],r['size'],r['query_index'])
        if key not in pmap or key in seen:raise ValueError('Unknown or duplicate request key')
        seen.add(key);ranking=validate_response(r,pmap[key],model,schema);target=gold[key]
        attempts.append({'source_method':key[0],'size':key[1],'query_index':key[2],
                         'status':r['status'],'api_latency_seconds':r.get('api_latency_seconds'),
                         'reported_cost_usd':r.get('reported_cost_usd')})
        for contract,targets in [('uuid_single_target',[target['gold_id']]),
                                 ('name_equivalent_single_target',target['name_equivalent_ids'])]:
            for k in [5,10]:
                quality.append({'source_method':key[0],'size':key[1],'query_index':key[2],
                                'model_id':model,'schema_version':schema,'status':r['status'],
                                'label_contract':contract,'k':k,
                                **({name:None for name in ['recall','hit','precision','f1','mrr','map','ndcg','completeness']}
                                   if r['status']=='unknown' else single_target_metrics(ranking if r['status']=='ok' else [],targets,k))})
    return quality,attempts,{'planned':len(plans),'observed_attempt_records':len(seen),
        'unattempted_or_missing':len(plans)-len(seen),
        'ok':sum(r['status']=='ok' for r in attempts),'failed':sum(r['status']=='failed' for r in attempts),
        'unknown':sum(r['status']=='unknown' for r in attempts),
        'known_reported_cost_usd':sum(r['reported_cost_usd'] for r in attempts if r['reported_cost_usd'] is not None),
        'unknown_cost_records':sum(r['reported_cost_usd'] is None for r in attempts),
        'note':'Missing/unattempted cases are not fabricated zero predictions. Unknown outcomes have null metrics. Known failed attempts score zero only for explicit operational-denominator analysis; successful-only, attempted, and planned denominators must remain separate.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--saved-results',required=True,type=Path)
    p.add_argument('--model-id',required=True);p.add_argument('--schema-version',required=True)
    p.add_argument('--study-results',type=Path,default=Path(__file__).parent/'results')
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    plans=json.loads((a.study_results/'jev-rerank-plan.json').read_text())['requests']
    raw=[json.loads(s) for s in (a.study_results/'per_case.jsonl').open()]
    records=[json.loads(s) for s in a.saved_results.open()]
    quality,attempts,coverage=evaluate(records,plans,raw,a.model_id,a.schema_version)
    a.output.mkdir(parents=True,exist_ok=True)
    if quality:write_csv(a.output/'quality.csv',quality)
    if attempts:write_csv(a.output/'attempts.csv',attempts)
    write_json(a.output/'coverage.json',coverage)
    print(json.dumps(coverage))

if __name__=='__main__':main()
