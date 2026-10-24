"""Export derived diverse-study evidence without private generation/account IDs."""
import argparse,collections,csv,datetime,decimal,json,pathlib,re,sys
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from core import selection_metrics,digest
from diverse_contract import PROFILES,canaries,payload

def rows(p):
    if not p.exists():return []
    with p.open() as f:return [json.loads(x) for x in f if x.strip()]

def main(runtime):
    work=runtime/'hosted-diverse-v2';out=HERE/'reports';out.mkdir(exist_ok=True);private=runtime/'hosted-execution'
    cases=rows(work/'cases.jsonl');calls=rows(private/'calls.jsonl');http=rows(private/'http-events.jsonl');lookup={c['case_key']:c for c in calls};public=[]
    for case in cases:
        item=dict(case)
        for stage,key in [('selector','selector_attempt_key'),('route','route_attempt_key')]:
            answer=lookup.get(case.get(key),{})
            item[stage+'_response']={k:answer.get(k) for k in ['http_status','status','content','model','provider','finish_reason','input_tokens','output_tokens','reasoning_tokens','cost_usd','request_sha256']}
        assert item['metrics']==selection_metrics(item['selected_ids'],item['gold_ids'],item['status'])
        public.append(item)
    planned=[]
    protocol=json.loads((HERE/'diverse-protocol.json').read_text())
    for p in PROFILES:
        model_cases=[r for r in cases if r['model']==p['model']]
        model_calls=[c for c in calls if c['case_key'].startswith('nonbenchmark-canary-v2-'+p['slug']) or c['case_key'] in {r.get('selector_attempt_key') for r in model_cases}|{r.get('route_attempt_key') for r in model_cases}]
        for method in protocol['methods']:
            subset=[r for r in model_cases if r['method']==method];valid=[r for r in subset if r['status']=='ok'];n=len(subset)
            planned.append({'model':p['model'],'method':method,'size':100,'planned':24,'attempted_logical_cases':n,'valid':len(valid),'unattempted':24-n,'status':'complete' if n==24 else 'not_started' if n==0 else 'partial','exact_count':sum(r['metrics']['exact'] for r in subset) if n else None,'decision_exact_rate_attempted':sum(r['metrics']['exact'] for r in subset)/n if n else None,'macro_f1_attempted':sum(r['metrics']['f1'] for r in subset)/n if n else None,'balanced_panel_only':True,'model_canary_attempts':sum(c['case_key'].startswith('nonbenchmark-canary-v2') for c in model_calls)})
    with (out/'diverse-summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(planned[0]));w.writeheader();w.writerows(planned)
    with (out/'diverse-cases.jsonl').open('w') as f:
        for item in public:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    canary=[]
    for c in calls:
        if not c['case_key'].startswith('nonbenchmark-canary-v2'):continue
        canary.append({k:c.get(k) for k in ['case_key','utc','kind','http_status','status','request_sha256','model','provider','input_tokens','output_tokens','cost_usd','finish_reason']})
    latest=canary[-1] if canary else {}
    receipt={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':protocol['protocol_sha256'],'benchmark_cases_planned':360,'benchmark_cases_completed':len(cases),'benchmark_cases_unattempted':360-len(cases),'v2_canary_attempts':len(canary),'canary_results':canary,'latest_http_status':latest.get('http_status'),'status':'halted_by_unknown_cost_guard' if latest.get('cost_usd') is None and latest else 'inspect_execution_receipt','reason':'Gemma DeepInfra returned HTTP429 upstream model rate limit; no generation ID, choices or usage. No automatic retry; different-model continuation paused by new unknown-cost policy.' if latest.get('http_status')==429 else None,'known_charged_cost_for_failed_canary_usd':latest.get('cost_usd'),'failed_case_replayed':False,'credential_retention':'Encrypted local credential retained at user request; no value/path/account metadata published.','scope':'No never-run cell is scored as zero. No hosted benchmark performance is claimed when completed count is zero.'}
    text=json.dumps(receipt,indent=2);assert not re.search(r'\bgen-[A-Za-z0-9-]+',text);(out/'diverse-execution.json').write_text(text)
    if latest.get('http_status')==429 and latest.get('case_key')=='nonbenchmark-canary-v2-gemma-1':
        p=PROFILES[0];c=canaries()[0];body=payload(p,c['messages'],c['kind'],c['values']);assert digest(body)==latest['request_sha256']
        snapshots=json.loads((HERE/'diverse-endpoint-snapshots.json').read_text());e=next(s['endpoint'] for s in snapshots if s['endpoint']['model_id']==p['model'])
        assert e['tag']=='deepinfra/fp8' and body['provider']['allow_fallbacks'] is False and 'tools' not in body and 'plugins' not in body
        bound=decimal.Decimal(e['context_length'])*decimal.Decimal(e['pricing']['prompt'])+decimal.Decimal(e['max_completion_tokens'])*decimal.Decimal(e['pricing']['completion'])
        candidate={'status':'candidate_only_not_applied_pending_review','case_id':latest['case_key'],'request_sha256':latest['request_sha256'],'request_reconstruction_matches':True,'http_status':429,'actual_cost_usd':None,'request_pin':body['provider'],'sent_max_tokens':128,'endpoint_context_tokens':e['context_length'],'endpoint_max_completion_tokens':e['max_completion_tokens'],'pricing':e['pricing'],'full_endpoint_bound_usd':str(bound),'candidate_unknown_hold_usd':'.02','reason':'No generation ID or usage supplied; cannot perform bound generation lookup. Preserve failed case and unknown actual cost. A second conservative hold requires explicit policy review before other-model continuation.'}
        (out/'gemma-unknown-bound-candidate.json').write_text(json.dumps(candidate,indent=2))
    print(json.dumps({'benchmark_completed':len(cases),'canary_attempts':len(canary),'latest_http':latest.get('http_status'),'public_private_id_check':'passed'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',required=True);main(pathlib.Path(p.parse_args().runtime_root))
