"""Terminal receipt and conservative STOP hold; never performs network requests."""
import argparse,csv,datetime,decimal,hashlib,json,pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from core import digest
from budget import Budget
from diverse_contract import PROFILES,payload,canaries
from mistral_contract import PROFILE as P
D=decimal.Decimal
def read(p):return json.loads(p.read_text())
def rows(p):
    if not p.exists():return []
    with p.open() as f:return [json.loads(x) for x in f if x.strip()]
def main(runtime):
    private=runtime/'hosted-execution';out=HERE/'mistral-reports';out.mkdir(exist_ok=True)
    calls=rows(private/'calls.jsonl');selected=[c for c in calls if c['case_key'].startswith('nonbenchmark-canary-v3-mistral-')];assert len(selected)==1
    failed=selected[0];fixture=canaries()[0];body=payload(P,fixture['messages'],fixture['kind'],fixture['values']);assert digest(body)==failed['request_sha256']
    assert failed['http_status']==429 and failed['status']=='provider_error' and failed.get('response_id') is None and failed.get('cost_usd') is None
    error=failed['provider_error'];assert error['provider_name']=='Parasail' and error['limit_source']=='upstream_provider_shared_pool'
    endpoint=read(HERE/'mistral-endpoint-snapshot.json')['endpoint'];assert endpoint['tag']=='parasail/bf16' and endpoint['quantization']=='bf16'
    assert D(endpoint['pricing']['prompt'])==D(P['input_rate']) and D(endpoint['pricing']['completion'])==D(P['output_rate'])
    bound=D(endpoint['context_length'])*D(P['input_rate'])+D(endpoint['max_completion_tokens'])*D(P['output_rate']);assert bound==D('.02162688')
    events=rows(private/'http-events.jsonl');http=next(r for r in reversed(events) if r.get('case_id')==failed['case_key'] and r['event']=='completed');assert http['cooldown']['retry_after_header_present'] is False
    proof={'status':'increased_conservative_stop_hold_no_continuation','case_id':failed['case_key'],'request_sha256':failed['request_sha256'],'request_reconstruction_matches':True,'http_status':429,'provider_name':'Parasail','limit_source':'upstream_provider_shared_pool','retry_after_header_present':False,'actual_cost_usd':None,'context_tokens':endpoint['context_length'],'endpoint_max_completion_tokens':endpoint['max_completion_tokens'],'pricing':endpoint['pricing'],'full_endpoint_bound_usd':str(bound),'conservative_stop_hold_usd':'.03','meaning':'Hold increased to cover entire advertised input/completion ceilings. Actual cost remains unknown. The STOP stays active; no retry, fallback or different model is authorized.'}
    (out/'mistral-failure-bound.json').write_text(json.dumps(proof,indent=2))
    budget=Budget(private/'budget-ledger.jsonl',2,3,1.5,600,{p['model']:p['cap'] for p in [*PROFILES,P]},legacy_model='deepseek/deepseek-v4-flash-0731');state,_=budget.state()
    if D(state[failed['case_key']]['reserve_usd'])<D('.03'):budget.raise_unknown_hold(failed['case_key'],'.03',hashlib.sha256((out/'mistral-failure-bound.json').read_bytes()).hexdigest())
    state,held=budget.state();known=sum((D(s['actual_usd']) for s in state.values() if s.get('actual_usd') is not None),D(0));holds=sum((D(s['reserve_usd']) for s in state.values() if s.get('actual_usd') is None),D(0))
    qwen=rows(runtime/'hosted-diverse-v2/cases.jsonl');mistral=rows(runtime/'hosted-mistral-v3/cases.jsonl');assert len(qwen)==120 and not mistral and len(state)==150
    receipt={'status':'terminal_no_more_generation','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_frozen_selector_plan':360,'original_plan_completed_qwen_cases':120,'original_plan_unattempted_deepseek_gemma_cases':240,'later_exploratory_added_mistral_plan':120,'later_mistral_completed_cases':0,'later_mistral_unattempted_cases':120,'combined_planned_selector_cases':480,'combined_completed_selector_cases':120,'combined_unattempted_selector_cases':360,'mistral_canaries_planned':3,'mistral_canaries_attempted':1,'mistral_selectors_attempted':0,'mistral_routes_attempted':0,'mistral_failure':proof,'generation_posts_all_attempts':len(state),'successful_qwen_generation_posts':147,'failed_generation_posts':3,'known_billed_sum_usd':str(known),'unknown_hold_usd':str(holds),'unknown_attempts':3,'guarded_total_usd':str(held),'actual_total_cost_usd':None,'global_cap_usd':'2.00','global_generation_cap':600,'recorded_read_only_http_gets':sum(r['event']=='completed' and r['http_method']=='GET' for r in events),'earlier_read_only_total':'unknown, not fully instrumented','credential_policy':'Retain encrypted local credential until explicit user instruction; no secret/account/path published.','interpretation':'Only Qwen has successful benchmark evidence. Other provider failures do not constitute a completed multi-model quality comparison. No further models or retries. The Mistral plan was frozen after Qwen outcomes and differs in family/provider/BF16 vs FP8; it remains unexecuted beyond the failed canary.'}
    assert held==D('.21411946') and known==D('.02411946') and holds==D('.19')
    (out/'terminal-execution.json').write_text(json.dumps(receipt,indent=2));(private/'terminal-no-more-generation.json').write_text(json.dumps({'status':'terminal_no_more_generation','reason':'Single additional authorized arm returned upstream429; stop, no cycling','generation_posts':150},indent=2))
    with (out/'unattempted-mistral-cells.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','method','size','planned','attempted','valid','unattempted','exact_rate','f1','status']);w.writeheader()
        for method in read(HERE/'mistral-protocol.json')['methods']:w.writerow({'model':P['model'],'method':method,'size':100,'planned':24,'attempted':0,'valid':0,'unattempted':24,'exact_rate':None,'f1':None,'status':'unattempted_after_upstream429_canary'})
    print(json.dumps({'status':receipt['status'],'generation_posts':len(state),'qwen_cases':120,'mistral_cases':0,'known_billed_usd':str(known),'unknown_holds_usd':str(holds),'guarded_total_usd':str(held),'recorded_read_only_gets':receipt['recorded_read_only_http_gets']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',required=True);main(pathlib.Path(p.parse_args().runtime_root))
