"""Export only whitelisted API/cost observations. Private IDs never leave runtime."""
import argparse,collections,datetime,json,pathlib,re,sys
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from core import digest
MODEL='deepseek/deepseek-v4-flash-0731'
def rows(p):
    if not p.exists():return []
    with p.open() as f:return [json.loads(x) for x in f if x.strip()]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime-root',required=True);args=parser.parse_args();private=pathlib.Path(args.runtime_root)/'hosted-execution';out=HERE/'reports';out.mkdir(exist_ok=True)
    events=rows(private/'http-events.jsonl');current={}
    allowed=['request_id','run_id','case_id','attempt_id','stage','method','utc','completed_utc','http_method','endpoint_family','generation_request','model_requested','provider_pin','latency_seconds','http_status','result_status','error_type','model','provider','input_tokens','output_tokens','reasoning_tokens','finish_reason','status','cost_usd']
    allowed.append('api_error_code')
    for event in events:
        key=event['request_id'];current.setdefault(key,{}).update({k:event.get(k) for k in allowed if k in event})
    old=rows(private/'calls.jsonl');private_ids=[]
    for i,call in enumerate(old):
        if call.get('response_id'):private_ids.append(call['response_id'])
        if call.get('http_request_id') in current:continue
        key='legacy-observed-'+str(i+1)
        current[key]={'request_id':key,'attempt_id':key,'run_id':'hosted-v1-20261024','case_id':call['case_key'],'stage':'canary' if call['case_key'].startswith('nonbenchmark-canary') else call.get('kind'),'utc':call.get('utc'),'http_method':'POST','endpoint_family':'chat/completions','generation_request':True,'model_requested':MODEL,'provider_pin':{'only':['deepinfra/fp8'],'allow_fallbacks':False},'model':call.get('model'),'provider':call.get('provider'),'input_tokens':call.get('input_tokens'),'output_tokens':call.get('output_tokens'),'reasoning_tokens':call.get('reasoning_tokens'),'latency_seconds':call.get('wall_seconds'),'http_status':call.get('http_status'),'status':'invalid_canary_missing_output_or_usage','finish_reason':call.get('finish_reason'),'cost_usd':call.get('cost_usd'),'record_scope':'Legacy source had no complete HTTP envelope; unobserved fields remain null. No private generation ID exported.'}
    budget={}
    for event in rows(private/'budget-ledger.jsonl'):
        if event['kind']=='reserve':budget[event['case_key']]=dict(event)
        else:budget[event['case_key']].update(event)
    for row in current.values():
        if not row.get('generation_request'):continue
        state=budget.get(row.get('case_id'),{})
        row.update(reserve_usd=state.get('reserve_usd'),actual_billed_cost_usd=state.get('actual_usd'),billing_state=state.get('kind','unavailable'),reconciliation_state='settled' if state.get('actual_usd') is not None else 'unknown')
    unknown=sum(x.get('actual_usd') is None for x in budget.values());held=sum(float(x.get('actual_usd') if x.get('actual_usd') is not None else x['reserve_usd']) for x in budget.values());known=sum(float(x['actual_usd']) for x in budget.values() if x.get('actual_usd') is not None)
    lookup=rows(private/'reconciliation.jsonl');last=lookup[-1] if lookup else {}
    status={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorized_global_cap_usd':2,'generation_request_ceiling':600,'generation_post_attempts':sum(bool(x.get('generation_request')) for x in current.values()),'benchmark_selector_cases_completed':0,'benchmark_routing_cases_completed':0,'known_billed_generation_cost_usd':known,'known_cost_records':len(budget)-unknown,'actual_total_generation_cost_usd':None if unknown else known,'unknown_cost_attempts':unknown,'conservative_reserved_or_known_total_usd':held,'latest_read_only_lookup':{'http_status':last.get('http_status'),'api_error_code':(last.get('lookup_error') or {}).get('code'),'identity_status':'identity_missing' if not last.get('data_id_present',False) else 'identity_mismatch' if not last.get('id_matches') else 'matched','root_type':'object' if last else None,'root_provider_keys':[k for k in last.get('response_keys',[]) if not k.startswith('_')],'client_metadata_keys':[k for k in last.get('response_keys',[]) if k.startswith('_')],'data_present':last.get('data_present',False),'data_type':last.get('data_type','absent' if last.get('returned_model') is None else 'object'),'data_id_present':last.get('data_id_present',False),'expected_id_matches_data_id':last.get('id_matches',False),'top_level_id_present':last.get('top_level_id_present',False),'expected_id_matches_top_level_id':last.get('top_level_id_matches',False),'returned_public_model':last.get('returned_model'),'returned_public_provider':last.get('returned_provider'),'reported_total_cost':last.get('reported_total_cost')},'historical_read_only_request_count':None,'historical_read_only_scope':'Read-only calls before transport instrumentation were not fully logged; no invented count or latency. All newly instrumented HTTP calls include stage and method.','credential_policy':'User requested encrypted local retention until explicit deletion instruction; never publish credential/ciphertext/path.','continuation':'blocked: HTTP404 did not provide matched generation total_cost. Failed canary remains failed. No new generation.'}
    aggregates=[]
    groups=collections.defaultdict(list)
    for row in current.values():groups[(row.get('run_id'),row.get('model_requested'),row.get('stage'),row.get('method'))].append(row)
    for (run,model,stage,method),items in groups.items():aggregates.append({'run_id':run,'model_requested':model,'stage':stage,'method':method,'physical_http_attempts':len(items),'generation_post_attempts':sum(bool(x.get('generation_request')) for x in items),'observed_response_cost_sum_usd':sum(float(x['cost_usd']) for x in items if x.get('cost_usd') is not None),'actual_billed_known_sum_usd':sum(float(x['actual_billed_cost_usd']) for x in items if x.get('actual_billed_cost_usd') is not None),'unknown_cost_records':sum(bool(x.get('generation_request')) and x.get('actual_billed_cost_usd') is None for x in items),'reserve_sum_usd':sum(float(x['reserve_usd']) for x in items if x.get('reserve_usd') is not None)})
    payloads={'api-call-ledger.jsonl':''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in current.values()),'status.json':json.dumps(status,indent=2),'api-call-cost-summary.json':json.dumps(aggregates,indent=2)}
    for name,text in payloads.items():
        assert not re.search(r'\bgen-[0-9A-Za-z-]+\b',text)
        assert all(identifier not in text for identifier in private_ids)
        (out/name).write_text(text)
    print(json.dumps({'generation_post_attempts':status['generation_post_attempts'],'actual_total_cost_usd':status['actual_total_generation_cost_usd'],'held_or_known_usd':held,'latest_lookup_http_status':status['latest_read_only_lookup']['http_status'],'public_private_id_check':'passed'}))
if __name__=='__main__':main()
