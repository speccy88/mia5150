#!/usr/bin/env python3
"""Offline terminal receipt audit; no project imports, network or live tests."""
import argparse,collections,decimal,hashlib,json,pathlib
D=decimal.Decimal
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def js(p):return json.loads(p.read_text(encoding='utf8'))
def jl(p):
    with p.open(encoding='utf8') as f:return [json.loads(s) for s in f if s.strip()]
def main(a):
    e=pathlib.Path(a.evidence);h=e/'hosted';old=pathlib.Path(a.qwen_evidence)/'hosted';manifest=js(e/'EVIDENCE_MANIFEST.json')
    assert manifest['source_commit']=='d76981132a4510804f3858e3bb4366e99144054c'
    for n,v in manifest['files_sha256'].items():assert sha(e/n)==v,n
    unchanged=['reports/diverse-cases.jsonl','reports/hosted-selection.csv','reports/hosted-paired-bootstrap.csv','diverse-protocol.json','diverse-request-plan.csv']
    for n in unchanged:assert (h/n).read_bytes()==(old/n).read_bytes(),n
    ledger=jl(h/'reports/api-call-ledger.jsonl');gen=[r for r in ledger if r['generation_request']];reads=[r for r in ledger if not r['generation_request']]
    assert len(ledger)==len({r['request_id'] for r in ledger})==159 and len(gen)==len({r['case_id'] for r in gen})==150 and len(reads)==9
    assert all(r['http_method']=='POST' for r in gen) and all(r['http_method']=='GET' for r in reads)
    assert collections.Counter(r['stage'] for r in reads)=={'authentication':6,'availability':3}
    old_gen={r['case_id']:r for r in jl(old/'reports/api-call-ledger.jsonl') if r['generation_request']}
    new_gen={r['case_id']:r for r in gen};assert set(new_gen)-set(old_gen)=={'nonbenchmark-canary-v3-mistral-1'}
    for k,row in old_gen.items():assert row==new_gen[k],k
    failed=new_gen['nonbenchmark-canary-v3-mistral-1'];assert failed['http_status']==429 and failed['status']=='provider_error' and failed['billing_state']=='unknown'
    assert failed['model_requested']=='mistralai/mistral-small-3.2-24b-instruct' and failed['provider_pin']=={'only':['parasail/bf16'],'allow_fallbacks':False,'require_parameters':True}
    assert failed['cooldown']['retry_after_header_present'] is False
    for k in ['model','provider','input_tokens','output_tokens','reasoning_tokens','finish_reason','cost_usd','actual_billed_cost_usd']:assert failed[k] is None
    proof=js(h/'mistral-reports/mistral-failure-bound.json');endpoint=js(h/'mistral-endpoint-snapshot.json')['endpoint'];protocol=js(h/'mistral-protocol.json');temp=dict(protocol);expected_protocol=temp.pop('protocol_sha256');assert digest(temp)==expected_protocol
    assert endpoint['tag']=='parasail/bf16' and endpoint['quantization']=='bf16' and endpoint['provider_name']=='Parasail'
    schema={'type':'json_schema','json_schema':{'name':'skillret_v2_selection','strict':True,'schema':{'type':'object','properties':{'selected_ids':{'type':'array','items':{'type':'string','pattern':'^s[0-9a-f]{12}$'},'maxItems':5}},'required':['selected_ids'],'additionalProperties':False}}}
    body={'model':'mistralai/mistral-small-3.2-24b-instruct','provider':{'only':['parasail/bf16'],'allow_fallbacks':False,'require_parameters':True},'messages':[{'role':'user','content':'Non-evaluation format check. Return exactly {"selected_ids":["s000000000000"]}.'}],'response_format':schema,'temperature':0,'max_tokens':128,'stream':False}
    assert digest(body)==proof['request_sha256']=='878814214909f29ab17d6035cfe6110b5a953317c689eff736bd692bc14db6ff'
    bound=D(endpoint['context_length'])*D(endpoint['pricing']['prompt'])+D(endpoint['max_completion_tokens'])*D(endpoint['pricing']['completion'])
    assert bound==D('.02162688')==D(proof['full_endpoint_bound_usd']) and bound<D('.03')==D(proof['conservative_stop_hold_usd'])
    assert proof['provider_name']=='Parasail' and proof['limit_source']=='upstream_provider_shared_pool' and proof['actual_cost_usd'] is None
    # Reconstruct all reserve/settle/unknown mutations rather than summing history.
    state={};peak=D(0);events=jl(h/'reports/budget-reservation-history.jsonl')
    for event in events:
        key=event['attempt_id']
        if event['kind']=='reserve':assert key not in state;state[key]=dict(event)
        else:assert key in state;state[key].update(event)
        total=sum((D(s['actual_usd']) if s.get('actual_usd') is not None else D(s['reserve_usd']) for s in state.values()),D(0));assert total<=D('2');peak=max(peak,total)
    assert len(state)==150 and len(events)==303
    for key,s in state.items():
        row=new_gen[key];assert s.get('actual_usd')==row['actual_billed_cost_usd'] and D(s['reserve_usd'])==D(row['reserve_usd']) and s['kind']==row['billing_state']
    assert state['nonbenchmark-canary-v3-mistral-1']['evidence_sha256']==sha(h/'mistral-reports/mistral-failure-bound.json')
    known=[s for s in state.values() if s.get('actual_usd') is not None];unknown=[s for s in state.values() if s.get('actual_usd') is None]
    assert len(known)==147 and len(unknown)==3
    cost=sum((D(s['actual_usd']) for s in known),D(0));hold=sum((D(s['reserve_usd']) for s in unknown),D(0));assert cost==D('.02411946') and hold==D('.19') and cost+hold==D('.21411946')
    receipt=js(h/'mistral-reports/terminal-execution.json')
    expected={'original_frozen_selector_plan':360,'original_plan_completed_qwen_cases':120,'original_plan_unattempted_deepseek_gemma_cases':240,'later_exploratory_added_mistral_plan':120,'later_mistral_completed_cases':0,'later_mistral_unattempted_cases':120,'combined_planned_selector_cases':480,'combined_completed_selector_cases':120,'combined_unattempted_selector_cases':360,'mistral_canaries_attempted':1,'mistral_selectors_attempted':0,'mistral_routes_attempted':0,'generation_posts_all_attempts':150,'successful_qwen_generation_posts':147,'failed_generation_posts':3,'unknown_attempts':3,'recorded_read_only_http_gets':9}
    for k,v in expected.items():assert receipt[k]==v,(k,receipt[k])
    assert receipt['actual_total_cost_usd'] is None and D(receipt['guarded_total_usd'])==cost+hold
    status=js(h/'reports/status.json');assert status['generation_post_attempts']==150 and status['unknown_cost_attempts']==3 and status['known_cost_records']==147 and status['actual_total_generation_cost_usd'] is None
    previous=js(h/'reports/diverse-execution.json');assert 'known_charged_cost_for_failed_canary_usd' not in previous and previous['failed_canary_actual_costs_known'] is False
    assert D(str(previous['latest_canary_reported_cost_usd']))==D('.00000387')
    # Static defense checks only; runtime stop-file existence is owner evidence.
    runner=(h/'runner.py').read_text();assert runner.count("terminal-no-more-generation.json")>=2
    source=(h/'run_diverse.py').read_text();assert "timing_scope=resolve_timing_scope" in source and "record['timing_scope']=timing_scope" in source
    amended=js(h/'mistral-protocol-metadata-amendment.json');assert amended['frozen_mistral_protocol_sha256']==expected_protocol
    result={'status':'passed','source_commit':manifest['source_commit'],'manifest_files_verified':len(manifest['files_sha256']),'prior_qwen_evidence_byte_identical':unchanged,'new_paid_attempts_since_qwen_snapshot':1,'total_generation_attempts':150,'successful_qwen_generation_attempts':147,'failed_provider_canaries':3,'logged_read_only_http_gets':9,'logged_readonly_breakdown':{'authentication':6,'availability':3},'public_merged_request_rows':159,'earlier_readonly_total':None,'budget_history_events_verified':303,'budget_history_unique_attempts':150,'maximum_reconstructed_guarded_total_usd':str(peak),'known_api_reported_cost_usd':str(cost),'unknown_holds_usd':str(hold),'guarded_known_plus_holds_usd':str(cost+hold),'actual_total_cost_usd':None,'mistral_failed_request_hash_reconstructed':True,'mistral_http_status':429,'mistral_provider_scope_owner_receipt':'Parasail; upstream_provider_shared_pool','mistral_retry_after_header_present':False,'mistral_full_endpoint_bound_usd':str(bound),'mistral_conservative_stop_hold_usd':'.03','mistral_benchmark_cases':0,'mistral_routing_calls':0,'mistral_no_replay_confirmed_in_public_request_ledger':True,'selector_case_accounting':expected,'failed_canary_cost_label_corrected':True,'timing_scope_metadata_repair_present':True,'terminal_generation_guard_present_in_source':True,'report_release_blockers':[],'evidence_sha256':{n:sha(h/n) for n in ['reports/api-call-ledger.jsonl','reports/budget-reservation-history.jsonl','mistral-reports/terminal-execution.json','mistral-reports/mistral-failure-bound.json']},'limits':['Only Qwen has successful benchmark evidence; multi-model quality comparison remains incomplete.','24 exposed balanced queries, not 120 independent queries or fresh held-out data; prior Qwen limitations unchanged.','Reported provider failure scope is preserved owner receipt; no private raw error body or account invoice was accessed.','Actual failed-call costs remain unknown; holds are conservative ceilings, not measured spend.','Runtime terminal stop-file existence and offline-test execution are owner evidence; source guards were inspected, no live calls performed.']}
    pathlib.Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--qwen-evidence',required=True);p.add_argument('--output',required=True);main(p.parse_args())
