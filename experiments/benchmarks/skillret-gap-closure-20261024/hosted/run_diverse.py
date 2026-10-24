"""Execute only the frozen diverse v2 panel; one shared budget, no retries."""
import argparse,csv,datetime,decimal,json,pathlib,sys,time
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from core import digest,retrieval_metrics,selection_metrics
from ledger import append_jsonl
from budget import Budget,BudgetStop
from envelope import valid_result
from runner import Client
from diverse_contract import PROFILES,payload,input_allowance,parse,canaries
D=decimal.Decimal

def read(p):return json.loads(p.read_text())
def rows(p):
    if not p.exists():return []
    with p.open() as f:return [json.loads(x) for x in f if x.strip()]

def accept(record,profile,kind,values):
    # Apply this to every canary, router and selector before content parsing.
    if not isinstance(record,dict) or record.get('http_status')!=200 or not valid_result(record,profile['model'],profile['provider_name']):raise ValueError('invalid_response_envelope')
    return parse(record['content'],kind,values)

def verify(client,p):
    response=client.http('models/'+p['model']+'/endpoints');data=response.get('data') or {};endpoint=next(e for e in data.get('endpoints',[]) if e['tag']==p['endpoint'])
    assert endpoint['status']==0 and endpoint['context_length']>=p['context']
    assert D(endpoint['pricing']['prompt'])<=D(p['input_rate']) and D(endpoint['pricing']['completion'])<=D(p['output_rate'])
    assert {'max_tokens','structured_outputs','response_format'}<=set(endpoint['supported_parameters'])
    if p['seed']:assert 'seed' in endpoint['supported_parameters']
    return {'model':p['model'],'provider':p['endpoint'],'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':0,'prices_within_frozen_limits':True}

def execute(runtime,key):
    out=runtime/'hosted-execution';work=runtime/'hosted-diverse-v2';plan=read(HERE/'diverse-protocol.json');private_plan=read(work/'frozen-plan.json');assert plan==private_plan
    check=dict(plan);protocol_sha=check.pop('protocol_sha256');assert digest(check)==protocol_sha
    shared=read(work/'shared-inputs.json');assert digest(shared['entries'])==plan['shared_inputs_sha256'] and plan['profiles']==PROFILES
    with (HERE/'diverse-request-plan.csv').open() as f:request_plan=list(csv.DictReader(f))
    typed=[{k:(int(v) if k in ['input_allowance','output_limit'] else v=='True' if k=='context_admitted' else v) for k,v in r.items()} for r in request_plan]
    assert digest(typed)==plan['request_plan_sha256'],'Request/token plan changed after freeze'
    allowances={(r['model'],r['query_id'],r['method'],r['stage']):r for r in request_plan}
    budget=Budget(out/'budget-ledger.jsonl',2,3,1.5,600,{p['model']:p['cap'] for p in PROFILES},legacy_model='deepseek/deepseek-v4-flash-0731')
    state,_=budget.state();assert state['nonbenchmark-canary-1']['kind']=='bounded_unknown' and D(state['nonbenchmark-canary-1']['reserve_usd'])>=D('.14') and state['nonbenchmark-canary-1']['actual_usd'] is None
    if any(s['kind'] in ['reserve','unknown'] for s in state.values()):raise BudgetStop('New unresolved/in-flight request; no expansion')
    calls={r['case_key']:r for r in rows(out/'calls.jsonl')};cases={r['case_id']:r for r in rows(work/'cases.jsonl')};labels=read(HERE.parent/'evaluator-labels.json')
    def request(client,p,key,msg,kind,values,planned):
        body=payload(p,msg,kind,values);assert input_allowance(body)<=int(planned['input_allowance'])
        if planned['request_sha256']!=digest(body) and planned['method']!='domain_vector':raise BudgetStop('Request differs from frozen hash')
        if key in calls:return calls[key]
        prior,_=budget.state()
        if key in prior:raise BudgetStop('Attempt exists without durable response; never replay')
        record=client.call(key,msg,kind,values,int(planned['input_allowance']));calls[key]=record;return record
    terminal=[]
    for p in PROFILES:
        client=Client(key,budget,None,out,{},profile=p);auth=client.http('key');assert auth.get('_http_status')==200 and isinstance(auth.get('data'),dict)
        append_jsonl(work/'availability.jsonl',verify(client,p));failed=False
        for i,c in enumerate(canaries()):
            cid='nonbenchmark-canary-v2-'+p['slug']+'-'+str(i+1);client.active_method='canary';planned=allowances[(p['model'],'canary-'+str(i+1),'canary',c['kind'])]
            try:answer=request(client,p,cid,c['messages'],c['kind'],c['values'],planned)
            except BudgetStop:
                state,_=budget.state()
                if any(s['kind'] in ['reserve','unknown'] for s in state.values()):raise
                append_jsonl(work/'canary-validation.jsonl',{'case_id':cid,'model':p['model'],'status':'failed_guard_after_known_billing','protocol_sha256':protocol_sha});failed=True;break
            try:assert accept(answer,p,c['kind'],c['values'])==c['expected']
            except (ValueError,AssertionError,TypeError):
                append_jsonl(work/'canary-validation.jsonl',{'case_id':cid,'model':p['model'],'status':'failed','response_status':answer.get('status') if answer else None,'protocol_sha256':protocol_sha});failed=True;break
            append_jsonl(work/'canary-validation.jsonl',{'case_id':cid,'model':p['model'],'status':'passed','protocol_sha256':protocol_sha})
        if failed:
            terminal.append({'model':p['model'],'status':'compatibility_failed_no_benchmark_expansion'});print(p['slug']+' compatibility failed; no benchmark requests for model',flush=True);continue
        for entry in shared['entries']:
            qid=entry['query_id'];method=entry['method'];cid='hv2-'+digest([protocol_sha,p['model'],qid,100,method]);client.active_method=method
            if cid in cases:continue
            record={'case_id':cid,'query_id':qid,'size':100,'seed':42,'method':method,'model':p['model'],'provider_pin':p['endpoint'],'protocol_sha256':protocol_sha,'panel':'balanced24_exposed','gold_ids':labels[qid],'selected_ids':[],'status':'pending'};started=time.perf_counter()
            branch='common'
            if method=='domain_vector':
                planned=allowances[(p['model'],qid,method,'routing')];answer=request(client,p,cid+'-routing',entry['route_messages'],'routing',shared['domains'],planned);record['route_attempt_key']=cid+'-routing'
                try:branch=accept(answer,p,'routing',shared['domains'])
                except (ValueError,TypeError):record['status']='invalid_routing_response'
                if record['status']=='pending':record['domain']=branch;record['routing_gold_retention']=sum(shared['major'][d]==branch for d in labels[qid])/len(labels[qid])
            if record['status']=='pending':
                branch_input=entry['branches'][branch];planned=allowances[(p['model'],qid,method,'selection')]
                if method=='domain_vector':
                    frozen_hash=digest([digest(payload(p,b['messages'],'selection',b['mapping'])) for b in entry['branches'].values()]);assert frozen_hash==planned['request_sha256']
                if planned['context_admitted']!='True':record['status']='context_preflight_excluded'
                else:
                    answer=request(client,p,cid,branch_input['messages'],'selection',branch_input['mapping'],planned);record['selector_attempt_key']=cid;record['candidate_ids']=branch_input['candidate_ids'];record['retrieval']=None if method=='listing' else retrieval_metrics(branch_input['ranking'],labels[qid])
                    try:record['selected_ids']=accept(answer,p,'selection',branch_input['mapping']);record['status']='ok'
                    except (ValueError,TypeError):record['status']='invalid_selection_response';record['returned_cardinality_unknown']=True
            record['metrics']=selection_metrics(record['selected_ids'],labels[qid],record['status']);record['method_body_wall_seconds']=time.perf_counter()-started;record['end_to_end_seconds']=None;record['timing_scope']=plan['timing_scope'];append_jsonl(work/'cases.jsonl',record);cases[cid]=record
            if len(cases)%20==0:print('hosted matched cases durable '+str(len(cases))+'/360',flush=True)
        terminal.append({'model':p['model'],'status':'benchmark_panel_complete'})
    current,held=budget.state();receipt={'status':'run_terminal','protocol_sha256':protocol_sha,'models':terminal,'canonical_cases':len(cases),'physical_generation_attempts_including_legacy':len(current),'guarded_total_usd':str(held),'actual_total_usd':None if any(x.get('actual_usd') is None for x in current.values()) else str(held),'legacy_unknown_cost_preserved':True,'cap_usd':'2.00','call_cap':600};(work/'execution.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',required=True);a=p.parse_args();key=sys.stdin.read().strip()
    if not key or '\n' in key:raise BudgetStop('Missing private stdin credential')
    execute(pathlib.Path(a.runtime_root),key)
if __name__=='__main__':
    try:main()
    except Exception as error:print('HOSTED_DIVERSE_HALTED: '+type(error).__name__+'; inspect private receipts; no automatic retry',file=sys.stderr);raise SystemExit(1)
