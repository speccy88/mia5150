"""Freeze a later Mistral arm from the existing shared inputs; no outcomes read."""
import argparse,csv,datetime,decimal,json,pathlib,sys,urllib.request
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from core import digest
from runner import NoRedirect
from diverse_contract import payload,input_allowance,canaries
from mistral_contract import PROFILE as P
D=decimal.Decimal
def main(runtime):
    work=runtime/'hosted-mistral-v3';work.mkdir(exist_ok=True);assert not (work/'cases.jsonl').exists(),'Arm already has outcomes'
    original=json.loads((HERE/'diverse-protocol.json').read_text());shared=json.loads((runtime/'hosted-diverse-v2/shared-inputs.json').read_text());assert digest(shared['entries'])==original['shared_inputs_sha256']
    url='https://openrouter.ai/api/v1/models/'+P['model']+'/endpoints'
    with urllib.request.build_opener(NoRedirect()).open(url,timeout=45) as response:data=json.load(response)
    endpoint=next(e for e in data['data']['endpoints'] if e['tag']==P['endpoint']);assert endpoint['status']==0 and endpoint['context_length']>=P['context'];assert endpoint['quantization']=='bf16'
    assert D(endpoint['pricing']['prompt'])<=D(P['input_rate']) and D(endpoint['pricing']['completion'])<=D(P['output_rate']);assert {'structured_outputs','response_format','max_tokens','temperature'}<=set(endpoint['supported_parameters'])
    request_plan=[]
    def add(qid,method,kind,bodies):
        a=max(map(input_allowance,bodies));assert a+128<=P['context'],'Context cell excluded; do not silently truncate'
        request_plan.append({'model':P['model'],'query_id':qid,'method':method,'stage':kind,'input_allowance':a,'output_limit':128,'request_sha256':digest(bodies[0]) if len(bodies)==1 else digest([digest(b) for b in bodies]),'max_cost_usd':str(D(a)*D(P['input_rate'])+D(128)*D(P['output_rate']))})
    for i,c in enumerate(canaries()):add('canary-'+str(i+1),'canary',c['kind'],[payload(P,c['messages'],c['kind'],c['values'])])
    for entry in shared['entries']:
        if entry['method']=='domain_vector':add(entry['query_id'],entry['method'],'routing',[payload(P,entry['route_messages'],'routing',shared['domains'])])
        add(entry['query_id'],entry['method'],'selection',[payload(P,b['messages'],'selection',b['mapping']) for b in entry['branches'].values()])
    total=sum((D(r['max_cost_usd']) for r in request_plan),D(0));assert total<=D('.35') and len(request_plan)==147
    protocol={'version':'hosted-mistral-v3-20261024','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_protocol_sha256':original['protocol_sha256'],'query_ids':original['query_ids'],'size':100,'seed':42,'methods':original['methods'],'profile':P,'shared_inputs_sha256':digest(shared['entries']),'request_plan_sha256':digest(request_plan),'planned_benchmark_cases':120,'planned_generation_requests':147,'model_cap_usd':'.35','conservative_planned_cost_usd':str(total),'input_allowance_sum':sum(r['input_allowance'] for r in request_plan),'output_allowance_sum':128*len(request_plan),'prior_generation_attempts':149,'cumulative_planned_generation_attempts':296,'prior_guarded_total_usd':'0.18411946','cumulative_guarded_plan_usd':str(D('.18411946')+total),'global_cap_usd':'2.00','global_generation_cap':600,'interpretation':'Later exploratory arm frozen after Qwen results, before any Mistral outcomes. Same exposed balanced panel, metadata, catalogs, order, retrieval inputs, v2 JSON schema, temperature0,128 output cap. Seed omitted in both arms. Parasail BF16 vs SiliconFlow FP8 and sequential service timing confound causal model comparisons. No native tokenizer usage estimate claim; same conservative engineering allowance. No further model/provider after this arm.'}
    protocol['protocol_sha256']=digest(protocol)
    (HERE/'mistral-protocol.json').write_text(json.dumps(protocol,indent=2));(work/'frozen-plan.json').write_text(json.dumps(protocol,indent=2));(work/'shared-inputs.json').write_text(json.dumps(shared,ensure_ascii=False))
    (HERE/'mistral-endpoint-snapshot.json').write_text(json.dumps({'source':url,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'endpoint':endpoint},indent=2))
    with (HERE/'mistral-request-plan.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(request_plan[0]));w.writeheader();w.writerows(request_plan)
    print(json.dumps({'status':'frozen_before_mistral_outcomes','planned_cases':120,'requests':len(request_plan),'input_allowance_sum':protocol['input_allowance_sum'],'output_allowance_sum':protocol['output_allowance_sum'],'max_arm_cost_usd':str(total),'arm_cap_usd':'.35','cumulative_guarded_plan_usd':protocol['cumulative_guarded_plan_usd'],'protocol_sha256':protocol['protocol_sha256']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',required=True);main(pathlib.Path(p.parse_args().runtime_root))
