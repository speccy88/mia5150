"""Offline contract checks for every model, routing and selection gate."""
import json,pathlib,sys
import pytest
ROOT=pathlib.Path(__file__).parent;sys.path.insert(0,str(ROOT/'hosted'))
from diverse_contract import PROFILES,payload,input_allowance,parse
from run_diverse import accept
from runner import Client
from budget import Budget,BudgetStop
from mistral_contract import PROFILE as MISTRAL

@pytest.mark.parametrize('profile',PROFILES)
def test_model_specific_route_rates_and_parameters(profile):
    body=payload(profile,[{'role':'user','content':'fixture'}],'selection',{})
    assert body['model']==profile['model'] and body['provider']['only']==[profile['endpoint']]
    assert body['provider']['allow_fallbacks'] is False and body['max_tokens']==128
    assert ('seed' in body)==profile['seed'] and ('reasoning' in body)==profile['reasoning_none']
    assert 'tools' not in body and 'plugins' not in body
    assert body['response_format']['json_schema']['schema']['type']=='object'
    assert input_allowance(body)>8192

@pytest.mark.parametrize('profile',PROFILES)
@pytest.mark.parametrize('kind',['selection','routing'])
def test_every_decision_requires_envelope_identity_usage_and_success(profile,kind):
    mapping={'s000000000000':'skill'} if kind=='selection' else ['category']
    text='{"selected_ids":["s000000000000"]}' if kind=='selection' else '{"category":"category"}'
    good={'http_status':200,'status':'returned','finish_reason':'stop','content':text,'model':profile['model'],'provider':profile['provider_name'],'cost_usd':0,'input_tokens':10,'output_tokens':8}
    assert accept(good,profile,kind,mapping)==(['skill'] if kind=='selection' else 'category')
    for field,value in [('status','provider_error'),('finish_reason','error'),('finish_reason','length'),('provider','WrongProvider'),('provider',None),('model','wrong/model'),('input_tokens',None),('cost_usd',None),('refusal_present',True),('http_status',500)]:
        bad={**good,field:value}
        with pytest.raises(ValueError):accept(bad,profile,kind,mapping)

def test_duplicate_unknown_or_extra_fields_fail_evaluator():
    for text in ['{"selected_ids":["s000000000000","s000000000000"]}','{"selected_ids":["s111111111111"]}','{"selected_ids":[],"extra":1}']:
        with pytest.raises(ValueError):parse(text,'selection',{'s000000000000':'skill'})

def test_unknown_cost_blocks_other_models_until_evidence_bound(tmp_path):
    b=Budget(tmp_path/'ledger.jsonl',2,3,1.5,600,{p['model']:p['cap'] for p in PROFILES},legacy_model=PROFILES[2]['model'])
    b.reserve('legacy','.00051456',model=PROFILES[2]['model'])
    with pytest.raises(BudgetStop):b.settle('legacy',None)
    with pytest.raises(BudgetStop):b.reserve('new','.01',model=PROFILES[0]['model'])
    b.bound_unknown('legacy','.14','a'*64);b.reserve('new','.01',model=PROFILES[0]['model']);assert len(b.state()[0])==2

def test_qwen_client_uses_its_model_cost_and_provider(tmp_path):
    p=PROFILES[1];b=Budget(tmp_path/'ledger.jsonl',2,3,1.5,600,{p['model']:p['cap']})
    c=Client('fixture',b,None,tmp_path,{},profile=p)
    def response(suffix,body):
        assert body['model']==p['model'] and body['provider']['only']==['siliconflow/fp8'] and 'seed' not in body
        return {'_http_status':200,'id':'gen-offline-fixture','model':p['model'],'provider':'SiliconFlow','usage':{'prompt_tokens':20,'completion_tokens':5,'cost':0},'choices':[{'message':{'content':'{"selected_ids":[]}'},'finish_reason':'stop'}]}
    c.http=response;body=payload(p,[{'role':'user','content':'fixture'}],'selection',{});allowance=input_allowance(body)
    result=c.call('fixture',[{'role':'user','content':'fixture'}],'selection',{},allowance)
    assert accept(result,p,'selection',{})==[]
    assert b.events()[0]['reserve_usd']==str(__import__('decimal').Decimal(allowance)*__import__('decimal').Decimal(p['input_rate'])+__import__('decimal').Decimal(128)*__import__('decimal').Decimal(p['output_rate']))

def test_unicode_line_separators_do_not_break_jsonl_records(tmp_path):
    from prepare_diverse import rows
    p=tmp_path/'unicode.jsonl';p.write_text(json.dumps({'text':'a\u2028b'},ensure_ascii=False)+'\n')
    assert rows(p)==[{'text':'a\u2028b'}]

def test_mistral_uses_distinct_pinned_bf16_route_and_matching_object_contract():
    body=payload(MISTRAL,[{'role':'user','content':'fixture'}],'selection',{})
    assert body['model']=='mistralai/mistral-small-3.2-24b-instruct'
    assert body['provider']=={'only':['parasail/bf16'],'allow_fallbacks':False,'require_parameters':True}
    assert body['max_tokens']==128 and body['temperature']==0 and 'seed' not in body
    record={'http_status':200,'status':'returned','finish_reason':'stop','content':'{"selected_ids":[]}','model':MISTRAL['model'],'provider':'Parasail','input_tokens':10,'output_tokens':5,'cost_usd':0}
    assert accept(record,MISTRAL,'selection',{})==[]
    with pytest.raises(ValueError):accept({**record,'provider':'SiliconFlow'},MISTRAL,'selection',{})

def test_mistral_reservation_cap_is_independent_inside_global_cap(tmp_path):
    b=Budget(tmp_path/'b.jsonl',2,3,1.5,600,{MISTRAL['model']:'.35'})
    with pytest.raises(BudgetStop):b.reserve('too-large','.35000001',model=MISTRAL['model'])
    b.reserve('fits','.32532660',model=MISTRAL['model']);b.settle('fits','.024')
    assert str(b.state()[1])=='0.024'

def test_actual_frozen_protocols_resolve_metadata_and_persist_result_before_network(tmp_path):
    from run_diverse import resolve_timing_scope
    from ledger import append_jsonl
    parent=json.loads((ROOT/'hosted/diverse-protocol.json').read_text())
    for name in ['diverse-protocol.json','mistral-protocol.json']:
        plan=json.loads((ROOT/'hosted'/name).read_text())
        scope=resolve_timing_scope(plan,parent)
        fixture={'case_id':'offline-persistence-fixture','status':'fixture_only','method_body_wall_seconds':None,'end_to_end_seconds':None,'timing_scope':scope}
        path=tmp_path/(name+'.jsonl');append_jsonl(path,fixture)
        assert json.loads(path.read_text())['timing_scope']==parent['timing_scope']
    mistral=json.loads((ROOT/'hosted/mistral-protocol.json').read_text())
    with pytest.raises(ValueError):resolve_timing_scope({**mistral,'parent_protocol_sha256':'wrong'},parent)
