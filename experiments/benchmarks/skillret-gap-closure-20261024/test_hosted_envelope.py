"""Synthetic offline regressions: these fixtures are not provider observations."""
import copy,importlib.util,json,pathlib,sys
import pytest
ROOT=pathlib.Path(__file__).parent;sys.path.insert(0,str(ROOT/'hosted'))
from envelope import decode_http,parse_envelope,canary_valid
from budget import Budget,BudgetStop
from envelope import cooldown_headers
spec=importlib.util.spec_from_file_location('tested_hosted_runner',ROOT/'hosted/runner.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def good():return {'id':'gen-fixture-1','model':runner.MODEL,'provider':'DeepInfra','usage':{'prompt_tokens':10,'completion_tokens':8,'cost':0},'choices':[{'message':{'content':'["s000000000000"]'},'finish_reason':'stop'}]}

@pytest.mark.parametrize('field,value',[('usage',None),('choices',[]),('choices',None)])
def test_optional_fields_keep_id(field,value):
    payload=good();payload[field]=value;record=parse_envelope(payload)
    assert record['response_id']=='gen-fixture-1' and not canary_valid(record,runner.MODEL)

def test_null_message_keeps_id():
    payload=good();payload['choices'][0]['message']=None;record=parse_envelope(payload)
    assert record['response_id']=='gen-fixture-1' and record['status']=='invalid_message'

@pytest.mark.parametrize('payload',[None,[],7,'text'])
def test_nonobject_envelopes_fail_closed(payload):assert not canary_valid(parse_envelope(payload),runner.MODEL)

def test_malformed_json_is_explicit_not_a_valid_canary():
    record=parse_envelope(decode_http(b'{not-json',200));assert record['status']=='malformed_json'

@pytest.mark.parametrize('kind',['top_error','choice_error','refusal','length','error_finish','empty','provider_absent','provider_wrong','model_absent','model_wrong'])
def test_valid_target_text_does_not_override_failure_or_provenance(kind):
    value=good()
    if kind=='top_error':value['error']={'code':502,'message':'synthetic'}
    elif kind=='choice_error':value['choices'][0]['error']={'code':502}
    elif kind=='refusal':value['choices'][0]['message']['refusal']='synthetic refusal'
    elif kind=='length':value['choices'][0]['finish_reason']='length'
    elif kind=='error_finish':value['choices'][0]['finish_reason']='error'
    elif kind=='empty':value['choices'][0]['message']['content']=''
    elif kind=='provider_absent':value.pop('provider')
    elif kind=='provider_wrong':value['provider']='OtherProvider'
    elif kind=='model_absent':value.pop('model')
    elif kind=='model_wrong':value['model']='other/model'
    assert not canary_valid(parse_envelope(value),runner.MODEL)

def test_known_zero_is_a_valid_cost_and_unknown_is_not():
    value=good();assert canary_valid(parse_envelope(value),runner.MODEL)
    value['usage']['cost']=None;assert not canary_valid(parse_envelope(value),runner.MODEL)

def client(tmp_path):
    budget=Budget(tmp_path/'budget.jsonl',2,3,1.5,600);budget.reserve('attempt','.00051456')
    with pytest.raises(BudgetStop):budget.settle('attempt',None)
    return runner.Client('fixture',budget,None,tmp_path,{},read_only=True)

@pytest.mark.parametrize('data,status',[({},'identity_missing'),({'id':'gen-other'},'identity_mismatch')])
def test_reconcile_missing_or_mismatched_id_retains_hold(tmp_path,data,status):
    c=client(tmp_path);c.http=lambda suffix:{'data':data,'_http_status':200}
    with pytest.raises(BudgetStop):c.reconcile('attempt','gen-fixture-1')
    assert str(c.budget.state()[1])=='0.00051456'
    row=json.loads((tmp_path/'reconciliation.jsonl').read_text());assert row['status']==status

@pytest.mark.parametrize('provider,model',[(None,None),('OtherProvider',runner.MODEL),('DeepInfra',runner.MODEL)])
def test_financial_settlement_separate_from_provenance(tmp_path,provider,model):
    c=client(tmp_path);c.http=lambda suffix:{'data':{'id':'gen-fixture-1','model':model,'provider_name':provider,'total_cost':0},'_http_status':200}
    result=c.reconcile('attempt','gen-fixture-1');assert c.budget.state()[1]==0
    assert result['provenance_verified']==(provider=='DeepInfra' and model==runner.MODEL)
    assert not canary_valid({'content':'["s000000000000"]','cost_usd':0},runner.MODEL)

@pytest.mark.parametrize('response',[{'error':{'code':404},'_http_status':404},{'_malformed_json':True,'_http_status':200}])
def test_bad_lookup_does_not_erase_unknown(tmp_path,response):
    c=client(tmp_path);c.http=lambda suffix:response
    with pytest.raises(BudgetStop):c.reconcile('attempt','gen-fixture-1')
    assert str(c.budget.state()[1])=='0.00051456'

def test_identity_saved_before_null_message_parsing(tmp_path):
    class Tokenizer:
        def encode(self,text):return type('Encoding',(),{'ids':[1]})()
    b=Budget(tmp_path/'b.jsonl',2,3,1.5,600);c=runner.Client('fixture',b,Tokenizer(),tmp_path,{})
    value=good();value['choices'][0]['message']=None;c.http=lambda *args:value
    result=c.call('canary-fixture',[{'role':'user','content':'fixture'}],'selection',[],8192)
    assert result['status']=='invalid_message'
    assert json.loads((tmp_path/'envelopes.jsonl').read_text())['response_id']=='gen-fixture-1'

@pytest.mark.parametrize('cost',[True,-1,'NaN','not-a-number'])
def test_invalid_matched_cost_retains_hold_with_history(tmp_path,cost):
    c=client(tmp_path);c.http=lambda suffix:{'data':{'id':'gen-fixture-1','model':runner.MODEL,'provider_name':'DeepInfra','total_cost':cost},'_http_status':200}
    with pytest.raises(BudgetStop):c.reconcile('attempt','gen-fixture-1')
    assert str(c.budget.state()[1])=='0.00051456'
    assert json.loads((tmp_path/'reconciliation.jsonl').read_text())['status']=='invalid_total_cost_hold_retained'

def test_retry_after_guidance_is_parsed_without_dumping_headers():
    assert cooldown_headers({})=={'retry_after_header_present':False}
    assert cooldown_headers({'Retry-After':'60','Account':'private'})=={'retry_after_header_present':True,'retry_after_seconds':60}
    assert 'retry_after_utc' in cooldown_headers({'Retry-After':'Sat, 24 Oct 2026 20:00:00 GMT'})
    assert cooldown_headers({'Retry-After':'untrusted header'})=={'retry_after_header_present':True,'retry_after_unparseable':True}
