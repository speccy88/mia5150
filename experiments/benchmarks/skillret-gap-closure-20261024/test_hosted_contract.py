import importlib.util,pathlib,sys,json
ROOT=pathlib.Path(__file__).parent
sys.path.insert(0,str(ROOT/'hosted'))
spec=importlib.util.spec_from_file_location('hosted_runner',ROOT/'hosted/runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
from core import chat

def test_hosted_payload_pins_provider_and_has_no_tools():
    body=runner.request_body([{'role':'user','content':'fixture'}],'selection',{'s000000000000':'doc'})
    assert body['provider']=={'only':['deepinfra/fp8'],'allow_fallbacks':False,'require_parameters':True}
    assert body['model']=='deepseek/deepseek-v4-flash-0731' and body['max_tokens']==128
    assert 'tools' not in body and 'api_key' not in json.dumps(body)

def test_hosted_message_conversion_preserves_json_data():
    user=json.dumps({'request':'untrusted fixture','skills':[]})
    assert runner.messages(chat('system fixture',user))==[{'role':'system','content':'system fixture'},{'role':'user','content':user}]

def test_hosted_schema_size_does_not_hide_catalog_enum():
    assert runner.schema('selection',range(6006))==runner.schema('selection',range(5))
    assert runner.schema('routing',['A','B'])['json_schema']['schema']['enum']==['A','B']

def test_authenticated_redirect_is_never_followed():
    import urllib.request
    request=urllib.request.Request(runner.API+'key',headers={'Authorization':'Bearer synthetic-nonsecret-fixture'})
    assert runner.NoRedirect().redirect_request(request,None,302,'redirect',{},'https://not-authorized.invalid') is None

def test_reconcile_only_blocks_post_and_generation_before_network_or_reserve():
    import pytest
    client=runner.Client('dummy',None,None,None,{},read_only=True)
    with pytest.raises(runner.BudgetStop):client.http('chat/completions',{'model':'fixture'})
    with pytest.raises(runner.BudgetStop):client.call('fixture',[],'selection',[],8192)
