import pytest
from ledger import canonical_attempts,finished_case_ids
from ledger import append_jsonl,store_case,recover_cases
import builtins,json

def test_only_unattempted_cases_can_resume_and_history_is_retained():
    old={'case_id':'a','status':'not_run_resource_guard','physical_call':False,'selector':None,'routing':None}
    new={'case_id':'a','status':'ok','physical_call':True}
    records=[old,new]
    assert len(records)==2 and canonical_attempts(records)['a']==new
    assert finished_case_ids(records)=={'a'}

@pytest.mark.parametrize('status',['ok','context_overflow','pipeline_error','invalid_selection','transport_or_guard_error'])
def test_attempted_failure_or_success_cannot_be_retried(status):
    with pytest.raises(ValueError):canonical_attempts([{'case_id':'a','status':status},{'case_id':'a','status':'ok'}])

def test_resource_stopped_unattempted_case_remains_eligible():
    assert finished_case_ids([{'case_id':'a','status':'not_run_resource_guard'}])==set()

def test_false_unattempted_model_call_is_rejected():
    with pytest.raises(ValueError):canonical_attempts([{'case_id':'a','status':'not_run_resource_guard','physical_call':True},{'case_id':'a','status':'ok'}])

def test_transient_append_open_failure_retries_io_only(tmp_path,monkeypatch):
    path=tmp_path/'rows.jsonl';real=builtins.open;attempts=[]
    def once(p,*args,**kwargs):
        if p==path and not attempts:attempts.append(1);raise PermissionError('Sharing conflict')
        return real(p,*args,**kwargs)
    monkeypatch.setattr(builtins,'open',once);append_jsonl(path,{'value':1})
    assert [json.loads(x) for x in path.read_text().splitlines()]==[{'value':1}]

def test_saved_response_spool_recovers_without_model_retry(tmp_path,monkeypatch):
    path=tmp_path/'rows.jsonl';spool=tmp_path/'spool';row={'case_id':'a','status':'ok','utc':'2026-10-24T00:00:00Z','attempt_number':1,'selected_ids':['observed']}
    import ledger
    real=ledger.append_jsonl
    def fail(*args):raise PermissionError('Locked destination')
    monkeypatch.setattr(ledger,'append_jsonl',fail)
    with pytest.raises(PermissionError):store_case(path,row,spool)
    assert json.loads(next(spool.glob('*.json')).read_text())==row
    monkeypatch.setattr(ledger,'append_jsonl',real)
    recovered=recover_cases(path,[],spool)
    assert recovered==[row] and recover_cases(path,recovered,spool)==[row]
