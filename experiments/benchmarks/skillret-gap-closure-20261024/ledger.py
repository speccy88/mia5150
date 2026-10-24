"""Append-only ledger rules: only an unattempted case may receive a new row."""
import json,os,pathlib,time

def append_jsonl(path,row):
    text=json.dumps(row,ensure_ascii=False)+'\n';last=None
    for attempt in range(60):
        try:handle=open(path,'a',encoding='utf-8')
        except PermissionError as error:last=error;time.sleep(.1);continue
        with handle:handle.write(text);handle.flush();os.fsync(handle.fileno())
        return
    raise last

def store_case(path,row,spool):
    if not row['status'].startswith('not_run'):
        spool=pathlib.Path(spool);spool.mkdir(parents=True,exist_ok=True)
        target=spool/(row['case_id']+'-'+str(row.get('attempt_number',1))+'.json')
        content=json.dumps(row,ensure_ascii=False)
        if target.exists():
            if json.loads(target.read_text())!=row:raise ValueError('Immutable case spool mismatch')
        else:
            with target.open('x',encoding='utf-8') as f:f.write(content);f.flush();os.fsync(f.fileno())
    append_jsonl(path,row)

def recover_cases(path,records,spool):
    known={(r['case_id'],r.get('attempt_number',1)) for r in records};recovered=[]
    for p in pathlib.Path(spool).glob('*.json'):
        row=json.loads(p.read_text());key=(row['case_id'],row.get('attempt_number',1))
        if key not in known:recovered.append(row)
    for row in sorted(recovered,key=lambda r:(r['utc'],r.get('attempt_number',1))):append_jsonl(path,row);records.append(row)
    canonical_attempts(records)
    return records
def canonical_attempts(records):
    latest={}
    for row in records:
        key=row['case_id']
        if key in latest:
            previous=latest[key]
            if not previous['status'].startswith('not_run'):
                raise ValueError('Attempted outcome was retried')
            if previous.get('physical_call') or previous.get('selector') or previous.get('routing'):
                raise ValueError('Unattempted row contains a model attempt')
        latest[key]=row
    return latest

def finished_case_ids(records):
    return {key for key,row in canonical_attempts(records).items() if not row['status'].startswith('not_run')}
