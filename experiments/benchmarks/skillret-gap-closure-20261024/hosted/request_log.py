"""Private physical HTTP event ledger. Public export uses an explicit whitelist."""
import datetime,time,uuid
from ledger import append_jsonl
from envelope import envelope_identity,parse_envelope

def begin(client,suffix,body):
    uid='http-'+uuid.uuid4().hex
    stage='generation' if body is not None else ('authentication' if suffix=='key' else 'reconciliation' if suffix.startswith('generation?') else 'availability')
    if body is not None:stage='canary' if (getattr(client,'active_case','') or '').startswith('nonbenchmark-canary') else getattr(client,'active_kind','generation')
    row={'event':'started','request_id':uid,'run_id':'hosted-followup-20261024','case_id':getattr(client,'active_case',None),'attempt_id':uid,'stage':stage,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'http_method':'POST' if body is not None else 'GET','endpoint_family':suffix.split('?')[0],'generation_request':body is not None,'model_requested':body.get('model') if isinstance(body,dict) else None,'provider_pin':body.get('provider') if isinstance(body,dict) else None}
    row['method']=getattr(client,'active_method',None)
    append_jsonl(client.out/'http-events.jsonl',row);client.last_http_request_id=uid
    return row,time.perf_counter()

def finish(client,start,started,payload=None,error=None):
    row={**start,'event':'completed','latency_seconds':time.perf_counter()-started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'http_status':payload.get('_http_status') if isinstance(payload,dict) else None,'result_status':'transport_error' if error else 'response_received','error_type':type(error).__name__ if error else None}
    if isinstance(payload,dict):
        row.update(response_keys=sorted(payload),response_error=client.safe_error(payload) if payload.get('error') else None)
        row['api_error_code']=payload['error'].get('code') if isinstance(payload.get('error'),dict) else None
        if start['generation_request']:
            parsed=parse_envelope(payload)
            row.update({k:parsed[k] for k in ['model','provider','input_tokens','output_tokens','reasoning_tokens','finish_reason','status','cost_usd']})
            # Native generation ID remains in this private ledger only.
            row['private_response_id']=parsed['response_id']
    append_jsonl(client.out/'http-events.jsonl',row)
