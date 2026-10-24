"""Save transparent progress without relabeling unattempted rows as completions."""
import collections,datetime,json,pathlib
from ledger import canonical_attempts
HERE=pathlib.Path(__file__).resolve().parent
def main():
    result={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'campaign_status':'in_progress_or_resource_paused; see current receipts','protocol_sha256':json.loads((HERE/'protocol.json').read_text())['protocol_sha256'],'models':{},'hosted_paid_calls':0,'human_adjudication':'pending'}
    for model,planned in [('qwen3_06_q8',5080),('qwen35_4b_q4',600)]:
        p=HERE/f'results/{model}-cases.jsonl';attempts=[]
        if p.exists():
            with p.open() as f:
                for line in f:
                    try:attempts.append(json.loads(line))
                    except json.JSONDecodeError:break
        current=canonical_attempts(attempts);counts=collections.Counter(r['status'] for r in current.values())
        result['models'][model]={'planned':planned,'ledger_case_attempt_rows':len(attempts),'unique_cases_in_ledger':len(current),'canonical_statuses':dict(counts),'remaining_unattempted':planned-sum(v for k,v in counts.items() if not k.startswith('not_run')),'physical_selector_attempts':sum(bool(r.get('selector',{}).get('physical_call')) for r in current.values() if r.get('selector')),'physical_route_attempts':sum(bool(r.get('routing',{}).get('physical_call')) and not r.get('routing_reused',False) for r in current.values() if r.get('routing'))}
    (HERE/'EXECUTION_STATUS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
