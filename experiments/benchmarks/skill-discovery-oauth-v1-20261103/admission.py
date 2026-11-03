"""One small OAuth admission/format probe per explicitly selected model; no retries."""
import argparse
import datetime
import json
from pathlib import Path
import time

import bench

MODELS = ['gpt-6-luna', 'gpt-6.1-sol', 'gpt-6.1-terra']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder/'admission.json'
    if status_path.exists():
        raise SystemExit('Admission evidence exists; no automatic replay of probes.')
    records = []
    candidates = [{'id':'synthetic-add','name':'arith-add','description':'Add two numbers.'},
                  {'id':'synthetic-weather','name':'weather-get','description':'Get the weather forecast.'}]
    query = {'query':'Add 2 and 3.', 'gold_skill':'arith-add', 'gold_id':'synthetic-add'}
    trial = {'n':2, 'query':query, 'candidates':candidates, 'messages':[
        {'role':'system','content':bench.SYSTEM+'1. arith-add: Add two numbers.\n2. weather-get: Get the weather forecast.'},
        {'role':'user','content':query['query']}]}
    for model in MODELS:
        config = json.loads((bench.ROOT/'configs'/f'oauth-{model}.json').read_text())
        bench.validate_config(config)
        record = {'model':model, 'state':'inflight', 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        records.append(record)
        bench.atomic_write(status_path, json.dumps(records, indent=2))
        started = time.perf_counter()
        try:
            result = bench.infer(config, trial)
            row = bench.empty_row(config, trial)
            ranks = bench.score(row, result['text'], candidates)
            usage = result['usage']
            token_ok = all(isinstance(usage.get(k),int) and usage[k]>=0 for k in ('input_tokens','output_tokens'))
            good = bool(row['exact_match'] and not row['error'] and not row['num_invalid_skills'] and token_ok)
            record.update(state='completed', admitted=good, latency_s=round(time.perf_counter()-started,6),
                          input_tokens=usage.get('input_tokens'),output_tokens=usage.get('output_tokens'),
                          selected_skill=row['selected_skill'], exact_match=row['exact_match'],
                          error=row['error'] or ('' if good else 'ADMISSION_FORMAT_OR_USAGE_FAILED'),
                          response_sha256=bench.digest(result['text'].encode()),
                          response_audit=result.get('response_audit',{}))
        except bench.BenchError as e:
            record.update(state='failed',admitted=False,latency_s=round(time.perf_counter()-started,6),
                          error=e.code,http_status=e.status,provider_code=e.provider_code,
                          request_id=e.request_id,retry_after_s=e.retry_after_s)
        bench.atomic_write(status_path, json.dumps(records, indent=2))
        print(json.dumps(record, ensure_ascii=False),flush=True)
        if record.get('error')=='RATE_LIMITED' or 'usage_' in (record.get('provider_code') or ''):
            print('Shared usage/quota error: stopping all further probes; no bypass.',flush=True)
            break


if __name__=='__main__':
    main()
