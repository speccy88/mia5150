"""Run only the three admitted OAuth variants sequentially, never a paid API fallback."""
import argparse
import datetime
import json
from pathlib import Path
import time

import bench
from admission import MODELS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    folder=Path(args.out)
    admissions=json.loads((folder/'admission.json').read_text())
    accepted={r['model'] for r in admissions if r.get('admitted')}
    if not accepted:
        raise SystemExit('No admitted models; no benchmark calls made.')
    for name,expected in bench.SOURCE_HASHES.items():
        if bench.file_hash(bench.ROOT/'data'/name)!=expected:
            raise SystemExit('Source bytes changed; no inference.')
    skills,queries=bench.load_data(bench.ROOT/'data/test.jsonl',bench.ROOT/'data/test_queries.json')
    statuses=[]
    for model in MODELS:
        if model not in accepted:
            continue
        config=json.loads((bench.ROOT/'configs'/f'oauth-{model}.json').read_text())
        if config['backend']!='chatgpt-oauth':
            raise SystemExit('Only ChatGPT OAuth is authorized for this suite.')
        bench.validate_config(config)
        started=time.perf_counter()
        print(f'START {model} OAuth v1 variant; 140 planned trials',flush=True)
        result=bench.run(config,skills,queries,folder/model,args.resume)
        result['wall_time_s']=round(time.perf_counter()-started,6)
        result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        statuses.append(result)
        bench.atomic_write(folder/'suite-status.json',json.dumps(statuses,indent=2))
        print('MODEL_STATUS '+json.dumps(result),flush=True)
        stop=result.get('stop') or {}
        if stop.get('error')=='RATE_LIMITED' or 'usage_' in (stop.get('provider_code') or ''):
            print('Shared quota/usage failure; stopping the complete suite, no bypass.',flush=True)
            break


if __name__=='__main__':
    main()
