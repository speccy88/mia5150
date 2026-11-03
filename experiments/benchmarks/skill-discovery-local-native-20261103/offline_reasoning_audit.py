"""Audit captured native outputs offline. Never invokes a model or rewrites source results."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import bench as old
import bench_native_v2 as new

ROOT=Path(__file__).resolve().parent

def main():
    skills,qs=old.load_data(ROOT/'data/test.jsonl',ROOT/'data/test_queries.json')
    trials={t['key']:t for t in old.make_plan(skills,qs)}
    output={'method':'Offline scoring of existing v2 raw captures; no model calls. v1 raw text was not saved. Hash matches identify captures byte-identical to v1 outputs, but these captures originated in separately recorded v2 inferences.', 'models':{}}
    for alias in ['qwen3-0.6b-q8','qwen3-1.7b-q8','qwen3.5-4b-q4km','ling-3.0-tiny-q4km']:
        v1=ROOT/'results/local-v1'/alias/'state.sqlite3'
        old_hashes={}
        if v1.exists():
            db=sqlite3.connect('file:'+str(v1)+'?mode=ro',uri=True)
            for key,audit in db.execute('SELECT key,audit FROM trials WHERE audit IS NOT NULL'):
                old_hashes[key]=json.loads(audit).get('response_sha256')
            db.close()
        records=[]
        for path in sorted((ROOT/'private-responses/local-native-v2'/alias).glob('*.json')):
            raw=json.loads(path.read_text(encoding='utf-8'))
            key=':'.join(path.stem.split('-')[0].split('_'))
            t=trials[key]
            old_row=old.empty_row({'backend':'llama','model':alias},t)
            old.score(old_row,raw['text'],t['candidates'])
            final,error=new.final_answer(raw['text'],raw.get('reasoning_prefix',''))
            new_row=old.empty_row({'backend':'llama','model':alias},t)
            if error:
                new_row['error']=error
            else:
                old.score(new_row,final,t['candidates'])
            records.append({'key':key,'same_raw_sha256_as_v1':old_hashes.get(key)==hashlib.sha256(raw['text'].encode()).hexdigest(),
                'prefilled_reasoning':bool(raw.get('reasoning_prefix')),
                'old_gold_rank':old_row['rank_of_gold'],'corrected_gold_rank':new_row['rank_of_gold'],
                'old_error':old_row['error'],'corrected_error':new_row['error'],
                'output_tokens':raw['usage'].get('output_tokens')})
        summary={'captured_raw_outputs':len(records),'same_raw_hash_as_v1':sum(r['same_raw_sha256_as_v1'] for r in records),
                 'old_top1_hits_on_these_captures':sum(r['old_gold_rank']==1 for r in records),
                 'corrected_top1_hits':sum(r['corrected_gold_rank']==1 for r in records),
                 'old_top10_hits_on_these_captures':sum(r['old_gold_rank']>0 for r in records),
                 'corrected_top10_hits':sum(r['corrected_gold_rank']>0 for r in records),
                 'reasoning_false_top10_hits_removed':sum(r['old_gold_rank']>0 and r['corrected_gold_rank']==0 for r in records),
                 'corrected_errors':dict(collections.Counter(r['corrected_error'] or 'OK' for r in records)),
                 'records':records}
        output['models'][alias]=summary
    (ROOT/'evidence/offline-reasoning-audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='records'} for k,v in output['models'].items()},indent=2))

if __name__=='__main__':
    main()
