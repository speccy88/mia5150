"""Independent schema, persistence and rank-metric checks; aggregate measured rows only."""
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics

ROOT=Path(__file__).resolve().parent
SIZES=[50,100,250,500,1000,2000,5000]

def istrue(x):
    return str(x).lower()=='true'

def review(suite_name='local-v1'):
    checks=[]
    summary=[]
    reference_plan=None
    for folder in sorted((ROOT/'results'/suite_name).iterdir()):
        if not folder.is_dir() or not (folder/'details.csv').exists():
            continue
        with (folder/'details.csv').open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            columns=reader.fieldnames
            rows=list(reader)
        status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {'complete':False}
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        plan=[(p['key'],p['prompt_sha256'],p['candidate_ids']) for p in manifest['plan']]
        if reference_plan is None:
            reference_plan=plan
        assert plan==reference_plan, 'Candidate/prompt plan differs: '+folder.name
        assert len(columns)==45 and len(set(columns))==45, 'CSV schema differs'
        assert len({(r['library_size'],r['query']) for r in rows})==len(rows), 'Duplicate trial keys'
        if status['complete']:
            assert len(rows)==140
            assert collections.Counter(int(r['library_size']) for r in rows)=={n:20 for n in SIZES}
        for r in rows:
            assert len(r)==45 and None not in r, 'Malformed CSV row'
            rank=int(r['rank_of_gold'])
            assert 0<=rank<=10
            assert istrue(r['exact_match'])==(rank==1)
            for k in [1,3,5,10]:
                hit=0<rank<=k
                expected={'recall':float(hit),'precision':1/k if hit else 0,
                          'ndcg':1/math.log2(rank+1) if hit else 0,
                          'mrr':1/rank if hit else 0,'map':1/rank if hit else 0}
                for name,value in expected.items():
                    assert math.isclose(float(r[f'{name}@{k}']),value,abs_tol=1e-12)
            if r['input_tokens']:
                assert int(r['total_tokens'])==int(r['input_tokens'])+int(r['output_tokens'])
                assert int(r['output_tokens'])<=500
                assert int(r['input_tokens'])+500<=manifest['config']['context_window']
            assert int(r['num_invalid_skills'])==len(json.loads(r['invalid_skills']))
        db=sqlite3.connect('file:'+str(folder/'state.sqlite3')+'?mode=ro',uri=True)
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        db_rows={key:json.loads(row) for key,row in db.execute('SELECT key,row_json FROM trials WHERE row_json IS NOT NULL')}
        assert len(db_rows)==len(rows) or not status['complete']
        # Compare persisted values independently of the CSV exporter.
        by_key={(str(v['library_size']),v['query']):v for v in db_rows.values()}
        for r in rows:
            if (r['library_size'],r['query']) in by_key:
                saved=by_key[(r['library_size'],r['query'])]
                assert all(str(saved[k])==r[k] for k in columns)
        db.close()
        checks.append({'model':folder.name,'rows':len(rows),'complete':status['complete'],
                       'schema_columns':len(columns),'sqlite_integrity':'ok',
                       'rank_metrics_recomputed':True,'csv_matches_sqlite':True,'candidate_plan_matches':True})
        for n in SIZES:
            group=[r for r in rows if int(r['library_size'])==n]
            measured=[r for r in group if r['input_tokens'] and r['output_tokens']]
            valid=[r for r in measured if not r['error']]
            hits=sum(istrue(r['exact_match']) for r in measured)
            summary.append({'model':folder.name,'library_size':n,'recorded_rows':len(group),
                'measured_inference_rows':len(measured),'valid_nonerror_rows':len(valid),
                'strict_top1_hits':hits,'strict_top10_hits':sum(float(r['recall@10']) for r in measured),
                'top1_rate_measured':hits/len(measured) if measured else None,
                'top1_precision_valid':sum(istrue(r['exact_match']) for r in valid)/len(valid) if valid else None,
                'top10_rate_measured':sum(float(r['recall@10']) for r in measured)/len(measured) if measured else None,
                'errors':dict(collections.Counter(r['error'] or 'OK' for r in group)),
                'outputs_at_500':sum(int(r['output_tokens'])==500 for r in measured),
                'input_token_min':min((int(r['input_tokens']) for r in measured),default=None),
                'input_token_max':max((int(r['input_tokens']) for r in measured),default=None),
                'median_llm_s':statistics.median(float(r['llm_latency_s']) for r in measured) if measured else None})
    out={'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,
         'summary':summary,'denominator_note':'Measured rows have real input/output token usage. OOM, gated access, context skips and pending trials have null measured accuracy, never an observed zero.'}
    evidence_name='result-review.json' if suite_name=='local-v1' else 'result-review-native-v2.json'
    (ROOT/'evidence'/evidence_name).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'models_reviewed':len(checks),'complete_models':sum(x['complete'] for x in checks),'checks_passed':True}),flush=True)
    return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--suite',default='local-v1')
    review(p.parse_args().suite)
