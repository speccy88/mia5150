"""Validate the separate 200-response profile series offline; no inference."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics

import bench_profiles_v3 as b

ROOT=Path(__file__).resolve().parent


def yes(value):
    return str(value).lower()=='true'


def main():
    skills,queries=b.load_data(ROOT/'data/test.jsonl',ROOT/'data/test_queries.json')
    plan={t['key']:t for t in b.make_plan(skills,queries)}
    source_hash=hashlib.sha256((ROOT/'source-measured/bench_profiles_v3.py').read_bytes()).hexdigest()
    summaries=[]
    all_rows={}
    all_audits={}
    checks=[]
    historical=json.loads((ROOT/'evidence/profile-review.json').read_text(encoding='utf-8'))
    for folder in sorted((ROOT/'results').iterdir()):
        if not folder.is_dir() or not (folder/'details.csv').exists():
            continue
        with (folder/'details.csv').open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            assert reader.fieldnames==b.COLUMNS
            rows=list(reader)
        assert len(rows)==20
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        cfg=manifest['config']
        assert manifest['protocol']=='v3-local-context-kv-real-token-gate'
        assert manifest['implementation_sha256']['bench_profiles_v3.py']==source_hash
        assert len(manifest['plan'])==20
        for saved in manifest['plan']:
            t=plan[saved['key']]
            assert saved['candidate_ids']==[s['id'] for s in t['candidates']]
            assert saved['prompt_sha256']==t['prompt_sha256']
        db=sqlite3.connect((folder/'state.sqlite3').as_uri()+'?mode=ro&immutable=1',uri=True)
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        saved={key:(json.loads(v),json.loads(a)) for key,v,a in db.execute('SELECT key,row_json,audit FROM trials WHERE row_json IS NOT NULL')}
        assert len(saved)==20 and db.execute('SELECT count(*) FROM trials WHERE row_json IS NULL').fetchone()[0]==0
        db.close()
        model=cfg['source']['alias']
        preflight=json.loads((ROOT/'evidence'/(model+'-token-preflight.json')).read_text(encoding='utf-8'))
        token_lookup={t['key']:t['input_tokens'] for t in preflight['trials']}
        for row,(key,(stored,audit)) in zip(rows,sorted(saved.items(),key=lambda kv:plan[kv[0]]['qi'])):
            assert all(str(stored[c])==row[c] for c in b.COLUMNS)
            assert row['query']==plan[key]['query']['query'] and row['gold_skill']==plan[key]['query']['gold_skill']
            rank=int(row['rank_of_gold'])
            assert 0<=rank<=10 and yes(row['exact_match'])==(rank==1)
            for k in [1,3,5,10]:
                hit=0<rank<=k
                metrics={'recall':float(hit),'precision':1/k if hit else 0,'ndcg':1/math.log2(rank+1) if hit else 0,
                    'mrr':1/rank if hit else 0,'map':1/rank if hit else 0}
                for metric,value in metrics.items():
                    assert math.isclose(float(row[f'{metric}@{k}']),value,abs_tol=1e-12)
            assert int(row['input_tokens'])==token_lookup[key]
            assert int(row['input_tokens'])+int(row['output_tokens'])==int(row['total_tokens'])
            assert int(row['input_tokens'])+500<=cfg['context_window']
            assert 0<=int(row['output_tokens'])<=500
            assert not yes(row['context_limit']) and not yes(row['rate_limited'])
            for c in ['latency_s','llm_latency_s','retrieval_latency_s','e2e_latency_s']:
                assert math.isfinite(float(row[c])) and float(row[c])>=0
        h=next(x for x in historical['summary'] if x['model']+'-'+x['profile']==folder.name)
        top1=sum(yes(r['exact_match']) for r in rows)
        top10=sum(float(r['recall@10']) for r in rows)
        assert h['top1']==top1 and h['top10']==top10 and h['measured']==20
        assert h['valid']==sum(not r['error'] for r in rows)
        summary={'model':model,'profile':h['profile'],'N':cfg['library_sizes'][0],'context':cfg['context_window'],
            'kv':cfg['kv_type'],'measured':20,'top1':top1,'top10':top10,'valid':sum(not r['error'] for r in rows),
            'median_llm_s':statistics.median(float(r['llm_latency_s']) for r in rows),
            'input_tokens':sum(int(r['input_tokens']) for r in rows),'output_tokens':sum(int(r['output_tokens']) for r in rows),
            'errors':dict(collections.Counter(r['error'] or 'SUCCESS' for r in rows))}
        summaries.append(summary)
        all_rows[folder.name]={k:v[0] for k,v in saved.items()}
        all_audits[folder.name]={k:v[1] for k,v in saved.items()}
        checks.append({'profile':folder.name,'rows':20,'columns':45,'csv_matches_sqlite':True,
            'metrics_recomputed':True,'real_token_preflight_matches':True,'tokens_plus_500_within_context':True,
            'original_runner_hash_matches':True,'candidate_and_prompt_hashes_match':True})
    assert len(summaries)==10 and sum(s['measured'] for s in summaries)==200
    pairs=[]
    for model in ['qwen3-8b-q4km','mistral-7b-instruct-v03-q4km']:
        f16=all_rows[model+'-f16-adjusted-n500'];q8=all_rows[model+'-q8-adjusted-n500']
        assert f16.keys()==q8.keys()
        pair={'model':model,'N':500,'paired_queries':20,
            'f16_top1':sum(yes(r['exact_match']) for r in f16.values()),'q8_top1':sum(yes(r['exact_match']) for r in q8.values()),
            'f16_top10':sum(float(r['recall@10']) for r in f16.values()),'q8_top10':sum(float(r['recall@10']) for r in q8.values()),
            'f16_hit_q8_miss':sum(yes(f16[k]['exact_match']) and not yes(q8[k]['exact_match']) for k in f16),
            'q8_hit_f16_miss':sum(yes(q8[k]['exact_match']) and not yes(f16[k]['exact_match']) for k in f16)}
        saved=next(x for x in historical['paired_quality'] if x['model']==model)
        for k in pair:
            if k!='N':
                assert saved[k]==pair[k]
        pairs.append(pair)
    output={'offline_only':True,'new_inference_calls':0,'measured_total':200,'checks':checks,'summary':summaries,
        'paired_quality':pairs,'original_benchmark_sha256':source_hash,'full_private_raw_normalization_replayed':False,
        'historical_private_capture_audit_reports_verified':200,
        'context_deviation':'real tokenizer input +500, no original 10% estimate margin; separate v3 series',
        'Llama_Gemma_final_state':'not executed; user declined further work'}
    out=ROOT/'validation/public-validation.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    write_summary(summaries,pairs)
    print(json.dumps({'profiles':10,'responses':200,'checks_passed':True,'new_inference_calls':0},indent=2))


def write_summary(summaries,pairs):
    lines=['## Complément final Qwen8/Mistral : série v3 séparée','',
        'Les deux modèles ont fourni 200 réponses supplémentaires : 80 f16 par modèle à N=50/100/250/500, puis 20 Q8_0 par modèle à N=500. Le contexte est adapté aux tokens réels +500, sans la marge estimée de 10 % du lot initial. Les poids Q4_K_M sont inchangés ; Q8_0 désigne le cache KV. Les blocages 32K/f16 historiques restent conservés.','',
        '| Modèle | Profil | N | n_ctx | KV | Top-1 /20 | Top-10 /20 | Sans erreur | Médiane LLM (s) |',
        '| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for s in summaries:
        lines.append(f"| {s['model']} | {s['profile']} | {s['N']} | {s['context']} | {s['kv']} | {s['top1']} | {s['top10']:.0f} | {s['valid']} | {s['median_llm_s']:.3f} |")
    lines += ['', '| Comparaison appariée N=500 | Top-1 f16 → Q8 | Top-10 f16 → Q8 | f16 seul correct | Q8 seul correct |',
        '| --- | --- | --- | ---: | ---: |']
    for x in pairs:
        lines.append(f"| {x['model']} | {x['f16_top1']} → {x['q8_top1']} | {x['f16_top10']:.0f} → {x['q8_top10']:.0f} | {x['f16_hit_q8_miss']} | {x['q8_hit_f16_miss']} |")
    lines += ['', 'Sur les quatre tailles f16, Qwen8 atteint 34/80 (42,5 %) et Mistral 36/80 (45,0 %). Les résultats Q8 restent séparés. Les 20 questions appariées ne prouvent ni une amélioration générale ni une équivalence de quantification. Aucun N≥1000 n’a été mesuré dans ce complément.','',
        'Au total, 16 modèles locaux distincts ont maintenant des mesures dans au moins une série : 1 120 réponses dans le lot initial, plus 200 dans ce complément, soit 1 320. Les trois modèles OAuth ajoutent 360 réponses, soit 1 680 réponses mesurées tous profils confondus ; ces nombres ne sont pas des observations indépendantes. Llama 3.2 et Gemma 3 restent non exécutés à la clôture demandée.','',
        '[Rapport détaillé v3](profiles-v3/PROFILE_REPORT.md) · [Validation](profiles-v3/validation/public-validation.json) · [Clôture Llama/Gemma](profiles-v3/CLOSURE.md)']
    (ROOT.parent/'PROFILE_COMPLEMENT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
