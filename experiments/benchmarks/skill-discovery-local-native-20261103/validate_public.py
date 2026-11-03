"""Offline CSV/SQLite/prompt validation and descriptive cross-route comparison."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics

import bench_native_v2 as bench

ROOT=Path(__file__).resolve().parent
OAUTH=ROOT.parent/'skill-discovery-oauth-v1-20261103'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def truth(v):
    return str(v).lower()=='true'


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==bench.COLUMNS
        return list(reader)


def main():
    skills,queries=bench.load_data(ROOT/'data/test.jsonl',ROOT/'data/test_queries.json')
    assert len(skills)==6006 and len(queries)==20
    plan=bench.make_plan(skills,queries)
    expected={t['key']:t for t in plan}
    checks=[]
    groups=[]
    coverage=[]
    canonical_totals=collections.Counter()
    for suite in ['local-native-v2','local-v1','diagnostic-vram-oversubscription']:
        for folder in sorted((ROOT/'results'/suite).iterdir()):
            if not folder.is_dir() or not (folder/'details.csv').exists():
                continue
            rows=read_csv(folder/'details.csv')
            assert len({(r['library_size'],r['query']) for r in rows})==len(rows)
            db=sqlite3.connect((folder/'state.sqlite3').as_uri()+'?mode=ro&immutable=1',uri=True)
            assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            stored={key:json.loads(v) for key,v in db.execute('SELECT key,row_json FROM trials WHERE row_json IS NOT NULL')}
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            assert len(manifest['plan'])==140
            for p in manifest['plan']:
                t=expected[p['key']]
                assert p['candidate_ids']==[s['id'] for s in t['candidates']]
                assert p['prompt_sha256']==t['prompt_sha256']
            for row in rows:
                matching=[k for k,t in expected.items() if t['n']==int(row['library_size']) and t['query']['query']==row['query']]
                assert len(matching)==1
                key=matching[0]
                assert all(str(stored[key][c])==row[c] for c in bench.COLUMNS),(folder.name,key,'persistence')
                rank=int(row['rank_of_gold'])
                assert 0<=rank<=10 and truth(row['exact_match'])==(rank==1)
                for k in [1,3,5,10]:
                    hit=0<rank<=k
                    values={'recall':float(hit),'precision':1/k if hit else 0,
                            'ndcg':1/math.log2(rank+1) if hit else 0,'mrr':1/rank if hit else 0,'map':1/rank if hit else 0}
                    for metric,value in values.items():
                        assert math.isclose(float(row[f'{metric}@{k}']),value,abs_tol=1e-12)
                for c in ['latency_s','llm_latency_s','retrieval_latency_s','e2e_latency_s']:
                    assert math.isfinite(float(row[c])) and float(row[c])>=0
                if row['input_tokens']:
                    assert int(row['total_tokens'])==int(row['input_tokens'])+int(row['output_tokens'])
                    assert int(row['output_tokens'])<=500
                    assert int(row['input_tokens'])+500<=manifest['config']['context_window']
                if truth(row['context_limit']):
                    assert row['error'].startswith('CONTEXT_LIMIT')
                assert row['gold_skill']==expected[key]['query']['gold_skill']
            intents=db.execute('SELECT count(*),sum(row_json IS NULL) FROM trials').fetchone()
            db.close()
            checks.append({'suite':suite,'model':folder.name,'rows':len(rows),'schema_columns':45,
                           'csv_matches_exported_sqlite':True,'metrics_recomputed':True,
                           'all_140_candidate_and_prompt_hashes_match':True,'trial_intents':intents[0],
                           'unresolved_intents':intents[1] or 0})
            if suite!='local-native-v2':
                continue
            assert len(rows)==140
            canonical_totals.update(r['error'] or 'SUCCESS' for r in rows)
            measured=[r for r in rows if r['input_tokens'] and r['output_tokens']]
            groups.append({'route':'local','model':folder.name,'rows':rows,'measured':measured})
    for folder in sorted(OAUTH.joinpath('results').iterdir()):
        if folder.is_dir() and (folder/'details.csv').exists():
            rows=read_csv(folder/'details.csv')
            groups.append({'route':'OAuth','model':folder.name,'rows':rows,'measured':[r for r in rows if r['input_tokens'] and r['output_tokens']]})
    common_sizes=[50,100,250,500]
    summary=[]
    import numpy as np
    indices=np.random.default_rng(20261103).integers(0,20,size=(10000,20))
    for group in groups:
        measured=group['measured']
        common=[r for r in measured if int(r['library_size']) in common_sizes]
        ci=None
        if common:
            assert len(common)==80
            lookup={(int(r['library_size']),r['query']):r for r in common}
            matrix=np.array([[float(truth(lookup[(n,q['query'])]['exact_match'])) for n in common_sizes] for q in queries])
            ci=[float(x) for x in np.quantile(matrix[indices].mean(axis=(1,2)),[.025,.975])]
        summary.append({'route':group['route'],'model':group['model'],'measured_common_sizes':len(common),
            'correct_top1':sum(truth(r['exact_match']) for r in common),
            'top1_measured':sum(truth(r['exact_match']) for r in common)/len(common) if common else None,
            'query_cluster_percentile95':ci,'unique_question_clusters':20 if common else 0,
            'valid_nonerror_common':sum(not r['error'] for r in common),
            'median_llm_s_common':statistics.median(float(r['llm_latency_s']) for r in common) if common else None,
            'input_tokens_common':sum(int(r['input_tokens']) for r in common) if common else None,
            'output_tokens_common':sum(int(r['output_tokens']) for r in common) if common else None})
        for n in bench.SIZES:
            rows=[r for r in group['rows'] if int(r['library_size'])==n]
            actual=[r for r in measured if int(r['library_size'])==n]
            errors=collections.Counter(r['error'] or 'SUCCESS' for r in rows)
            coverage.append({'route':group['route'],'model':group['model'],'N':n,'planned':20,
                'measured_with_usage':len(actual),'valid_nonerror':sum(not r['error'] for r in actual),
                'top1_measured':sum(truth(r['exact_match']) for r in actual)/len(actual) if actual else None,
                'errors':dict(errors),'median_llm_s':statistics.median(float(r['llm_latency_s']) for r in actual) if actual else None})
    assert len([g for g in groups if g['route']=='local' and g['measured']])==14
    assert len([g for g in groups if g['route']=='local' and not g['measured']])==4
    assert sum(len(g['measured']) for g in groups if g['route']=='local')==1120
    assert canonical_totals=={'SUCCESS':542,'REASONING_NO_FINAL_ANSWER':532,'INVALID_OUTPUT':42,
        'MALFORMED_REASONING_BOUNDARY':4,'ACCESS_GATED_ACCEPTANCE_REQUIRED':160,
        'HARDWARE_VRAM_CAPACITY_EXCEEDED':160,'CONTEXT_LIMIT_ESTIMATED':1080}
    result={'offline_only':True,'new_inference_calls':0,'checks':checks,'canonical_error_counts':dict(canonical_totals),
            'common_sizes':common_sizes,'common_summary':summary,'coverage':coverage,
            'uncertainty':'10000 paired question-cluster resamples, seed 20261103; descriptive, conditional on biased fixed lists; no iid n=80 claim.',
            'full_raw_response_normalization_replayed':False}
    save(ROOT/'validation/public-validation.json',result)
    with (ROOT/'validation/common-panel-summary.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    plot(coverage)
    report(summary)
    print(json.dumps({'public_sqlite_checked':len(checks),'local_models_measured':14,'local_models_blocked':4,
                      'local_measured_rows':1120,'local_csv_rows':2520,'oauth_measured_rows':360,
                      'all_140_logical_prompts_match':True,'new_inference_calls':0},indent=2))


def plot(coverage):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    profile_file=ROOT/'profiles-v3/validation/public-validation.json'
    coverage=list(coverage)
    if profile_file.exists():
        profiles=json.loads(profile_file.read_text(encoding='utf-8'))['summary']
        for model in sorted({s['model'] for s in profiles}):
            for kv in ['f16','q8_0']:
                series=[s for s in profiles if s['model']==model and s['kv']==kv]
                for n in bench.SIZES:
                    s=next((x for x in series if x['N']==n),None)
                    coverage.append({'route':'v3 '+kv,'model':model,'N':n,'planned':20 if s else 0,
                        'measured_with_usage':20 if s else 0,'top1_measured':s['top1']/20 if s else None,'errors':{}})
    route_order={'OAuth':0,'local':1,'v3 f16':2,'v3 q8_0':3}
    order=sorted({(r['route'],r['model']) for r in coverage},key=lambda x:(route_order[x[0]],x[1]))
    grid=np.full((len(order),len(bench.SIZES)),np.nan)
    fig,(left,right)=plt.subplots(1,2,figsize=(16,13.0),gridspec_kw={'width_ratios':[1.15,1]},layout='constrained')
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#edf0f2')
    for i,(route,model) in enumerate(order):
        for j,n in enumerate(bench.SIZES):
            r=next(x for x in coverage if x['route']==route and x['model']==model and x['N']==n)
            if r['top1_measured'] is not None:
                grid[i,j]=r['top1_measured']
    image=left.imshow(np.ma.masked_invalid(grid),vmin=0,vmax=1,cmap=cmap,aspect='auto')
    right.imshow(np.zeros_like(grid),vmin=0,vmax=1,cmap=plt.get_cmap('Greys'),aspect='auto',alpha=.08)
    for i,(route,model) in enumerate(order):
        for j,n in enumerate(bench.SIZES):
            r=next(x for x in coverage if x['route']==route and x['model']==model and x['N']==n)
            val=r['top1_measured']
            left.text(j,i,f'{val:.0%}' if val is not None else '—',ha='center',va='center',fontsize=9,
                      color='white' if val is not None and val<.55 else '#172331')
            errors=r['errors']
            if r['measured_with_usage']:
                text=f"{r['measured_with_usage']}/20"
            elif r['planned']==0:
                text='NP'
            elif 'CONTEXT_LIMIT_ESTIMATED' in errors:
                text='0/20 C'
            elif 'NOT_RUN_AFTER_MODEL_STOP' in errors:
                text='0/20 H'
            else:
                text='0/20 B'
            right.text(j,i,text,ha='center',va='center',fontsize=8.5)
    labels=[({'OAuth':'OAuth','local':'Local 32K','v3 f16':'V3 adjusted f16','v3 q8_0':'V3 Q8 KV'}[route]+' · ')+model for route,model in order]
    left.set_yticks(range(len(order)),labels=labels,fontsize=9)
    right.set_yticks(range(len(order)),labels=['']*len(order))
    for ax in (left,right):
        ax.set_xticks(range(len(bench.SIZES)),labels=[str(n) for n in bench.SIZES])
        ax.set_xlabel('Number of skills (IDs)')
        ax.set_xticks(np.arange(-.5,len(bench.SIZES),1),minor=True)
        ax.set_yticks(np.arange(-.5,len(order),1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1)
        ax.tick_params(which='minor',bottom=False,left=False)
        ax.axhline(2.5,color='#405061',linewidth=2)
        if profile_file.exists():
            ax.axhline(20.5,color='#405061',linewidth=2)
    left.set_title('Exact top-1 / actually measured responses',fontsize=12)
    right.set_title('Coverage: measured responses / 20 planned',fontsize=12)
    bar=fig.colorbar(image,ax=left,location='bottom',shrink=.75,pad=.045)
    bar.set_ticks([0,.25,.5,.75,1],labels=['0%','25%','50%','75%','100%'])
    fig.suptitle('Skill discovery, final results · distinct OAuth, local 32K and v3 context/KV series\nThe same 20 questions repeat; golds always occupy positions 1–20',fontsize=14)
    fig.supxlabel('C: estimated exclusion · B: gated access or VRAM · H: 1 HTTP rejection + 19 unattempted · NP: not planned in this series\nNanbeige has a measured zero. Gray cells without scores are not measured failures. Context/output controls differ.',fontsize=10)
    (ROOT/'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT/'figures/coverage-comparison.png',dpi=165)
    plt.close(fig)


def report(summary):
    lines=['# Descriptive comparison: identical libraries, distinct inference conditions','',
        'The public copies preserve measured outcomes. This comparison does not establish API/OAuth/local equivalence: local models are quantized, use a 32K window and 500-token output cap, and have a native final-answer adapter; OAuth omits temperature/output controls and reports medium reasoning. No new model calls were made for the comparison.','',
        '![Measured quality and coverage](figures/coverage-comparison.png)','',
        'The plot uses the same 20 questions at each size. Every colored zero comes from actual measured outputs; missing scores indicate exclusions, blocks or unattempted sizes. Format errors and reasoning without final answers stay in the denominator of measured local outputs.','',
        '## Shared panel: N = 50, 100, 250, 500','',
        'These scores use 80 measurements per executed model: 20 questions repeated at four sizes. Intervals resample 20 question clusters with all four sizes together, 10,000 times, seed 20261103. They are descriptive and conditional on the biased lists, rather than treating 80 cases as independent. Blocked models receive no interval.','',
        '| Route | Model | Measurements | Top-1 exact | Question-cluster bootstrap, 95% | Outputs without errors | Median LLM (s) |',
        '| --- | --- | ---: | ---: | --- | ---: | ---: |']
    for s in summary:
        n=s['measured_common_sizes'];ci=s['query_cluster_percentile95']
        rate=f"{s['correct_top1']}/{n} ({s['top1_measured']:.1%})" if n else 'Not measured'
        interval=f'{ci[0]:.1%} to {ci[1]:.1%}' if ci else '—'
        latency=f"{s['median_llm_s_common']:.3f}" if n else '—'
        lines.append(f"| {s['route']} | {s['model']} | {n} | {rate} | {interval} | {s['valid_nonerror_common']} | {latency} |")
    lines += ['', '## Limitations and evidence','',
        '- **Position bias:** all golds occupy positions 1–20, using the same seed as question selection. Differences cannot be generalized to random or central positions.',
        '- **Initial local coverage:** 14 models executed, 1,120 outputs with actual usage; four models without inference. The 1,080 estimated-context rows do not measure native maximum model windows.',
        '- **OAuth coverage:** 360 measured responses at N≤2000. At N=5000 each model has one HTTP string-length rejection and 19 unattempted cases, rather than an established token-context limit.',
        '- **Nanbeige:** all 80 measured outputs lack a final answer within the 500-token budget. Its zero does not establish a general semantic inability.',
        '- **Local hardware:** transformer/output layers and KV are on CUDA, but llama.cpp b11960 retains CPU_Mapped input/embedding weights and CPU preparation. This profile does not demonstrate zero CPU participation. See [hardware scope](evidence/NATIVE_GPU_SCOPE.md).',
        '- **Latency:** local and OAuth HTTP timings include different work and output lengths. No hardware-speed ratio is claimed.',
        '- **Native-v2 definition:** this corrects reasoning parsing; the biased scientific v1 protocol is unchanged. The scientific v2 proposed in the OAuth report was not executed.',
        '- **External comparability:** agreement with the reference cloud runner remains unverified; duplicate names and lexical query hints remain.',
        '', '[Local report](LOCAL_REPORT.md) · [OAuth report](../skill-discovery-oauth-v1-20261103/REPORT.md) · [Offline validation](validation/public-validation.json) · [Shared-panel summary](validation/common-panel-summary.csv)', '',
        'CSV bytes are unchanged. Distributed databases are nonresumable audit exports. Full private HTTP outputs are not redistributed, so public validation recomputes ranking metrics and persistence without replaying every normalization from raw text. Transformation records and hashes describe privacy adaptations.']
    if (ROOT/'PROFILE_COMPLEMENT.md').exists():
        lines += ['',(ROOT/'PROFILE_COMPLEMENT.md').read_text(encoding='utf-8')]
    (ROOT/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
