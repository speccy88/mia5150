"""Validate the completed OAuth run and compute query-cluster uncertainty; no network."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics

import numpy as np

import bench
from admission import MODELS


def yes(value):
    return str(value).lower() == 'true'


def wilson(successes, n):
    if not n:
        return None
    z=1.959963984540054
    p=successes/n
    mid=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [mid-half,mid+half]


def validate_rows(model, rows, expected, database_rows):
    assert len(rows)==140, (model,'row count',len(rows))
    assert len({(r['library_size'],r['query']) for r in rows})==140
    nonrun=0
    for r in rows:
        assert r['model']==model
        n=int(r['library_size'])
        assert n in bench.SIZES and r['query'] in expected
        assert r['gold_skill']==expected[r['query']]['gold_skill']
        if r['error'].startswith('NOT_RUN'):
            nonrun+=1
            assert all(r[k]=='' for k in ('input_tokens','output_tokens','total_tokens'))
        else:
            stored=database_rows[(n,r['query'])]
            for k in bench.COLUMNS:
                assert str(stored[k])==r[k], (model,n,k,'CSV/SQLite disagreement')
        rank=int(r['rank_of_gold'])
        assert 0<=rank<=10
        assert yes(r['exact_match'])==(rank==1)
        ranking=json.loads(r['ranked_skills'])
        invalid=json.loads(r['invalid_skills'])
        assert len(ranking)<=10
        assert int(r['num_valid_skills'])+int(r['num_invalid_skills'])==len(ranking)
        assert int(r['num_invalid_skills'])==len(invalid)
        if ranking:
            assert r['selected_skill']==ranking[0]
        for k in (1,3,5,10):
            hit=rank>0 and rank<=k
            expected_values={f'recall@{k}':float(hit), f'precision@{k}':1/k if hit else 0,
                             f'ndcg@{k}':1/math.log2(rank+1) if hit else 0,
                             f'mrr@{k}':1/rank if hit else 0,f'map@{k}':1/rank if hit else 0}
            for field,value in expected_values.items():
                assert abs(float(r[field])-value)<1e-12,(model,n,field)
        for k in ('latency_s','llm_latency_s','retrieval_latency_s','e2e_latency_s'):
            assert math.isfinite(float(r[k])) and float(r[k])>=0
        assert float(r['retrieval_latency_s'])==0
        assert r['est_cost_usd']=='', 'OAuth monetary cost was not measured'
        if not r['error']:
            assert all(r[k].isdigit() for k in ('input_tokens','output_tokens','total_tokens'))
            assert int(r['total_tokens'])==int(r['input_tokens'])+int(r['output_tokens'])
        if r['error']=='CONTEXT_LIMIT_ESTIMATED':
            assert yes(r['context_limit']) and float(r['latency_s'])==0
    return {'planned_rows':140,'recorded_rows':140-nonrun,'not_run_rows':nonrun,
            'schema_columns':45,'csv_matches_sqlite':True,'metric_arithmetic_checked':True}


def analyze(folder):
    folder=Path(folder).resolve()
    skills, queries=bench.load_data(bench.ROOT/'data/test.jsonl',bench.ROOT/'data/test_queries.json')
    expected={q['query']:q for q in queries}
    all_rows={}
    audits={}
    validation={}
    summaries=[]
    per_size=[]
    for model in MODELS:
        path=folder/model
        assert (path/'status.json').exists(),f'Model still running: {model}'
        with (path/'details.csv').open(encoding='utf-8',newline='') as f:
            reader=csv.DictReader(f)
            assert reader.fieldnames==bench.COLUMNS
            rows=list(reader)
        all_rows[model]=rows
        db=sqlite3.connect((path/'state.sqlite3').as_uri()+'?mode=ro',uri=True)
        stored={}
        model_audits=[]
        for row_json,audit_json in db.execute('SELECT row_json,audit FROM trials WHERE row_json IS NOT NULL ORDER BY ordinal'):
            row=json.loads(row_json)
            stored[(int(row['library_size']),row['query'])]=row
            model_audits.append(json.loads(audit_json))
        db.close()
        audits[model]=model_audits
        validation[model]=validate_rows(model,rows,expected,stored)
        valid=[r for r in rows if not r['error']]
        notrun=[r for r in rows if r['error'].startswith('NOT_RUN')]
        errors=[r for r in rows if r['error'] and not r['error'].startswith('NOT_RUN')]
        summary={'model':model,'planned':len(rows),'completed_responses':len(valid),
                 'rejected_or_error':len(errors),'not_run':len(notrun),
                 'correct_top1_valid':sum(yes(r['exact_match']) for r in valid),
                 'conditional_top1_valid':sum(yes(r['exact_match']) for r in valid)/len(valid) if valid else None,
                 'mean_llm_latency_s':statistics.mean(float(r['llm_latency_s']) for r in valid) if valid else None,
                 'median_llm_latency_s':statistics.median(float(r['llm_latency_s']) for r in valid) if valid else None,
                 'input_tokens_reported':sum(int(r['input_tokens']) for r in valid),
                 'output_tokens_reported':sum(int(r['output_tokens']) for r in valid),
                 'context_flags':sum(yes(r['context_limit']) for r in rows),
                 'rate_limit_flags':sum(yes(r['rate_limited']) for r in rows),
                 'invalid_name_slots':sum(int(r['num_invalid_skills']) for r in valid),
                 'case_only_top1_extra':sum(yes(r['fuzzy_match']) and not yes(r['exact_match']) for r in valid),
                 'stop':json.loads((path/'status.json').read_text())['stop'],
                 'reasoning_efforts':dict(collections.Counter(a.get('reasoning_effort','not_reported') for a in model_audits))}
        summaries.append(summary)
        for n in bench.SIZES:
            group=[r for r in rows if int(r['library_size'])==n]
            measured=[r for r in group if not r['error'].startswith('NOT_RUN')]
            v=[r for r in group if not r['error']]
            correct=sum(yes(r['exact_match']) and not r['error'] for r in group)
            full=len(measured)==20
            per_size.append({'model':model,'N':n,'planned':20,'recorded':len(measured),'valid':len(v),
                 'correct_top1':correct,'exact_set_accuracy_all':correct/20 if full else None,
                 'selection_precision_valid':correct/len(v) if v else None,
                 'selection_recall_all':correct/20 if full else None,
                 'wilson95_top1':wilson(correct,20) if full else None,
                 'mean_llm_latency_s':statistics.mean(float(r['llm_latency_s']) for r in v) if v else None,
                 'input_tokens_mean':statistics.mean(int(r['input_tokens']) for r in v) if v else None,
                 'recall_at_10':sum(float(r['recall@10']) for r in group)/20 if full else None,
                 'precision_at_10':sum(float(r['precision@10']) for r in group)/20 if full else None,
                 'mrr_at_10':sum(float(r['mrr@10']) for r in group)/20 if full else None,
                 'ndcg_at_10':sum(float(r['ndcg@10']) for r in group)/20 if full else None})
    common=[n for n in bench.SIZES if all(next(s for s in per_size if s['model']==m and s['N']==n)['valid']==20 for m in MODELS)]
    # Same sampled question indices for every model and size preserve pairing and
    # within-question dependence. These are 20 clusters, never 120/140 iid trials.
    rng=np.random.default_rng(20261103)
    indices=rng.integers(0,20,size=(10000,20))
    samples={}
    bootstrap=[]
    for model in MODELS:
        lookup={(int(r['library_size']),r['query']):r for r in all_rows[model]}
        matrix=np.array([[float(yes(lookup[(n,q['query'])]['exact_match'])) for n in common] for q in queries])
        values=matrix[indices].mean(axis=(1,2))
        samples[model]=values
        bootstrap.append({'model':model,'common_sizes':common,'unique_query_clusters':20,'mean_accuracy':float(matrix.mean()),
                          'query_cluster_percentile95':[float(x) for x in np.quantile(values,[.025,.975])],
                          'seed':20261103,'replicates':10000})
    differences=[]
    observed={r['model']:r['mean_accuracy'] for r in bootstrap}
    for i,left in enumerate(MODELS):
        for right in MODELS[i+1:]:
            delta=samples[left]-samples[right]
            differences.append({'left':left,'right':right,'difference_percentage_points':float((observed[left]-observed[right])*100),
                                'paired_query_cluster_percentile95_pp':[float(x*100) for x in np.quantile(delta,[.025,.975])]})
    counts=collections.Counter(s['name'] for s in skills)
    def norm(text):
        return re.sub(r'[^\w]+',' ',text.casefold()).strip()
    leak=[q['gold_skill'] for q in queries if norm(q['gold_skill']) in norm(q['query'])]
    plan=bench.make_plan(skills,queries)
    protocol={'records':len(skills),'unique_names':len(counts),'duplicate_name_groups':sum(v>1 for v in counts.values()),
              'records_with_nonunique_names':sum(v for v in counts.values() if v>1),
              'gold_queries_with_nonunique_name':sum(counts[q['gold_skill']]>1 for q in queries),
              'query_name_phrase_matches_normalized':len(leak),'name_phrase_match_gold_labels':leak,
              'by_size':[]}
    for n in bench.SIZES:
        ts=[t for t in plan if t['n']==n]
        protocol['by_size'].append({'N':n,'gold_replaced_last':sum(t['gold_replaced_last'] for t in ts),
            'gold_positions':[next(i for i,s in enumerate(t['candidates'],1) if s['id']==t['query']['gold_id']) for t in ts],
            'distinct_names':len({s['name'] for s in ts[0]['candidates']}),
            'instructions_characters':len(ts[0]['messages'][0]['content']),
            'instructions_utf8_bytes':len(ts[0]['messages'][0]['content'].encode()),
            'v1_estimated_tokens':n*44.5+200})
    result={'models':summaries,'by_size':per_size,'query_cluster_bootstrap':bootstrap,
            'paired_differences':differences,'protocol_audit':protocol,'validation':validation,
            'unique_questions':20,'comparisons_are_descriptive':True}
    bench.atomic_write(folder/'analysis.json',json.dumps(result,indent=2,ensure_ascii=False))
    bench.atomic_write(folder/'protocol-audit.json',json.dumps(protocol,indent=2,ensure_ascii=False))
    return result


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    colors=['#0072B2','#D55E00','#009E73']
    markers=['o','s','^']
    fig,axes=plt.subplots(1,2,figsize=(12.5,5),layout='constrained')
    for model,color,marker,offset in zip(MODELS,colors,markers,[.97,1,1.03]):
        data=[r for r in result['by_size'] if r['model']==model and r['exact_set_accuracy_all'] is not None]
        xs=np.array([r['N'] for r in data])*offset
        ys=np.array([r['exact_set_accuracy_all'] for r in data])
        ci=np.array([r['wilson95_top1'] for r in data])
        axes[0].errorbar(xs,ys,yerr=[ys-ci[:,0],ci[:,1]-ys],color=color,marker=marker,
                         linewidth=1.8,capsize=2,label=model,alpha=.9)
        axes[1].plot(xs,[r['mean_llm_latency_s'] for r in data],color=color,marker=marker,linewidth=1.8,label=model)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks(bench.SIZES,labels=['50','100','250','500','1000','2000','5000'])
        ax.tick_params(axis='x',labelsize=9)
        ax.set_xlabel('Nombre de skills (IDs)')
        ax.grid(alpha=.17)
        ax.axvspan(3500,6500,color='#eceff2')
        ax.set_xlim(40,6500)
    axes[0].set_ylim(0,1.04)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel('Top-1 exact / 20 questions')
    axes[0].set_title('Précision observée et intervalles de Wilson à 95 %',fontsize=11)
    axes[0].text(4800,.47,'Rejet\nN = 5000\npas de score',ha='center',va='center',fontsize=8)
    axes[0].legend(loc='lower left',fontsize=9)
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel('Secondes, moyenne des réponses valides')
    axes[1].set_title('Latence HTTP du modèle, raisonnement medium',fontsize=11)
    axes[1].text(4800,.45*axes[1].get_ylim()[1],'Rejet\nN = 5000\nnon mesuré',ha='center',va='center',fontsize=8)
    fig.suptitle('Variante OAuth v1 — les 20 mêmes questions à chaque taille',fontsize=13)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    result=analyze(args.out)
    plot(result,Path(args.out)/'figures/scaling.png')
    print(json.dumps({'models':result['models'],'validation':result['validation'],
                      'bootstrap':result['query_cluster_bootstrap'],'differences':result['paired_differences']},indent=2))
