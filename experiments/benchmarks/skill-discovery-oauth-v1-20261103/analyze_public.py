"""Recheck distributed CSV calculations, bootstrap and prompts offline; no credentials."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import bench
from admission import MODELS
from analyze_results import plot, validate_rows, wilson, yes


def main():
    root=Path(__file__).resolve().parent
    result=json.loads((root/'results/analysis.json').read_text(encoding='utf-8'))
    skills,queries=bench.load_data(root/'data/test.jsonl',root/'data/test_queries.json')
    expected={q['query']:q for q in queries}
    plan=bench.make_plan(skills,queries)
    all_rows={}
    checks=[]
    for model in MODELS:
        folder=root/'results'/model
        with (folder/'details.csv').open(encoding='utf-8',newline='') as f:
            reader=csv.DictReader(f)
            assert reader.fieldnames==bench.COLUMNS
            rows=list(reader)
        all_rows[model]=rows
        # Recover rank directly from the retained normalized label sequence, not
        # from rank_of_gold. Full HTTP-text normalization cannot be replayed.
        for row in rows:
            labels=json.loads(row['ranked_skills'])
            label_rank=next((i for i,label in enumerate(labels,1) if label==row['gold_skill']),0)
            assert int(row['rank_of_gold'])==label_rank,(model,row['library_size'],'label rank')
        # The stored CSV/SQLite agreement in results-validation.json is historical
        # evidence. This public pass checks the CSV only, never claims a new DB check.
        self_rows={(int(r['library_size']),r['query']):r for r in rows if not r['error'].startswith('NOT_RUN')}
        validate_rows(model,rows,expected,self_rows)
        valid=[r for r in rows if not r['error']]
        summary=next(s for s in result['models'] if s['model']==model)
        actual={'planned':len(rows),'completed_responses':len(valid),
                'rejected_or_error':sum(bool(r['error']) and not r['error'].startswith('NOT_RUN') for r in rows),
                'not_run':sum(r['error'].startswith('NOT_RUN') for r in rows),
                'correct_top1_valid':sum(yes(r['exact_match']) for r in valid),
                'conditional_top1_valid':sum(yes(r['exact_match']) for r in valid)/len(valid),
                'mean_llm_latency_s':statistics.mean(float(r['llm_latency_s']) for r in valid),
                'median_llm_latency_s':statistics.median(float(r['llm_latency_s']) for r in valid),
                'input_tokens_reported':sum(int(r['input_tokens']) for r in valid),
                'output_tokens_reported':sum(int(r['output_tokens']) for r in valid),
                'context_flags':sum(yes(r['context_limit']) for r in rows),
                'rate_limit_flags':sum(yes(r['rate_limited']) for r in rows),
                'invalid_name_slots':sum(int(r['num_invalid_skills']) for r in valid),
                'case_only_top1_extra':sum(yes(r['fuzzy_match']) and not yes(r['exact_match']) for r in valid)}
        for key,value in actual.items():
            assert summary[key]==value,(model,key)
        for n in bench.SIZES:
            group=[r for r in rows if int(r['library_size'])==n]
            measured=[r for r in group if not r['error'].startswith('NOT_RUN')]
            v=[r for r in group if not r['error']]
            saved=next(s for s in result['by_size'] if s['model']==model and s['N']==n)
            correct=sum(yes(r['exact_match']) for r in v)
            assert saved['recorded']==len(measured) and saved['valid']==len(v)
            assert saved['correct_top1']==correct
            assert saved['exact_set_accuracy_all']==(correct/20 if len(measured)==20 else None)
            assert saved['wilson95_top1']==(wilson(correct,20) if len(measured)==20 else None)
            assert saved['mean_llm_latency_s']==(statistics.mean(float(r['llm_latency_s']) for r in v) if v else None)
            assert saved['input_tokens_mean']==(statistics.mean(int(r['input_tokens']) for r in v) if v else None)
            for metric in ('recall','precision','mrr','ndcg'):
                value=sum(float(r[f'{metric}@10']) for r in group)/20 if len(measured)==20 else None
                assert saved[f'{metric}_at_10']==value
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        planned=manifest['plan']
        for t,saved in zip(plan,planned,strict=True):
            assert saved['candidate_ids']==[s['id'] for s in t['candidates']]
            assert saved['prompt_sha256']==hashlib.sha256(bench.canonical(t['messages']).encode()).hexdigest()
        checks.append({'model':model,'rows':len(rows),'schema_columns':45,
                       'csv_arithmetic_and_saved_summaries_match':True,'all_140_prompt_hashes_match':True})
    rng=np.random.default_rng(20261103)
    indices=rng.integers(0,20,size=(10000,20))
    bootstrap={}
    means={}
    for saved in result['query_cluster_bootstrap']:
        model=saved['model']
        lookup={(int(r['library_size']),r['query']):r for r in all_rows[model]}
        matrix=np.array([[float(yes(lookup[(n,q['query'])]['exact_match'])) for n in saved['common_sizes']] for q in queries])
        samples=matrix[indices].mean(axis=(1,2))
        bootstrap[model]=samples
        means[model]=float(matrix.mean())
        assert saved['mean_accuracy']==means[model]
        assert saved['query_cluster_percentile95']==[float(x) for x in np.quantile(samples,[.025,.975])]
    for saved in result['paired_differences']:
        left,right=saved['left'],saved['right']
        assert math.isclose(saved['difference_percentage_points'],(means[left]-means[right])*100,abs_tol=1e-12)
        assert saved['paired_query_cluster_percentile95_pp']==[float(x*100) for x in np.quantile(bootstrap[left]-bootstrap[right],[.025,.975])]
    plot(result,root/'figures/scaling.png')
    proof={'offline_csv_validation':True,'models':checks,'query_cluster_bootstrap_reproduced':True,
           'sqlite_rechecked_in_public_package':False,'model_calls':0,'credentials_required':False}
    (root/'validation/public-reproduction.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof,indent=2))


if __name__=='__main__':
    main()
