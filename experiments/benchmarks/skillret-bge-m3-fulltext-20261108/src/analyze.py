"""Source-backed summaries and paired question-cluster bootstrap contrasts."""
import csv
import json
from pathlib import Path
import numpy as np
from common import (STUDY,QWEN,CONFIGS,SIZES,atomic_json,bootstrap,digest,file_hash,
                    jsonl,latency_summary,panel)

R=STUDY/'results'
METRICS=('Top1_ID_accuracy','Recall@5','Recall@10','NDCG@10','MRR@10')
CONTRASTS=(('B_matched','Qwen'),('A_metadata','Qwen'),('C_native8k','Qwen'),('B_matched','A_metadata'),('C_native8k','B_matched'))


def write_csv(path,rows):
    if rows:
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def read(path):
    return json.loads(path.read_text())


def analyze():
    audit=read(R/'input_audit.json');calls=jsonl(R/'embedding_calls.jsonl');telemetry=jsonl(R/'telemetry.jsonl')
    history=jsonl(R/'indexing_history.jsonl');progress=read(R/'progress.json')
    perfs={};scaling=[];fulls={'Qwen':read(QWEN/'results/full_evaluation.json')}
    per_queries={'Qwen':fulls['Qwen']['per_query']}
    fulltable=[];scaling_data={'Qwen':{n:read(QWEN/f'results/scaling-n{n}.json') for n in SIZES}}
    all_query_rows=[]
    for config in CONFIGS:
        folder=R/config
        if not (folder/'full_evaluation.json').exists():continue
        full=read(folder/'full_evaluation.json');fulls[config]=full;per_queries[config]=full['per_query']
        rows=jsonl(folder/'full_rankings.jsonl');all_query_rows.extend(rows)
        stages=[h for h in history if h['config']==config]
        samples=[s for s in telemetry if s['phase'].startswith(config+':')]
        events=[c for c in calls if c['phase'].startswith(config+':')]
        sums={key:sum(h[key] for h in stages) for key in ('index_wall_s','uncached_calls','cached_calls','uncached_tokens','encoding_s_this_stage','logical_encoding_s_from_measured_receipts')}
        sums['uncached_input_tokens_per_active_second']=sums['uncached_tokens']/sums['encoding_s_this_stage'] if sums['encoding_s_this_stage'] else None
        sums['uncached_chunks_per_active_second']=sums['uncached_calls']/sums['encoding_s_this_stage'] if sums['encoding_s_this_stage'] else None
        sums.update({'skills':6006,'chunks':audit['views'][config]['chunks'],
            'gpu_peak_sampled_mib':max((s['gpu_memory_mib'] for s in samples),default=None),
            'cuda_peak_allocated_mib':max((c['cuda_peak_allocated_bytes']/1024**2 for c in events),default=None),
            'cuda_peak_reserved_mib':max((c['cuda_peak_reserved_bytes']/1024**2 for c in events),default=None),
            'system_peak_used_gib':max((s['system_used_bytes']/1024**3 for s in samples),default=None),
            'system_min_available_gib':min((s['system_available_bytes']/1024**3 for s in samples),default=None),
            'gpu_peak_temperature_c':max((s['gpu_temperature_c'] for s in samples),default=None),
            'gpu_utilization_mean_percent':float(np.mean([s['gpu_utilization_percent'] for s in samples])) if samples else None,
            'host_swap_in_delta_bytes':max((s['swap_in_bytes'] for s in samples),default=0)-min((s['swap_in_bytes'] for s in samples),default=0),
            'host_swap_out_delta_bytes':max((s['swap_out_bytes'] for s in samples),default=0)-min((s['swap_out_bytes'] for s in samples),default=0),
            'logical_vector_bytes_float32':audit['views'][config]['chunks']*1024*4,
            'uncached_query_embedding':latency_summary([r['query_embedding_uncached_s'] for r in rows]),
            'uncached_query_total':latency_summary([r['total_retrieval_uncached_s'] for r in rows]),
            'query_search':latency_summary([r['similarity_search_s'] for r in rows]),
            'cached_query_search':latency_summary([r['cached_query_search_s'] for r in rows])})
        perfs[config]=sums
        fulltable.append({'model_configuration':config,'questions':4392,'library_size':6006,
            **full['aggregate'],'median_query_total_s':sums['uncached_query_total']['median_s'],
            'p95_query_total_s':sums['uncached_query_total']['p95_s'],'index_wall_s':sums['index_wall_s'],
            'index_cached_chunks':sums['cached_calls'],'peak_vram_sampled_mib':sums['gpu_peak_sampled_mib'],
            'comparison':'same pinned corpus/query/qrels; B exact Qwen spans; encoder/tokenizer/prefix/precision/runtime differ'})
        scaling_data[config]={n:read(folder/f'scaling-n{n}.json') for n in SIZES}
        for n,data in scaling_data[config].items():
            scaling.append({'model':'BAAI/bge-m3','config':config,'library_size':n,'questions':20,**data['evaluation']['aggregate'],
                'median_query_total_s':float(np.median([r['total_retrieval_uncached_s'] for r in data['rows']]))})
    qperf=read(QWEN/'results/performance.json')
    fulltable.insert(0,{'model_configuration':'Qwen3 8B Q4_K_M full text','questions':4392,'library_size':6006,
        **fulls['Qwen']['aggregate'],'median_query_total_s':qperf['full_uncached_retrieval_including_cache_persistence']['median_s'],
        'p95_query_total_s':qperf['full_uncached_retrieval_including_cache_persistence']['p95_s'],
        'index_wall_s':qperf['index_wall_seconds_completed_stages'],'index_cached_chunks':None,
        'peak_vram_sampled_mib':qperf['primary_indexing_query_gpu_peak_sampled_mib'],
        'comparison':'Immutable archived run; full source Qwen-token chunks and query instruction; Ollama HTTP runtime'})
    base_path=STUDY.parent/'skillret-embedding-jev-fusion/results/baseline_skillret_06b.json';base=read(base_path)['evaluation']
    # Use only matching columns; no historical latency is inferred.
    historical={key:None for key in fulltable[0]}
    historical.update({'model_configuration':'SkillRet Embedding 0.6B full text','questions':4392,'library_size':6006,
        **base['aggregate'],'comparison':'Context only: stripped fields, 32K cap, two truncated documents; latency boundary not matched'})
    fulltable.append(historical)
    # Canonical columns are fixed by the Qwen row; all newer rows include same metric set.
    write_csv(R/'full_comparison.csv',fulltable);atomic_json(R/'full_comparison.json',fulltable)
    write_csv(R/'scaling_summary.csv',scaling);atomic_json(R/'performance.json',{'configurations':perfs,
        'all_monitored_phases_including_loading_and_feasibility':{
            'gpu_peak_sampled_mib':max((s['gpu_memory_mib'] for s in telemetry),default=None),
            'system_peak_used_gib':max((s['system_used_bytes']/1024**3 for s in telemetry),default=None),
            'system_min_available_gib':min((s['system_available_bytes']/1024**3 for s in telemetry),default=None),
            'gpu_peak_temperature_c':max((s['gpu_temperature_c'] for s in telemetry),default=None),
            'host_swap_in_delta_bytes':max((s['swap_in_bytes'] for s in telemetry),default=0)-min((s['swap_in_bytes'] for s in telemetry),default=0),
            'host_swap_out_delta_bytes':max((s['swap_out_bytes'] for s in telemetry),default=0)-min((s['swap_out_bytes'] for s in telemetry),default=0)},
        'model':read(R/'model.json'),'total_primary_wall_s':progress.get('wall_seconds_this_invocation'),
        'index_cost_boundary':'Per-view wall includes shared-cache reuse/resume. Logical encoding sum uses measured original vector receipts, not a fresh standalone wall run.',
        'query_boundary':'Fresh batch-one encode plus source/ID validation, atomic vector cache persistence and CPU search; ranking-file fsync excluded, as in Qwen.',
        'sampled_telemetry_caveat':'One-second whole-device samples can miss subsecond peaks; CUDA allocator peaks reported separately.'})
    write_csv(R/'official_query_timings.csv',[{k:v for k,v in row.items() if k not in ('ranking','scores')} for row in all_query_rows])
    # QID alignment is explicit; a cluster is one question, never query x size.
    qids=list(fulls['Qwen']['per_query']);paired={};deltas=[]
    for treatment,reference in CONTRASTS:
        if treatment not in per_queries or reference not in per_queries:continue
        assert set(per_queries[treatment])==set(per_queries[reference])==set(qids)
        matrix=np.asarray([[per_queries[treatment][qid][m]-per_queries[reference][qid][m] for m in METRICS] for qid in qids])
        summary=bootstrap(matrix);key=treatment+'-minus-'+reference
        paired[key]={'metrics':list(METRICS),'query_ids_sha256':digest(qids),'paired_question_bootstrap':summary,
            'primary_prespecified':treatment=='B_matched' and reference=='Qwen',
            'multiplicity':'Primary B-minus-Qwen NDCG@10; other contrasts/metrics exploratory without multiplicity adjustment'}
        for j,m in enumerate(METRICS):
            lo,hi=summary['per_column_95ci'][j]
            deltas.append({'treatment':treatment,'reference':reference,'metric':m,'question_clusters':4392,
                'mean_difference':summary['mean'][j],'ci95_low':lo,'ci95_high':hi,'ci_excludes_zero':bool(lo>0 or hi<0)})
    atomic_json(R/'paired_official.json',paired);write_csv(R/'paired_official.csv',deltas)
    paired_scaling={};scaling_deltas=[]
    for treatment,reference in CONTRASTS:
        if treatment not in scaling_data or reference not in scaling_data:continue
        key=treatment+'-minus-'+reference;paired_scaling[key]={}
        for metric in METRICS:
            matrix=np.asarray([[scaling_data[treatment][n]['rows'][i][metric]-scaling_data[reference][n]['rows'][i][metric] for n in SIZES] for i in range(20)])
            summary=bootstrap(matrix);paired_scaling[key][metric]=summary
            for j,n in enumerate(SIZES):
                lo,hi=summary['per_column_95ci'][j]
                scaling_deltas.append({'treatment':treatment,'reference':reference,'metric':metric,'library_size':n,
                    'question_clusters':20,'mean_difference':summary['mean'][j],'ci95_low':lo,'ci95_high':hi})
    atomic_json(R/'paired_scaling.json',paired_scaling);write_csv(R/'paired_scaling.csv',scaling_deltas)
    references=[{'model':'bge-small-en-v1.5','NDCG@5':.5257,'NDCG@10':.5451,'Recall@10':.6001,'Completeness@10':.4397},
                {'model':'bge-large-en-v1.5','NDCG@5':.5704,'NDCG@10':.5900,'Recall@10':.6437,'Completeness@10':.4834}]
    p=STUDY.parent/'skillret-local-metadata-20261005/official/README.md'
    for row in references:row.update({'evidence':'Published official README reference, not locally rerun here; document/protocol not matched to complete-source max chunks',
        'source':str(p.relative_to(STUDY.parents[2])),'source_sha256':file_hash(p),'questions_reported':4392,'skills_reported':6006})
    atomic_json(R/'contextual_bge_references.json',references);write_csv(R/'contextual_bge_references.csv',references)
    print('Analysis summaries compiled',len(perfs),'measured BGE configurations',flush=True)

if __name__=='__main__':analyze()
