"""Paired LoRA-minus-base comparisons for identical views and question clusters."""
import csv
import numpy as np
from common import *

R=STUDY/'results'
PRIMARY_METRIC='NDCG@10'
METRICS=('Top1_ID_accuracy','Hit@1','Recall@1','Recall@5','Recall@10','Recall@20',
    'NDCG@5','NDCG@10','MRR@5','MRR@10','MAP@5','MAP@10','Precision@5','Precision@10',
    'Completeness@5','Completeness@10','Coverage@20','Coverage@50')

def csv_write(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)

def metric(row,name):return row['Coverage@20'] if name=='Recall@20' else row[name]

def contrast(base,treatment,metrics=METRICS):
    qids=list(base);assert set(qids)==set(treatment)
    b=np.asarray([[metric(base[qid],m) for m in metrics] for qid in qids]);t=np.asarray([[metric(treatment[qid],m) for m in metrics] for qid in qids])
    d=t-b;summary=bootstrap(d);rows=[]
    for i,m in enumerate(metrics):
        lo,hi=summary['per_column_95ci'][i];reference=float(b[:,i].mean());effect=float(d[:,i].mean())
        rows.append({'metric':m,'base_mean':reference,'lora_mean':float(t[:,i].mean()),
            'absolute_difference':effect,'difference_percentage_points':effect*100,
            'relative_difference_percent':effect/reference*100 if reference else None,
            'ci95_low':lo,'ci95_high':hi,'ci95_low_percentage_points':lo*100,'ci95_high_percentage_points':hi*100,
            'improved_queries':int(np.sum(d[:,i]>1e-12)),'degraded_queries':int(np.sum(d[:,i]<-1e-12)),
            'unchanged_queries':int(np.sum(np.abs(d[:,i])<=1e-12))})
    return rows,summary,qids,d

def telemetry_summary(rows):
    return {'samples':len(rows),'gpu_peak_sampled_mib':max(r['gpu_memory_mib'] for r in rows),
      'gpu_peak_temperature_c':max(r['gpu_temperature_c'] for r in rows),
      'gpu_utilization_mean_percent':float(np.mean([r['gpu_utilization_percent'] for r in rows])),
      'host_peak_used_gib':max(r['system_used_bytes'] for r in rows)/2**30,
      'host_min_available_gib':min(r['system_available_bytes'] for r in rows)/2**30,
      'host_swap_in_delta_bytes':max(r['swap_in_bytes'] for r in rows)-min(r['swap_in_bytes'] for r in rows),
      'host_swap_out_delta_bytes':max(r['swap_out_bytes'] for r in rows)-min(r['swap_out_bytes'] for r in rows)} if rows else None

def analyze():
    verify_baseline();assert read(R/'validation.json')['completed_requested_scope']
    # Reuse measured cost summarization only; its Qwen comparison contrasts are disabled.
    summary=module('immutable_bge_cost_summary',BASE/'src/analyze.py');summary.CONTRASTS=();summary.analyze()
    costs=read(R/'performance.json');base_costs=read(BASE/'results/performance.json')
    comparisons=[];scaling=[];paired={};per_query_deltas=[];allrows=[];paired_sizes={};size_rows=[]
    official={};base_official={}
    for config in CONFIGS:
        base=read(BASE/'results'/config/'full_evaluation.json');treatment=read(R/config/'full_evaluation.json')
        base_official[config]=base;official[config]=treatment
        for label,data,performance in [('Base',base,base_costs),('LoRA',treatment,costs)]:
            p=performance['configurations'][config]
            comparisons.append({'model_configuration':'BGE-M3 '+label+' '+config,'questions':4392,
                **{m:metric(data['aggregate'],m) for m in METRICS},
                'median_query_total_s':p['uncached_query_total']['median_s'],'p95_query_total_s':p['uncached_query_total']['p95_s'],
                'index_wall_s':p['index_wall_s'],'peak_sampled_vram_mib':p['gpu_peak_sampled_mib'],
                'comparison':'Paired identical inference representation; C observed indexing includes shared vector-cache reuse'})
        rows,boot,qids,deltas=contrast(base['per_query'],treatment['per_query'])
        for row in rows:row.update({'configuration':config,'question_clusters':4392,'contrast':'LoRA minus matching base'})
        allrows.extend(rows)
        paired[config]={'metrics':rows,'bootstrap':boot,'query_ids_sha256':digest(qids),
            'primary_prespecified':config=='B_matched','primary_metric':PRIMARY_METRIC,
            'multiplicity':'B NDCG@10 primary; all other view/metric contrasts exploratory without adjustment'}
        per_query_deltas.extend({'configuration':config,'query_id':qid,**{m:float(deltas[i,j]) for j,m in enumerate(METRICS)}} for i,qid in enumerate(qids))
        paired_sizes[config]={}
        per_size={}
        for n in SIZES:
            b=read(BASE/'results'/config/f'scaling-n{n}.json');t=read(R/config/f'scaling-n{n}.json')
            assert [r['query_id'] for r in b['rows']]==[r['query_id'] for r in t['rows']]
            per_size[n]=(b,t)
            for label,data in [('Base',b),('LoRA',t)]:
                scaling.append({'model':'BGE-M3 '+label,'configuration':config,'skills':n,'questions':20,
                    **{m:metric(data['evaluation']['aggregate'],m) for m in METRICS}})
        # Columns are repeated sizes, with each question retained as one linked row.
        for m in METRICS:
            matrix=np.asarray([[metric(per_size[n][1]['rows'][i],m)-metric(per_size[n][0]['rows'][i],m) for n in SIZES] for i in range(20)])
            boot=bootstrap(matrix);paired_sizes[config][m]=boot
            for j,n in enumerate(SIZES):
                lo,hi=boot['per_column_95ci'][j]
                base_mean=metric(per_size[n][0]['evaluation']['aggregate'],m);effect=boot['mean'][j]
                size_rows.append({'configuration':config,'metric':m,'skills':n,'question_clusters':20,
                    'difference':effect,'difference_percentage_points':effect*100,
                    'relative_difference_percent':effect/base_mean*100 if base_mean else None,'ci95_low':lo,'ci95_high':hi,
                    'improved_queries':int(np.sum(matrix[:,j]>1e-12)),'degraded_queries':int(np.sum(matrix[:,j]<-1e-12)),
                    'unchanged_queries':int(np.sum(np.abs(matrix[:,j])<=1e-12))})
    historical=read(BASE/'results/full_comparison.json')
    # Preserve genuine missing historical fields rather than fill them with zeros.
    for old in (historical[0],historical[-1]):
        comparisons.append({'model_configuration':old['model_configuration'],'questions':4392,
            **{m:old.get('Coverage@20' if m=='Recall@20' else m) for m in METRICS},
            'median_query_total_s':old.get('median_query_total_s'),'p95_query_total_s':old.get('p95_query_total_s'),
            'index_wall_s':old.get('index_wall_s'),'peak_sampled_vram_mib':old.get('peak_vram_sampled_mib'),
            'comparison':old['comparison']})
    atomic_json(R/'full_comparison.json',comparisons);csv_write(R/'full_comparison.csv',comparisons)
    atomic_json(R/'paired_base_vs_lora.json',paired);csv_write(R/'paired_base_vs_lora.csv',allrows)
    csv_write(R/'per_query_lora_differences.csv',per_query_deltas)
    atomic_json(R/'paired_scaling_base_vs_lora.json',paired_sizes);csv_write(R/'paired_scaling_base_vs_lora.csv',size_rows)
    csv_write(R/'scaling_summary.csv',scaling)
    # Separate long-running training/mining/validation from final inference resources.
    within={}
    for label,models in [('Base',base_official),('LoRA',official)]:
        for treatment,reference in [('B_matched','A_metadata'),('C_native8k','B_matched')]:
            rows,boot,qids,_=contrast(models[reference]['per_query'],models[treatment]['per_query'],
                metrics=('Top1_ID_accuracy','Recall@5','Recall@10','NDCG@10','MRR@10'))
            within[label+' '+treatment+' minus '+reference]={'metrics':rows,'bootstrap':boot,'query_ids_sha256':digest(qids),'exploratory':True}
    atomic_json(R/'paired_within_model_views.json',within)
    telemetry=jsonl(R/'telemetry.jsonl');training=[r for r in telemetry if r['phase'] in ('training:pilot','training:full')]
    resources={'main_training':telemetry_summary(training),
       'smoke':telemetry_summary([r for r in telemetry if r['phase']=='training:smoke']),
       'training_only_mining':telemetry_summary([r for r in telemetry if r['phase'].startswith('mining:')]),
       'final_inference':telemetry_summary([r for r in telemetry if any(r['phase'].startswith(c+':') for c in CONFIGS)]),
       'all_lora_phases':telemetry_summary(telemetry),'sample_interval_s':1,
       'caveat':'Whole-host counters and whole-device samples; no electricity measurement; CUDA allocator peaks are separately saved'}
    atomic_json(R/'resources.json',resources)
    csv_write(R/'query_component_comparison.csv',[{'configuration':c,'model':label,
       'embedding_median_s':p['configurations'][c]['uncached_query_embedding']['median_s'],
       'search_median_s':p['configurations'][c]['query_search']['median_s'],
       'total_uncached_median_s':p['configurations'][c]['uncached_query_total']['median_s'],
       'cached_vector_search_median_s':p['configurations'][c]['cached_query_search']['median_s']} for c in CONFIGS for label,p in [('Base',base_costs),('LoRA',costs)]])
    print('Paired identical-view LoRA analysis completed',flush=True)

if __name__=='__main__':analyze()
