"""Paired official comparisons and fixed-denominator status accounting."""
from common import *
import csv

def paired(a,b):
 ids=sorted(a['per_query']);assert ids==sorted(b['per_query']);delta=np.array([b['per_query'][i]['NDCG@10']-a['per_query'][i]['NDCG@10'] for i in ids]);rng=np.random.default_rng(42);samples=[]
 for start in range(0,10000,100):samples.extend(delta[rng.integers(0,len(delta),(100,len(delta)))].mean(axis=1).tolist())
 return {'ndcg10_delta':float(delta.mean()),'ci95':[float(x) for x in np.percentile(samples,[2.5,97.5])],'improved':int((delta>1e-12).sum()),'degraded':int((delta< -1e-12).sum()),'unchanged':int((np.abs(delta)<=1e-12).sum()),'bootstrap_samples':10000,'bootstrap_seed':42,'unit':'question','uncertainty_scope':'conditional on locked selection and observed dataset, not complete seed/pretraining/selection uncertainty'}

def main():
 conditions={};s,q,g=data('test');arch=read(OLD/'results/baseline_skillret_06b.json');conditions['archived-base']=evaluate(g,arch['rankings'])
 for p in (STUDY/'results/final').glob('*.metrics.json'):conditions[p.name[:-13]]=read(p)
 comparisons={name:paired(conditions['archived-base'],ev) for name,ev in conditions.items() if name!='archived-base'}
 if 'released-reranker' in conditions and 'trained-reranker' in conditions:comparisons['local-reranker-training-effect']=paired(conditions['released-reranker'],conditions['trained-reranker'])
 for name,start,end in [('local-reranker-training-in-combined','released-combined','trained-combined'),('query-training-in-reranker-pipeline','trained-reranker','trained-combined'),('released-fusion-effect','released-reranker-unfused','released-reranker'),('trained-fusion-effect','trained-reranker-unfused','trained-reranker'),('local-reranker-training-unfused','released-reranker-unfused','trained-reranker-unfused'),('query-training-vs-reconstructed-start','reconstructed-base','adapted-query'),('query-mixture-vs-reconstructed-start','reconstructed-base','mixture-query')]:
  if start in conditions and end in conditions:comparisons[name]=paired(conditions[start],conditions[end])
 save(STUDY/'results/paired_final_comparisons.json',comparisons)
 timing={};ceilings={};coverage={};status={}
 def span_fraction(windows):
  spans=sorted((a,b) for row in windows for a,b in row['spans']);end=0;count=0
  for a,b in spans:count+=max(0,b-max(a,end));end=max(end,b)
  return count/max(1,windows[0]['body_characters'])
 for name in conditions:
  path=STUDY/'results/final'/f'{name}.jsonl'
  if not records_exist(path):
   if name=='archived-base':status[name]={'query_count':len(g),'no_record':0,'failed_inference':None,'successful_retrieval_miss_at50':sum(not set(arch['rankings'][qid][:50])&positives for qid,positives in g.items()),'note':'Historical saved rankings do not carry a separate inference status'}
   continue
  records=rows(path);valid=[r for r in records if r.get('status')=='ok' and 'timing' in r]
  assert len(records)==len({r['query_id'] for r in records}) and {r['query_id'] for r in records}<=set(g)
  status[name]={'query_count':len(g),'no_record':len(g)-len(records),'failed_inference':sum(r['status']!='ok' for r in records),'successful_retrieval_miss_at50':sum(r['status']=='ok' and not set(r['ranking'][:50])&g[r['query_id']] for r in records)}
  if valid:
   timing[name]={'query_count':len(valid),'failed_or_unmeasured':len(records)-len(valid),'boundary':'Components measured locally; sums exclude model loading/switching and are not an online end-to-end latency. See the separate actual primary latency sample.','seconds':{k:summary([r['timing'][k] for r in valid]) for k in ('encode','search','render','rerank','active_component_sum')}}
   reuse=[r['offline_score_reuse'] for r in valid if 'offline_score_reuse' in r]
   if reuse:
    assert len(reuse)==len(valid)
    direct=[r['uncached_audit_rerank_seconds'] for r in reuse if r['checked_against_original_batch_execution']]
    timing[name]['boundary']='Offline shared-score work. Rerank time measures cache lookups and missing batches; full uncached runtime uses a separate sample.'
    timing[name]['offline_score_reuse']={k:sum(r[k] for r in reuse) for k in ('reused','fresh_inputs','forward_batches','dummy_inputs')}
    timing[name]['uncached_stride_audit_rerank_seconds']=summary(direct) if direct else None
  if records and 'candidates' in records[0]:
   ceilings[name]={}
   for k in (20,50):
    recalls=[];ideals=[]
    for r in records:
     positives=g[r['query_id']];hits=len(set(r.get('candidates',[])[:k])&positives);recalls.append(hits/len(positives));ideals.append(sum(1/math.log2(i+2) for i in range(min(10,hits)))/sum(1/math.log2(i+2) for i in range(min(10,len(positives)))))
    ceilings[name][str(k)]={'candidate_macro_recall':float(np.mean(recalls)),'ideal_candidate_ndcg10':float(np.mean(ideals)),'fixed_query_count':len(records)}
   spans=[span_fraction(w) for r in records for w in r.get('selections',[]) if w]
   if spans:coverage[name]={'candidate_count':len(spans),'union_body_character_fraction':summary(spans),'metadata_tokens_omitted_total':sum(x['metadata_tokens_omitted'] for r in records for ws in r.get('selections',[]) for x in ws),'boundary':'Source offsets are unioned so overlapping evidence is not counted twice.'}
 geometry={};b=np.load(CACHE/'vectors/test-base-queries/vectors.npy',mmap_mode='r').astype(np.float64);bn=np.linalg.norm(b,axis=1)
 for label,name in [('seed42','test-adapted-queries'),('seed43','test-adapted-queries-seed43')]:
  p=CACHE/'vectors'/name/'vectors.npy'
  if not p.exists():continue
  a=np.load(p,mmap_mode='r').astype(np.float64);cos=np.einsum('ij,ij->i',a,b)/(np.linalg.norm(a,axis=1)*bn);angles=np.degrees(np.arccos(np.clip(cos,-1.,1.)));geometry[label]={'paired_queries':len(cos),'cosine':summary(cos.tolist()),'angle_degrees':summary(angles.tolist()),'document_index_unchanged':True,'interpretation':'Measured preservation, not causal attribution to the anchor term without an anchor ablation.'}
 save(STUDY/'results/query_geometry.json',geometry)
 save(STUDY/'results/timing_components.json',timing);save(STUDY/'results/candidate_ceilings.json',ceilings);save(STUDY/'results/evidence_coverage.json',coverage);save(STUDY/'results/final_status_counts.json',status)
 metrics=list(next(iter(conditions.values()))['aggregate']);path=STUDY/'results/final_metrics.csv'
 with path.open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['condition','queries','missing','failed_inference','successful_miss_at50',*metrics]);writer.writeheader()
  for name,ev in conditions.items():
   counts=status[name]
   writer.writerow({'condition':name,'queries':ev['query_count'],'missing':counts['no_record'],'failed_inference':counts['failed_inference'],'successful_miss_at50':counts['successful_retrieval_miss_at50'],**{k:ev['aggregate'][k] for k in metrics}})
 print(json.dumps({name:{'NDCG@10':ev['aggregate']['NDCG@10'],'Recall@10':ev['aggregate']['Recall@10'],'MRR@10':ev['aggregate']['MRR@10']} for name,ev in conditions.items()},indent=2))
if __name__=='__main__':main()
