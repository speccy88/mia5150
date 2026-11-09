"""Actual uncached primary-system latency, respecting one resident transformer."""
from engine import *

def query_vector(m,tok,text,weight):
 with torch.inference_mode():
  if weight<1:
   if hasattr(m,'disable_adapter'):
    with m.disable_adapter():base=vector(m,tok,text).cpu().numpy()
   else:base=vector(m,tok,text).cpu().numpy()
  if weight>0:adapted=vector(m,tok,text).cpu().numpy()
 return mix_query_vectors(base if weight<1 else None,adapted if weight>0 else None,weight)

def measure_condition(l,condition,sample,docs,dids):
 from reranker import Renderer,score_sequences
 from train_reranker import normalized_fusion
 out=[];cfg=l['released_best_pipeline'] if condition=='released-best-pipeline' else l['reranker'];qc=l['query'];renderer=None
 needs_reranker=condition in ('trained-reranker','trained-combined','released-reranker','released-best-pipeline')
 released=condition in ('released-reranker','released-best-pipeline')
 weight=1 if condition=='adapted-query' else qc['mixture_weight'] if condition in ('mixture-query','trained-combined') else 0
 one_time_load=0
 def encoder():return load(adapter=CACHE/qc['relative_cache_path'] if weight>0 else None)
 if not needs_reranker:
  t=time.perf_counter();m,tok=encoder();one_time_load=time.perf_counter()-t
 for x in sample:
  begin=time.perf_counter();encoder_load=0;reranker_load=0;selection=0;rerank=0
  if needs_reranker:
   t=time.perf_counter();m,tok=encoder();encoder_load=time.perf_counter()-t
  t=time.perf_counter();value=query_vector(m,tok,PREFIX+x['query'],weight);enc=time.perf_counter()-t;t=time.perf_counter();cands,dense=retrieve(value,docs,dids);search=time.perf_counter()-t;ranking=cands
  if needs_reranker:
   del m;torch.cuda.empty_cache();t=time.perf_counter();m,tok=load('reranker',adapter=None if released else CACHE/cfg['relative_cache_path']);reranker_load=time.perf_counter()-t
   if renderer is None:renderer=Renderer(tok,'test')
   t=time.perf_counter();sequences=[];owners=[];k=cfg['candidates']
   for j,sid in enumerate(cands[:k]):
    for tokens,meta in renderer.render(x['query'],sid,cfg['budget'],cfg['full_coverage']):sequences.append(tokens);owners.append(j)
   selection=time.perf_counter()-t;t=time.perf_counter();scores=score_sequences(m,tok,sequences);rerank=time.perf_counter()-t;rr=[max(v for v,j in zip(scores,owners) if j==z) for z in range(k)];order=normalized_fusion(dense[:k],rr,cfg['fusion_weight']);ranking=[cands[j] for j in order]+cands[k:];del m;torch.cuda.empty_cache()
  elapsed=time.perf_counter()-begin;out.append({'query_id':x['id'],'condition':condition,'query_encodings':2 if 0<weight<1 else 1,'encoder_loading_seconds':encoder_load,'uncached_query_encoding_seconds':enc,'candidate_search_seconds':search,'reranker_loading_seconds':reranker_load,'window_selection_seconds':selection,'rerank_seconds':rerank,'true_end_to_end_seconds':elapsed,'ranking':ranking,'includes_model_switching':needs_reranker});print('Actual latency sample',condition,len(out),'/20',round(elapsed,3),flush=True)
 if not needs_reranker:
  del m;torch.cuda.empty_cache()
 return out,{'condition':condition,'sample_rule':'20 smallestSHA256(latency_seed42,query_id), fixed before final inference','query_count':len(out),'median_seconds':float(np.median([x['true_end_to_end_seconds'] for x in out])),'p95_seconds':float(np.percentile([x['true_end_to_end_seconds'] for x in out],95)),'uncached_application_inference':True,'filesystem_page_cache':'not forcibly cleared','one_resident_model':True,'includes_model_loading_and_switching':needs_reranker,'one_time_encoder_load_seconds_excluded_from_query_only_latencies':one_time_load,'note':'Reranker pipelines switch models per request under the one-resident-transformer constraint and include checkpoint verification and model loading in the measured boundary. Query-only systems keep their single encoder resident and record startup separately.'}

def main():
 from evaluate_final import lock
 l=lock();s,q,g=data('test');del s,g;docs,dids=load_vectors('test-documents')
 sample=sorted(q,key=lambda x:digest({'latency_seed':42,'query_id':x['id']}))[:20]
 conditions=[('reconstructed-base','-frozen'),(l['primary_condition'],'')]
 recommendation=l['development_promotion']['deployment_condition']
 if recommendation not in ('archived-base',l['primary_condition']):conditions.append((recommendation,'-deployment'))
 for condition,suffix in conditions:
  records,receipt=measure_condition(l,condition,sample,docs,dids)
  keys=('encoder_loading_seconds','uncached_query_encoding_seconds','candidate_search_seconds',
        'reranker_loading_seconds','window_selection_seconds','rerank_seconds')
  receipt['component_seconds']={key:summary([r[key] for r in records]) for key in keys}
  want={r['query_id'] for r in records};expected={r['query_id']:r for r in iter_rows(STUDY/'results/final'/f'{condition}.jsonl') if r['query_id'] in want}
  assert len(records)==len(want)==len(expected)==20
  receipt['rankings_exact_vs_full_test']=sum(r['ranking']==expected[r['query_id']]['ranking'] for r in records)
  receipt['full_test_inference_failures_on_sample']=sum(r['status']!='ok' for r in expected.values())
  write_rows(STUDY/f'results/latency-end-to-end20{suffix}.jsonl',records)
  save(STUDY/f'results/latency-end-to-end20{suffix}.json',receipt)
if __name__=='__main__':main()
