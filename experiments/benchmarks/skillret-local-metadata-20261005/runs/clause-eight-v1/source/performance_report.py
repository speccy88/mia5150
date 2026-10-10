"""Measured phase timings, cache sizes and failure rates; no fabricated zeros."""
from harness import *
import numpy as np
labels=json.loads((R/'manifests/evaluator-qrels.json').read_text());rows=[]
def quantiles(values):return {'p50':float(np.median(values)),'p95':float(np.quantile(values,.95)),'mean':float(np.mean(values))} if values else None
for role in ['qwen_embed_06','skillret_embed_06','skillrouter_embed_06','skillreason_embed_06']:
 p=R/f'results/{role}-header-0.json'
 if not p.exists():continue
 data=json.loads(p.read_text());lat=[a+b for a,b in zip(data['query_stats']['batch_seconds'],data['search_seconds'])];cache=R/f'cache/{role}-header-0-docs.npy';rows.append({'system':role,'cohort':'128 public development','ndcg10':evaluate(labels,data['rankings'])['aggregate']['NDCG@10'],'phase':'warm query encoding plus exact search; document indexing separate','seconds':quantiles(lat),'startup_seconds':data['startup_seconds'],'document_forward_seconds':sum(data['document_stats']['batch_seconds']),'index_bytes':cache.stat().st_size if cache.exists() else None,'failure_rate':sum(not r for r in data['rankings'].values())/len(data['rankings'])})
for kind in ['header','full']:
 p=R/f'results/bm25-locked-{kind}.json'
 if p.exists():
  data=json.loads(p.read_text());timing=data.get('timing',data.get('timings',{}));lat=list(timing.get('per_query_seconds',{}).values());rows.append({'system':'BM25-'+kind,'cohort':'128 public development','ndcg10':evaluate(labels,data['rankings'])['aggregate']['NDCG@10'],'phase':'query tokenization/retrieval; indexing separate','seconds':quantiles(lat),'index_seconds':timing.get('index_seconds'),'index_bytes':None,'index_note':'Original BM25 index was in memory, not serialized; process resource receipt available. Do not infer zero storage.','failure_rate':sum(not r for r in data['rankings'].values())/len(data['rankings'])})
for name in ['skillret_rerank_06-header-32','skillrouter_rerank_06-header-32','skillreason_rerank_06-header-32','r3_rerank_fp16_local-header-32','jina-native','jevk5-pointwise','plumb-pointwise','qwen8-q4']:
 p=R/f'results/{name}.json'
 if not p.exists():continue
 data=json.loads(p.read_text());lat=[r['seconds'] for r in data['raw']];rows.append({'system':name,'cohort':str(len(data['rankings']))+' public development','ndcg10':data['metrics']['aggregate']['NDCG@10'],'phase':'candidate reranking only; excludes retrieval and loading','seconds':quantiles(lat),'startup_seconds':data.get('startup_seconds'),'index_bytes':None,'failure_rate':data['metrics']['empty_rate']})
save(R/'results/performance.json',{'rows':rows,'interpretation':'Each row uses its stated cohort and native deployment implementation. All recorded query durations including failures enter latency summaries. Different precisions and candidate counts are not a pure hardware/model speed comparison. See resource telemetry for peak RAM/GPU/CPU; missing values are unavailable, not zero.'})
