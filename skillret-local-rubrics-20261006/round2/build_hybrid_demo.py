"""Build a self-contained cached demo; label display never reaches inference."""
import json,sys,dataclasses
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2';sys.path.insert(0,str(ROOT))
from harness import qrels,evaluate
from hybrid_retrieval import CachedRetriever,CachedJev,HybridRetriever
def main():
 specs=[('Initial12: posthoc fusion','jev-typed-12-v1','exploration-freeze.json','Posthoc fusion; exploratory.'),('Selective8: mixed result','selective-jev-8-v1','next-confirmation-8.json','Fixed selective policy; no-call cases shown as unavailable rerank.'),('Confirmation8: fixed fusion','fusion-confirmation-8-v1','fusion-confirmation-plan.json','Fixed before new predictions; public historical exposure.')]
 sources=[];needed=set()
 for title,folder,manifest,note in specs:
  rows=json.loads((HERE/folder/'results.json').read_text())['rows'];queries={q['id']:q for q in json.loads((HERE/manifest).read_text())['queries']}
  sources.append((title,rows,queries,note));needed.update(d for row in rows for d in row['candidate_ids'])
 headers={}
 with (ROOT/'data/skills/test.jsonl').open() as f:
  for line in f:
   s=json.loads(line)
   if s['id'] in needed:headers[s['id']]={k:s[k] for k in ('id','name','description')}
 labels=qrels('test');panels=[];verified=0
 for title,rows,queries,note in sources:
  pipeline=HybridRetriever(CachedRetriever({r['id']:r['candidate_ids'] for r in rows}),headers,CachedJev(rows));view=[]
  for row in rows:
   q=queries[row['id']];result=pipeline.search(q);metrics={k:evaluate({q['id']:labels[q['id']]},{q['id']:v})['aggregate']['NDCG@10'] for k,v in [('baseline',result.baseline),('jev',result.reranked),('fusion',result.fused)]}
   if row['status']=='ok':
    if 'ranking' in row:assert result.fused==row['ranking']
    verified+=1
   delta=metrics['fusion']-metrics['baseline']
   view.append({'id':q['id'],'query':q['query'],'baseline':result.baseline,'reranked':result.reranked,'fused':result.fused,'status':result.status,'gold':[d for d,v in labels[q['id']].items() if v>0],**{'ndcg_'+k:v for k,v in metrics.items()},'outcome':'No rerank' if result.status!='ok' else 'Gain' if delta>1e-12 else 'Loss' if delta< -1e-12 else 'Tie'})
  panels.append({'title':title,'rows':view,'note':note})
 data={'panels':panels,'skills':headers,'evidence':'Shared commitments $1.75/$4; all reservations retained. No charge for using this demo. Original incomplete OpenRouter result preserved separately. Full provenance and negative findings: ROUND2_CLASS_DEMO.md.'}
 template=(HERE/'hybrid_demo.html').read_text();embedded=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
 (HERE/'hybrid-demo.html').write_text(template.replace('__DATA__',embedded));(HERE/'hybrid-demo-data.json').write_text(json.dumps(data,indent=2))
 (HERE/'hybrid-demo-verification.json').write_text(json.dumps({'cases':sum(len(x['rows']) for x in panels),'successful_cache_replays':verified,'known_fusion_outputs_matched':True,'network_calls':0},indent=2));print(json.dumps({'cases':28,'successful_cache_replays':verified,'network_calls':0,'html':str(HERE/'hybrid-demo.html')}))
if __name__=='__main__':main()
