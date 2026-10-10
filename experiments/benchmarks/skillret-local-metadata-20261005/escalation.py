"""Disjoint calibration/evaluation of a bounded local routing policy."""
from harness import *
import numpy as np

base=json.loads((R/'results/skillret_embed_06-header-0.json').read_text());strong=json.loads((R/'results/skillret_rerank_06-header-32.json').read_text())
queries=json.loads((R/'manifests/scorer-queries.json').read_text());docs=np.load(R/'cache/skillret_embed_06-header-0-docs.npy');qv=np.load(R/'cache/skillret_embed_06-header-0-queries.npy')
ids=list(strong['rankings']);calibration=ids[:16];evaluation=ids[16:];labels=json.loads((R/'manifests/evaluator-qrels.json').read_text());margins={}
for q,v in zip(queries,qv):
 s=np.sort(docs@v);margins[q['id']]=float(s[-1]-s[-2])
threshold=float(np.median([margins[q] for q in calibration]))
ranking={q:(strong['rankings'][q] if margins[q]<=threshold else base['rankings'][q]) for q in evaluation}
rawstrong=strong.get('raw',[]);costs={row['id']:row['seconds'] for row in rawstrong};lat=[]
for q in evaluation:
 i=next(i for i,x in enumerate(queries) if x['id']==q);lat.append(base['query_stats']['batch_seconds'][i]+base['search_seconds'][i]+(costs.get(q,0) if margins[q]<=threshold else 0))
save(R/'results/escalation.json',{'calibration_queries':calibration,'evaluation_queries':evaluation,'threshold':threshold,'rule':'Rerank when top1-top2 dense cosine margin <= calibration median; no label-tuned threshold. Fixed 50% calibration budget.','rankings':ranking,'metrics':evaluate({q:labels[q] for q in evaluation},ranking),'baseline':evaluate({q:labels[q] for q in evaluation},base['rankings']),'always_strong':evaluate({q:labels[q] for q in evaluation},strong['rankings']),'escalation_rate':sum(margins[q]<=threshold for q in evaluation)/len(evaluation),'estimated_warm_latency_seconds':{'mean':float(np.mean(lat)),'p50':float(np.median(lat)),'p95':float(np.quantile(lat,.95))},'latency_note':'Sum of separately measured warm query encoding, search and reranking; excludes model switching/startup, not deployment end-to-end latency.','margins':margins})
print('Escalation development diagnostic saved.')
