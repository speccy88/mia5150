"""Auditable structural baselines; no dependency/compatibility labels invented."""
import numpy as np
from harness import *
def dsr(ids,quality,docs,query,k=5,blend=.9):
 # S12 equations 6--8, greedy exact small-matrix logdet, quality order output.
 residual=docs-(docs@query)[:,None]*query;z=blend*residual+(1-blend)*docs;z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
 kernel=quality[:,None]*((1+z@z.T)/2)*quality[None,:];chosen=[]
 for _ in range(min(k,len(ids))):
  remaining=[i for i in range(len(ids)) if i not in chosen]
  def gain(i):
   ix=chosen+[i];sign,value=np.linalg.slogdet(kernel[np.ix_(ix,ix)]+np.eye(len(ix))*1e-10);return value if sign>0 else -float('inf')
  chosen.append(max(remaining,key=gain))
 return [ids[i] for i in sorted(chosen,key=lambda i:-quality[i])]
def route(scores,groups,top_domains=1,global_fallback=0):
 # Domain centroids use mean normalized document similarity; no query labels.
 unique=sorted(set(groups));means={g:float(np.mean(scores[np.array(groups)==g])) for g in unique};chosen=sorted(unique,key=lambda g:(-means[g],g))[:top_domains]
 order=np.argsort(-scores,kind='stable');restricted=[int(i) for i in order if groups[i] in chosen]
 if global_fallback:restricted=list(dict.fromkeys([int(i) for i in order[:global_fallback]]+restricted))
 return restricted
def run(role='skillret_embed_06',kind='header'):
 skills=load('skills','test');ids=[s['id'] for s in skills];index={d:i for i,d in enumerate(ids)};docs=np.load(R/f'cache/{role}-{kind}-0-docs.npy');queries=np.load(R/f'cache/{role}-{kind}-0-queries.npy');qmeta=json.loads((R/'manifests/scorer-queries.json').read_text());groups=[s.get('domain') or s.get('major') or 'unknown' for s in skills];save(R/'manifests/domain-map.json',dict(zip(ids,groups)))
 labels=json.loads((R/'manifests/evaluator-qrels.json').read_text());systems={n:{} for n in ['top5','mmr','dsr-embedding-quality','domain1','domain2-global']};selections={n:{} for n in systems}
 for q,v in zip(qmeta,queries):
  scores=docs@v;order=np.argsort(-scores,kind='stable');pool=[ids[i] for i in order[:20]];rel={d:float(scores[index[d]]) for d in pool};vectors={d:docs[index[d]] for d in pool};quality=1/(1+np.exp(-scores[order[:20]]))
  current={'top5':pool[:5],'mmr':mmr(pool,rel,vectors,.8,5,rel[pool[0]]-.2),'dsr-embedding-quality':dsr(pool,quality,docs[order[:20]],v),'domain1':[ids[i] for i in route(scores,groups,1)[:100]],'domain2-global':[ids[i] for i in route(scores,groups,2,5)[:100]]}
  for n,r in current.items():systems[n][q['id']]=r;selections[n][q['id']]=set_metrics(r[:5],labels[q['id']])
 for n,r in systems.items():save(R/f'results/structure-{n}.json',{'rankings':r,'metrics':evaluate(labels,r),'selected_set_metrics':selections[n],'note':'Fixed five-skill budget; no gold count. DSR embedding-quality ablation, not reranker-quality paper reproduction.'})
if __name__=='__main__':run()
