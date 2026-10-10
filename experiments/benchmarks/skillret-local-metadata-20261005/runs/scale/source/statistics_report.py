from harness import *
import numpy as np
def paired(labels,left,right,seed=5150):
 a=evaluate(labels,left);b=evaluate(labels,right);keys=list(labels);parents={q:q for q in keys};owner={}
 def find(q):
  while parents[q]!=q:parents[q]=parents[parents[q]];q=parents[q]
  return q
 for q,g in labels.items():
  for d in g:
   if d in owner:parents[find(q)]=find(owner[d])
   else:owner[d]=q
 groups={}
 for q in keys:groups.setdefault(find(q),[]).append(q)
 clusters=list(groups.values());rng=np.random.default_rng(seed);out={}
 for metric in ['NDCG@10','Recall@10','Completeness@10','MRR@5']:
  diffs={q:b['per_query'][q][metric]-a['per_query'][q][metric] for q in keys};samples=[]
  for _ in range(2000):
   draw=[q for i in rng.integers(0,len(clusters),len(clusters)) for q in clusters[i]];samples.append(np.mean([diffs[q] for q in draw]))
  out[metric]={'delta':float(np.mean(list(diffs.values()))),'cluster_bootstrap_ci95':np.quantile(samples,[.025,.975]).tolist(),'wins':sum(v>1e-9 for v in diffs.values()),'losses':sum(v< -1e-9 for v in diffs.values()),'ties':sum(abs(v)<=1e-9 for v in diffs.values())}
 return {'queries':len(keys),'gold-overlap_clusters':len(clusters),'contrasts':out,'interpretation':'Exploratory, not multiplicity-adjusted or independent confirmation.'}
out={};labels=json.loads((R/'manifests/evaluator-qrels.json').read_text());dense=json.loads((R/'results/skillret_embed_06-header-0.json').read_text())['rankings'];fusion=json.loads((R/'results/rrf-skillret-bm25.json').read_text())['rankings'];out['fusion_minus_dense']=paired(labels,dense,fusion)
for role in ['skillret_rerank_06','skillrouter_rerank_06','skillreason_rerank_06','r3_rerank_06']:
 p=R/f'results/{role}-header-32.json'
 if p.exists():
  r=json.loads(p.read_text())['rankings'];subset={q:labels[q] for q in r};out[role+'_minus_dense']=paired(subset,dense,r)
for mode in ['hard','replay']:
 a=R/f'results/lora-{mode}-base.json';b=R/f'results/lora-{mode}-adapted.json'
 if a.exists() and b.exists():
  before=json.loads(a.read_text())['rankings'];after=json.loads(b.read_text())['rankings'];l=qrels('train');out[f'lora-{mode}']=paired({q:l[q] for q in before},before,after)
save(R/'results/paired-statistics.json',out);print(out,flush=True)
