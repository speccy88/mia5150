"""Proposal scaling: identical gold-conditioned nested libraries, three seeds."""
from harness import *
from structures import route
import numpy as np, random, time
skills=load('skills','test');byid={s['id']:s for s in skills};ids=list(byid);ix={d:i for i,d in enumerate(ids)};queries=json.loads((R/'manifests/scorer-queries.json').read_text())[:8];labels=qrels('test');vectors=np.load(R/'cache/skillret_embed_06-header-0-docs.npy');qv=np.load(R/'cache/skillret_embed_06-header-0-queries.npy');cases=[];raw=[];lexcache={}
for q,v in zip(queries,qv):
 scores=vectors@v
 for seed in (5150,5151,5152):
  for condition in ['random','hard-overlap']:
   gold=sorted(labels[q['id']]);distractors=[d for d in ids if d not in gold];random.Random(f'{seed}/{q["id"]}').shuffle(distractors)
   if condition=='hard-overlap':
    rng=np.random.default_rng(seed+int(digest(q['id'])[:8],16));noise=rng.gumbel(scale=.03,size=len(distractors));distractors=[d for d,n in sorted(zip(distractors,noise),key=lambda z:-(float(scores[ix[z[0]]])+z[1]))]
   for size in (10,100,1000,6006):
    library=gold+distractors[:size-len(gold)];random.Random(seed+size).shuffle(library);key=f'{q["id"]}/{seed}/{condition}/{size}';cases.append({'id':key,'query':q['query'],'query_id':q['id'],'seed':seed,'condition':condition,'size':size,'skill_ids':library})
    subset=[byid[d] for d in sorted(library)];cachekey=digest(sorted(library))+q['id'];start=time.perf_counter()
    if cachekey not in lexcache:lexcache[cachekey]=lexical(subset,[q],'header',min(100,size))[0][q['id']]
    lexical_r=lexcache[cachekey];dense=sorted(library,key=lambda d:(-float(scores[ix[d]]),d));subscores=np.array([scores[ix[d]] for d in library]);groups=[byid[d].get('domain') or byid[d].get('major') or 'unknown' for d in library];domain=[library[i] for i in route(subscores,groups,1)];hybrid=rrf([dense[:100],lexical_r]);rankings={'bm25':lexical_r,'vector':dense[:100],'domain-vector':domain[:100],'hybrid':hybrid[:100]}
    for method,r in rankings.items():raw.append({'case':key,'query_id':q['id'],'size':size,'seed':seed,'condition':condition,'method':method,'rankings':r,'metrics':evaluate({q['id']:labels[q['id']]},{q['id']:r})['aggregate'],'selected':set_metrics(r[:5],labels[q['id']])})
    print(key,flush=True)
save(R/'manifests/nested-libraries.json',{'protocol':'Per-query gold-conditioned nested libraries; qrels used only for library construction/evaluation, never scorer input. Not official fixed-pool evaluation. At size 10 Recall@10 is trivial. Hard-overlap distractors use query-dense cosine plus seeded Gumbel noise scale .03; random and hard conditions remain separate.','cases':cases});save(R/'results/scale-header.json',{'raw':raw,'views':['header'],'prompt_family':'handled separately by prompt_scale.py','indexing':'BM25 reindexed inside each distinct library; dense embeddings fixed per document.'})
