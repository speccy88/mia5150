from harness import *
import numpy as np
labels=json.loads((R/'manifests/evaluator-qrels.json').read_text());systems={};summary={}
for name in ['bm25-locked-header','bm25-locked-full','qwen_embed_06-header-0','skillret_embed_06-header-0','skillrouter_embed_06-header-0','skillreason_embed_06-header-0','skillret_embed_06-full-0','tfidf-fields','tfidf-flattened','field-six-uniform']:
 p=R/f'results/{name}.json'
 if p.exists():systems[name]=json.loads(p.read_text())['rankings']
if 'skillret_embed_06-header-0' in systems:
 dense=systems['skillret_embed_06-header-0'];lex=systems['bm25-locked-header'];systems['rrf-skillret-bm25']={q:rrf([dense[q],lex[q]])[:100] for q in labels}
 save(R/'results/rrf-skillret-bm25.json',{'rankings':systems['rrf-skillret-bm25'],'metrics':evaluate(labels,systems['rrf-skillret-bm25'])})
for n,r in systems.items():summary[n]=evaluate(labels,r)
base='bm25-locked-header';contrasts={};rng=np.random.default_rng(5150)
for name,m in summary.items():
 values=np.array([m['per_query'][q]['NDCG@10']-summary[base]['per_query'][q]['NDCG@10'] for q in labels]);boots=np.mean(values[rng.integers(0,len(values),(2000,len(values)))],axis=1);contrasts[name]={'mean_delta':float(values.mean()),'query_bootstrap_ci95':np.quantile(boots,[.025,.975]).tolist(),'caveat':'Exploratory public-development comparison; queries may share source skills; not confirmatory.'}
save(R/'results/comparison.json',{'systems':summary,'contrasts_vs_bm25_header':contrasts});print({n:m['aggregate']['NDCG@10'] for n,m in summary.items()},flush=True)
