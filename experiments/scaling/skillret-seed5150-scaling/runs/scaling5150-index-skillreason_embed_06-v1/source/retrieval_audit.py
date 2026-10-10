"""Candidate coverage and blinded review material; evaluator-only labels."""
from harness import *
import numpy as np
labels=json.loads((R/'manifests/evaluator-qrels.json').read_text());base=json.loads((R/'results/skillret_embed_06-header-0.json').read_text())['rankings'];lex=json.loads((R/'results/bm25-locked-header.json').read_text())['rankings'];fusion=json.loads((R/'results/rrf-skillret-bm25.json').read_text())['rankings'];out={};candidates={}
for k in [20,50,100]:
 rows=[]
 for q,g in labels.items():
  gold=set(g);a=set(base[q][:k]);b=set(lex[q][:k]);c=set(fusion[q][:k]);rows.append({'id':q,'gold_count':len(gold),'overlap_jaccard':len(a&b)/len(a|b) if a|b else 0,'lexical_unique_gold':sorted((b-a)&gold),'fusion_rescued_gold':sorted((c-a)&gold),'fusion_displaced_gold':sorted((a-c)&gold),'dense_recall':len(a&gold)/len(gold),'union_recall':len((a|b)&gold)/len(gold),'fusion_recall':len(c&gold)/len(gold)})
  for channel,r in [('dense',base),('lexical',lex),('rrf',fusion)]:candidates.setdefault(f'{channel}-{k}',{})[q]=r[q][:k]
 out[k]={'per_query':rows,'mean_dense_recall':float(np.mean([r['dense_recall'] for r in rows])),'mean_union_recall':float(np.mean([r['union_recall'] for r in rows])),'mean_fusion_recall':float(np.mean([r['fusion_recall'] for r in rows])),'rescued_positive_count':sum(len(r['fusion_rescued_gold']) for r in rows),'displaced_positive_count':sum(len(r['fusion_displaced_gold']) for r in rows)}
save(R/'results/candidate-coverage.json',out)
for name,r in candidates.items():save(R/f'manifests/candidates-{name}.json',{'rankings':r,'sha256':digest(r),'immutable_protocol':'This artifact is frozen; changing generator requires a new filename/run.'})
queries={q['id']:q['query'] for q in json.loads((R/'manifests/scorer-queries.json').read_text())};review=[]
for q in sorted(labels,key=digest)[:24]:
 gold=set(labels[q]);swap=int(digest(q)[:2],16)%2;rankings=[base[q],fusion[q]];review.append({'id':q,'query':queries[q],'gold_ids':sorted(gold),'system_A':rankings[swap][:10],'system_B':rankings[1-swap][:10],'annotations':{'dense_pool_miss':bool(gold-set(base[q][:20])),'dense_ranking_miss_within_pool':bool(gold-set(base[q][:5])) and gold.issubset(base[q][:20]),'multi_skill_partial':len(gold)>1 and 0<len(gold&set(base[q][:5]))<len(gold)},'reviewer_note':'Assign semantic causes after blind review; downstream wrong-use cannot be inferred from retrieval.'})
save(R/'evidence/blind-review.json',review);save(R/'evidence/blind-review-key.json',{q:('A=dense,B=RRF' if int(digest(q)[:2],16)%2==0 else 'A=RRF,B=dense') for q in labels});print('Coverage and review artifacts saved.')
