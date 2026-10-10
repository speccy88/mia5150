"""Cached coverage on already-reviewed panels only; no inference or new labels."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from harness import qrels,evaluate
HERE=ROOT/'round2'
def main():
 panels={'initial12':[q['id'] for q in json.loads((HERE/'exploration-freeze.json').read_text())['queries']],'selective8':[q['id'] for q in json.loads((HERE/'next-confirmation-8.json').read_text())['queries']]}
 dense=json.loads((ROOT/'results/public-full-skillret_embed_06.json').read_text())['rankings'];bm=json.loads((ROOT/'results/public-full-bm25.json').read_text())['rankings'];all_labels=qrels('test');out={}
 for name,ids in panels.items():
  labels={q:all_labels[q] for q in ids};base={q:dense[q][:20] for q in ids};union={q:list(dict.fromkeys(base[q]+bm[q][:10])) for q in ids}
  oracle=lambda ranks:{q:sorted(r,key=lambda d:(-labels[q].get(d,0),r.index(d))) for q,r in ranks.items()}
  details=[]
  for q in ids:
   gold={d for d,v in labels[q].items() if v>0};new=(gold&set(union[q]))-set(base[q])
   details.append({'id':q,'gold_count':len(gold),'base_covered':len(gold&set(base[q])),'union_covered':len(gold&set(union[q])),'new_gold_ids':sorted(new),'union_size':len(union[q])})
  out[name]={'base_pool_oracle_ndcg10':evaluate(labels,oracle(base))['aggregate']['NDCG@10'],'union_pool_oracle_ndcg10':evaluate(labels,oracle(union))['aggregate']['NDCG@10'],'new_relevant_candidates':sum(len(x['new_gold_ids']) for x in details),'details':details}
 (HERE/'coverage-diagnostic.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
