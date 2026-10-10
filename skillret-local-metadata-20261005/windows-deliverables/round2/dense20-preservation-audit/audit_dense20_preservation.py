"""Read-only audit of the already completed matched30 coverage experiment."""
import csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2';sys.path.insert(0,str(ROOT))
from harness import qrels
from probe_contract import digest
from candidate_budget import pools as construct
def main():
 original=HERE/'candidate-budget30-v1';freeze=json.loads((original/'freeze.json').read_text());stored=json.loads((original/'pools.json').read_text());old=json.loads((original/'evaluation.json').read_text())
 files=[ROOT/'results/public-full-skillret_embed_06.json',ROOT/'results/public-full-bm25.json'];sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files};assert sources==freeze['source_hashes']
 dense,bm=[json.loads(p.read_text())['rankings'] for p in files];control=stored['dense30'];treatment=stored['dense20_lexical10']
 assert digest(control)==freeze['control_pools_sha256'] and digest(treatment)==freeze['treatment_pools_sha256']
 assert len(control)==len(treatment)==4392 and set(control)==set(treatment)
 # Verify construction without labels first; qrels are only used afterward.
 for q in control:
  a,b=construct(dense[q],bm[q]);assert a==control[q] and b==treatment[q]
  assert len(b)==len(set(b))==30 and b[:20]==dense[q][:20]
 labels=qrels('test');assert set(labels)==set(control);per=[]
 for q in sorted(control):
  gold={d for d,n in labels[q].items() if n>0};base=set(dense[q][:20]);a=gold&set(control[q]);b=gold&set(treatment[q]);assert gold and (gold&base)<=b
  per.append({'id':q,'positives':len(gold),'dense20_positives':len(gold&base),'dense30_hits':len(a),'hybrid30_hits':len(b),'dense20_positive_losses':len((gold&base)-b),'dense30_positive_losses':len(a-b),'gained_vs_dense30':len(b-a),'dense30_recall30':len(a)/len(gold),'hybrid30_recall30':len(b)/len(gold),'dense30_oracle_recall5':min(5,len(a))/len(gold),'hybrid30_oracle_recall5':min(5,len(b))/len(gold),'dense30_oracle_recall10':min(10,len(a))/len(gold),'hybrid30_oracle_recall10':min(10,len(b))/len(gold)})
 aggregate={k:sum(x[k] for x in per)/len(per) for k in ['dense30_recall30','hybrid30_recall30','dense30_oracle_recall5','hybrid30_oracle_recall5','dense30_oracle_recall10','hybrid30_oracle_recall10']}
 assert abs(aggregate['dense30_recall30']-old['macro_recall30']['dense30'])<1e-12 and abs(aggregate['hybrid30_recall30']-old['macro_recall30']['dense20_lexical10'])<1e-12
 assert sum(x['dense30_positive_losses'] for x in per)==130 and sum(x['gained_vs_dense30'] for x in per)==183
 result={'status':'existing_exact_comparison_verified_not_repeated','query_count':4392,'candidate_budget_each':30,'dense20_prefix_preserved_for_every_query':True,'dense20_positive_pairs_preserved':sum(x['dense20_positives'] for x in per),'dense20_positive_pairs_lost':0,'dense30_positive_pairs_lost':130,'gained_vs_dense30':183,'aggregate':aggregate,'original_oracle_ndcg10':old['oracle_ndcg10'],'original_query_wins_ties_losses':old['wins_ties_losses'],'source_hashes':sources,'original_protocol_sha256':freeze['protocol_sha256'],'original_artifacts_sha256':{n:hashlib.sha256((original/n).read_bytes()).hexdigest() for n in ['freeze.json','pools.json','evaluation.json','per-query.json']},'new_inference_calls':0,'new_api_calls':0,'new_downloads':0,'new_gpu_work':False,'exposure':'All4392 are historically exposed public test queries. No untouched confirmation used. No gold-positive insertion or label-aware construction.','decision':'Do not repeat the exact coverage experiment or run another convenient local reranker panel: mean oracle NDCG10 headroom improves only0.000373 with123 regressing queries. Preserve all evidence. A future comparison needs a separately justified protocol, not a broad sweep.'}
 out=HERE/'dense20-preservation-audit-v1';out.mkdir(exist_ok=False);(out/'audit.json').write_text(json.dumps(result,indent=2))
 with (out/'per-query.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(per[0]));writer.writeheader();writer.writerows(per)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
