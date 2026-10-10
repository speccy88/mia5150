"""Frozen cached equal-budget coverage diagnostic; no generation or tuning."""
import collections,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2';sys.path.insert(0,str(ROOT))
from harness import qrels,evaluate
from probe_contract import digest
def pools(dense,lexical):
 if len(dense)<30 or len(set(dense[:30]))!=30:raise ValueError('Dense pool shortfall')
 first=dense[:20];new=[]
 for d in lexical[:100]:
  if d not in first and d not in new:new.append(d)
  if len(new)==10:break
 if len(new)!=10:raise ValueError('Lexical pool shortfall')
 return dense[:30],first+new
def main():
 protocol=json.loads((HERE/'candidate_budget_protocol.json').read_text());out=HERE/'candidate-budget30-v1';out.mkdir(exist_ok=False)
 # Record source hashes and constructed pools before loading qrels.
 dense_path=ROOT/'results/public-full-skillret_embed_06.json';bm_path=ROOT/'results/public-full-bm25.json'
 dense=json.loads(dense_path.read_text())['rankings'];bm=json.loads(bm_path.read_text())['rankings'];assert set(dense)==set(bm) and len(dense)==4392
 control={};treatment={}
 for q in sorted(dense):control[q],treatment[q]=pools(dense[q],bm[q])
 freeze={'protocol':protocol,'protocol_sha256':digest(protocol),'source_hashes':{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in (dense_path,bm_path)},'control_pools_sha256':digest(control),'treatment_pools_sha256':digest(treatment),'query_count':4392,'all_pools_exactly30':True}
 (out/'freeze.json').write_text(json.dumps(freeze,indent=2));(out/'pools.json').write_text(json.dumps({'dense30':control,'dense20_lexical10':treatment}))
 labels=qrels('test');assert set(labels)==set(control)
 domains={}
 with (ROOT/'data/skills/test.jsonl').open() as f:
  for line in f:
   s=json.loads(line);domains[s['id']]=str(s.get('domain') or 'unspecified')
 rows=[];buckets=collections.defaultdict(lambda:{'queries':set(),'pairs':0,'dense_hits':0,'hybrid_hits':0,'gained':0,'lost':0});total=0;hitd=0;hith=0;gain=0;loss=0
 for q,g in labels.items():
  gold={d for d,v in g.items() if v>0};a=gold&set(control[q]);b=gold&set(treatment[q]);total+=len(gold);hitd+=len(a);hith+=len(b);gain+=len(b-a);loss+=len(a-b)
  rows.append({'id':q,'gold_count':len(gold),'dense_hits':len(a),'hybrid_hits':len(b),'dense_recall30':len(a)/len(gold),'hybrid_recall30':len(b)/len(gold),'delta_recall30':(len(b)-len(a))/len(gold),'gained_ids':sorted(b-a),'lost_ids':sorted(a-b),'domains':sorted({domains[d] for d in gold})})
  for d in gold:
   r=buckets[domains[d]];r['queries'].add(q);r['pairs']+=1;r['dense_hits']+=d in a;r['hybrid_hits']+=d in b;r['gained']+=d in b-a;r['lost']+=d in a-b
 oracle=lambda ranks:{q:sorted(pool,key=lambda d:(-labels[q].get(d,0),pool.index(d))) for q,pool in ranks.items()}
 oracle_metrics={k:evaluate(labels,oracle(ranks))['aggregate']['NDCG@10'] for k,ranks in [('dense30',control),('dense20_lexical10',treatment)]}
 domain_rows=[{'domain':name,**{k:v for k,v in r.items() if k!='queries'},'query_count':len(r['queries']),'dense_pair_recall':r['dense_hits']/r['pairs'],'hybrid_pair_recall':r['hybrid_hits']/r['pairs']} for name,r in buckets.items()]
 result={'query_count':len(rows),'positive_pairs':total,'macro_recall30':{'dense30':sum(r['dense_recall30'] for r in rows)/len(rows),'dense20_lexical10':sum(r['hybrid_recall30'] for r in rows)/len(rows)},'pair_coverage':{'dense30_hits':hitd,'hybrid_hits':hith,'dense30_recall':hitd/total,'hybrid_recall':hith/total,'gained_pairs':gain,'lost_pairs':loss,'net_pairs':gain-loss},'completeness30':{'dense30':sum(r['dense_hits']==r['gold_count'] for r in rows)/len(rows),'dense20_lexical10':sum(r['hybrid_hits']==r['gold_count'] for r in rows)/len(rows)},'oracle_ndcg10':oracle_metrics,'wins_ties_losses':[sum(r['delta_recall30']>0 for r in rows),sum(r['delta_recall30']==0 for r in rows),sum(r['delta_recall30']<0 for r in rows)],'worst_regressions':sorted((r for r in rows if r['delta_recall30']<0),key=lambda r:(r['delta_recall30'],r['id']))[:10],'domains':sorted(domain_rows,key=lambda r:(r['hybrid_pair_recall']-r['dense_pair_recall'],r['domain'])),'protocol_sha256':digest(protocol),'paid_calls':0,'caveat':'Exactly30 candidates for both methods, full4392 fixed denominator. Oracle uses labels and is not actual reranker quality. Domains overlap in query membership and are diagnostic gold metadata. Historically exposed public data; no tuning.'}
 (out/'evaluation.json').write_text(json.dumps(result,indent=2));(out/'per-query.json').write_text(json.dumps(rows,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('domains','worst_regressions')},indent=2));print(json.dumps({'worst_regressions':result['worst_regressions'][:3],'domains_n_ge100':[r for r in result['domains'] if r['query_count']>=100]},indent=2))
if __name__=='__main__':main()
