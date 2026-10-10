"""Frozen zero-paid lexical representation test. Run under guard.py."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2';sys.path.insert(0,str(ROOT))
from harness import lexical,rrf,evaluate,qrels
from request_clause_view import request_view
from probe_contract import digest
from jev_probe import save
PIN='3e125ee205be9fd1ee267c3b9e6799ccef96a70005cfc36938831b98c6e97201'
def main():
 p=json.loads((HERE/'request-clause-protocol.json').read_text());assert p.pop('protocol_sha256')==PIN and digest(p)==PIN
 assert hashlib.sha256((HERE/'request_clause_view.py').read_bytes()).hexdigest()==p['rule_sha256']
 out=HERE/'clause-8-v1';out.mkdir(exist_ok=False)
 queries=[]
 for q in p['queries']:
  assert request_view(q['query'])==q['request_view']
  queries.extend([{'id':q['id'],'query':q['query']},{'id':q['id']+'::view','query':q['request_view']}])
 skills=[];h=hashlib.sha256()
 with (ROOT/'data/skills/test.jsonl').open('rb') as f:
  for line in f:
   h.update(line);s=json.loads(line);skills.append({k:s[k] for k in ('id','name','description')})
 assert h.hexdigest()=='1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81'
 ranks,timing=lexical(skills,queries,'header',top=100)
 cached=json.loads((ROOT/'results/public-full-bm25.json').read_text())['rankings'];ids=[q['id'] for q in p['queries']]
 mismatch=[q for q in ids if ranks[q]!=cached[q]]
 save(out/'baseline-reproduction.json',{'query_count':8,'exact_top100_matches':8-len(mismatch),'mismatched_ids':mismatch,'protocol_sha256':PIN})
 if mismatch:raise ValueError('Cached baseline reproduction failed; no outcome claims')
 original={q:ranks[q][:20] for q in ids};combined={q:rrf([ranks[q],ranks[q+'::view']])[:20] for q in ids}
 save(out/'predictions.json',{'baseline':original,'fused':combined,'raw_top100':ranks,'timing':timing,'protocol_sha256':PIN,'paid_calls':0})
 labels=qrels('test');labels={q:labels[q] for q in ids}
 dense=json.loads((ROOT/'results/public-full-skillret_embed_06.json').read_text())['rankings'];dense={q:dense[q][:20] for q in ids}
 oracle=lambda r:{q:sorted(pool,key=lambda d:(-labels[q].get(d,0),pool.index(d))) for q,pool in r.items()}
 metrics={k:evaluate(labels,v) for k,v in {'BM25 original':original,'Original+request view':combined,'SkillRet reference':dense,'BM25 pool oracle':oracle(original),'Request-view pool oracle':oracle(combined)}.items()}
 per=[]
 for q in ids:
  b=metrics['BM25 original']['per_query'][q];v=metrics['Original+request view']['per_query'][q]
  per.append({'id':q,'baseline_recall20':b['Recall@20'],'view_recall20':v['Recall@20'],'delta_recall20':v['Recall@20']-b['Recall@20'],'baseline_ndcg10':b['NDCG@10'],'view_ndcg10':v['NDCG@10']})
 result={'metrics':metrics,'per_query':per,'protocol_sha256':PIN,'baseline_reproduction_exact8':True,'views_changed':sum(q['query']!=q['request_view'] for q in p['queries']),'paid_calls':0,'caveat':'One frozen8-query public historically exposed panel, not blind data. Changed candidate coverage is intentional; separate oracle and actual ranking metrics. No generated rewriting, GPU, sweep, new embedding or paid calls.'}
 save(out/'evaluation.json',result);print(json.dumps({'aggregate':{k:{m:v['aggregate'][m] for m in ('Recall@20','NDCG@10','Completeness@20')} for k,v in metrics.items()},'per_query':per,'baseline_reproduction_exact8':True,'paid_calls':0},indent=2))
if __name__=='__main__':main()
