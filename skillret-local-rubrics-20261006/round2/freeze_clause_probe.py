"""Freeze a distinct lexical query-view experiment; no evaluation or calls."""
import hashlib,json
from pathlib import Path
from request_clause_view import request_view
from probe_contract import digest
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2'
def main():
 excluded={q['id'] for q in json.loads((ROOT/'manifests/scorer-queries.json').read_text())}
 for name in ['next-confirmation-8.json','fusion-confirmation-plan.json']:
  excluded.update(q['id'] for q in json.loads((HERE/name).read_text())['queries'])
 queries=[{'id':q['id'],'query':q['query']} for q in map(json.loads,(ROOT/'data/queries/test.jsonl').read_text().splitlines()) if q['id'] not in excluded]
 selected=sorted(queries,key=lambda q:hashlib.sha256(('mia5150-request-clause-v1:'+q['id']).encode()).hexdigest())[:8]
 rows=[{**q,'request_view':request_view(q['query'])} for q in selected]
 p={'queries':rows,'hypothesis':'A deterministic request-clause lexical view fused with original-query BM25 using fixed RRF60 increases candidate Recall@20 versus original-query BM25.','primary_metric':'Macro Recall@20 all8','secondary':['NDCG@10','pool oracle NDCG@10','Completeness@20','coverage relative to cached SkillRet'],'retrieval_contract':'Same6006 corpus, header name+description, harness.lexical/bm25s English stopwords, top100, one CPU thread. First reproduce original-query BM25 rank lists on8; mismatch blocks claims. Fixed equal-weight original/clause RRF60, skill-ID tie break, top20.','selection':'Salted SHA256 query IDs, salt mia5150-request-clause-v1:, excluding previous128 scorer plus both recent8 panels; no label-based selection.','rule_sha256':hashlib.sha256((HERE/'request_clause_view.py').read_bytes()).hexdigest(),'calls_max':0,'training':False,'gpu':False,'parameter_sweep':False,'status':'Prepared only; no retrieval or labels evaluated for this protocol yet.','exposure':'Historically exposed public benchmark. New experiment cohort is disjoint from earlier rerank panels, not blind final data.','resource_requirement':'Check host/guest availableRAM>=2GiB and no active paging immediately before lexical run; stop on pressure. No GPU or external API.'}
 p['protocol_sha256']=digest(p);path=HERE/'request-clause-protocol.json'
 if path.exists():assert json.loads(path.read_text())==p
 else:path.write_text(json.dumps(p,indent=2))
 print(json.dumps({'ids':[r['id'] for r in rows],'views_changed':sum(r['query']!=r['request_view'] for r in rows),'protocol_sha256':p['protocol_sha256'],'network_calls':0,'labels_read':False}))
if __name__=='__main__':main()
