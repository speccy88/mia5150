"""Freeze two rubrics, development cohort, conditional confirmation and gate."""
import hashlib,json
from pathlib import Path
from prompt_rubrics import RUBRICS,request
from probe_contract import digest
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2'
def main():
 old=json.loads((HERE/'exploration-freeze.json').read_text());dev=old['queries'][:6]
 excluded={q['id'] for q in json.loads((ROOT/'manifests/scorer-queries.json').read_text())}
 for name in ['next-confirmation-8.json','fusion-confirmation-plan.json','request-clause-protocol.json']:
  excluded.update(q['id'] for q in json.loads((HERE/name).read_text())['queries'])
 qs=[{'id':q['id'],'query':q['query']} for q in map(json.loads,(ROOT/'data/queries/test.jsonl').read_text().splitlines()) if q['id'] not in excluded]
 confirm=sorted(qs,key=lambda q:hashlib.sha256(('mia5150-prompt-confirmation-v1:'+q['id']).encode()).hexdigest())[:6]
 dense=json.loads((ROOT/'results/public-full-skillret_embed_06.json').read_text())['rankings'];bm=json.loads((ROOT/'results/public-full-bm25.json').read_text())['rankings']
 pools={q['id']:old['candidate_ids'][q['id']] for q in dev};pools.update({q['id']:dense[q['id']][:20] for q in confirm})
 bm_pools={q['id']:bm[q['id']][:20] for q in dev}
 needed={d for p in list(pools.values())+list(bm_pools.values()) for d in p};headers={};h=hashlib.sha256()
 with (ROOT/'data/skills/test.jsonl').open('rb') as f:
  for line in f:
   h.update(line);s=json.loads(line)
   if s['id'] in needed:headers[s['id']]={k:s[k] for k in ('id','name','description')}
 assert h.hexdigest()=='1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81'
 packets={'dev':[{'id':q['id'],'pool':pools[q['id']],'request':request(q,[headers[d] for d in pools[q['id']]],['role','constraint'])} for q in dev],'confirmation':{v:[{'id':q['id'],'pool':pools[q['id']],'request':request(q,[headers[d] for d in pools[q['id']]],['original',v])} for q in confirm] for v in RUBRICS}}
 for item,q in zip(packets['dev'],dev):
  lexical=request(q,[headers[d] for d in bm_pools[q['id']]],['original'])
  item['bm25_pool']=bm_pools[q['id']]
  item['request']['questions'].update({k.replace('original_','bm25_original_'):v for k,v in lexical['questions'].items()})
  assert len(json.dumps(item['request'],ensure_ascii=False).encode())<=60000
 p={'version':'prompt-engineering-v1','rubrics':RUBRICS,'type':'score','score_mapping':'Probability-weighted ordered criterion index0..3; original dense order breaks score ties; unchanged equal-weight RRF60.','dev_queries':dev,'confirmation_queries':confirm,'candidate_ids':pools,'dev_baseline':'Cached original Jev Score + RRF60 on the same first6 initial development queries.','confirmation_baseline':'Fresh original rubric and selected new rubric evaluated in one40-question call per query, same state and candidates.','selection':'Confirmation6 selected with saltedSHA256 mia5150-prompt-confirmation-v1:, excluding128 scorer and prior8+8+8 panels; no labels used.','promising_gate':'All6 dev calls valid; delta mean NDCG@10>=0.03 versus cached original-rubric RRF60, at least2 query wins and more wins than losses. Choose highest mean delta; exact tie prefers role. Only one confirmation panel and one selected variant. No gate changes or prompt editing afterward.','confirmation_interpretation':'Primary paired NDCG@10 versus fresh original-rubric RRF60, denominator6; directional support only if delta>0 and wins>losses. Report uncertainty, all negatives and failures.','max_calls':12,'reserve_usd':'0.10','max_input_tokens':768000,'usage_bound_usd':'0.032256','requests_sha256':digest(packets),'model':'jev-1.13.0','exposure':'Development reused deliberately to isolate rubric effects; separate confirmation cohort remains historically exposed public benchmark, not genuine blind test. Dev cached-control differences may include temporal/order nondeterminism; confirmation has paired same-call control.'}
 p['version']='prompt-engineering-v2-user-recall-bm25-amendment'
 p['amendment']='User added trueRecall5/10 and BM25top20-to-Jev arm BEFORE any inference. Originalv1 plan is preserved; thisv2 supersedes it.'
 p['metrics']=['True macro Recall@5','True macro Recall@10','NDCG@10','Pool Recall@20','Pool-oracle Recall@5/10 and NDCG@10','Completeness reported separately; never substituted for recall']
 p['bm25_candidate_ids']=bm_pools
 p['bm25_arm']='Same6 development queries, BM25top20 candidates, unchanged original operational Score rubric. Report BM25 baseline, direct Jev ranking and fixed RRF60 separately. Different pool is a separate retrieval arm, not a prompt-only effect.'
 p['packing']='Each development call has60 Score questions:20 role+20 constraint on dense pool and20 original on BM25 pool. Confirmation remains40 questions with fresh original+chosen variant on the dense pool. No additional calls solely for BM25 or Recall metrics. Max12 calls and768k input tokens; fullrequest<=60k UTF8 bytes.'
 p['protocol_sha256']=digest(p)
 for name,value in [('prompt-engineering-plan-v2.json',p),('prepared-prompt-engineering-v2.json',packets)]:
  path=HERE/name
  if path.exists():assert json.loads(path.read_text())==value
  else:path.write_text(json.dumps(value,indent=2))
 print(json.dumps({'protocol_sha256':p['protocol_sha256'],'dev_ids':[q['id'] for q in dev],'confirmation_ids':[q['id'] for q in confirm],'max_dev_payload_bytes':max(len(json.dumps(x['request']).encode()) for x in packets['dev']),'max_calls':12,'reserve_usd':'0.10','network_calls':0}))
if __name__=='__main__':main()
