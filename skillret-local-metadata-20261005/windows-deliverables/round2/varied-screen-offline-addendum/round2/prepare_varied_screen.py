"""Final user-steered v3; previous plans preserved, no earlier live stage calls."""
import itertools,json,hashlib
from pathlib import Path
from varied_prompts import request,SPECS,VARIANTS
from probe_contract import digest
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2'
def main():
 prior=json.loads((HERE/'prompt-engineering-plan.json').read_text());dev=prior['dev_queries'];confirm=prior['confirmation_queries']
 bm=json.loads((ROOT/'results/public-full-bm25.json').read_text())['rankings'];qs=dev+confirm;pools={q['id']:bm[q['id']][:20] for q in qs};needed={d for p in pools.values() for d in p};headers={};h=hashlib.sha256()
 with (ROOT/'data/skills/test.jsonl').open('rb') as f:
  for line in f:
   h.update(line);s=json.loads(line)
   if s['id'] in needed:headers[s['id']]={k:s[k] for k in ('id','name','description')}
 assert h.hexdigest()=='1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81'
 def item(q,variants,depth,tag):
  pool=pools[q['id']][:depth]
  return {'id':q['id'],'depth':depth,'tag':tag,'variants':variants,'pool':pool,'request':request(q,[headers[d] for d in pool],variants)}
 groups=[(['original','action'],20,'original_action'),(['anchored','role_choice'],20,'anchored_role'),(['constraint'],20,'constraint'),(['original'],10,'depth10')]
 dev_packets=[item(q,v,d,t) for q in dev for v,d,t in groups]
 candidate_variants=VARIANTS[1:];confirm_packets={}
 for n in (1,):
  for combo in itertools.combinations(candidate_variants,n):confirm_packets['+'.join(combo)]=[item(q,['original',*combo],20,'confirmation') for q in confirm]
 packets={'dev':dev_packets,'confirmation':confirm_packets}
 p={'version':'prompt-engineering-v3-factorized-bm25','supersedes':'User expanded diversity before anyv1/v2 inference. v1 retained; v2 failed offline payload validation; onlyv3 executes.','specifications':SPECS,'variants':VARIANTS,'original':'Exact previous four-level Score rubric retained as control.','score_mapping':{'score':'0..3 weighted level index','noul':'P(true)*3 for common display only, ranking monotonic','choice':'Expected ordinal role index with irrelevant0,topical1,support2,direct3; normalize probability sum; no learned weights.'},'dev_queries':dev,'confirmation_queries':confirm,'pools':pools,'factorization':'Five types/rubrics on BM25top20. Separate depth10 original-rubric arm; no mixing depth change with rubric effect. Same candidates for each depth20 comparison. Direct rerank and equal-weight RRF60 reported separately.','primary':'True macro Recall@5 with full positive-qrels denominator; direct Jev ordering.','secondary':['Recall@10','NDCG@10','Recall@20 pool ceiling','oracle Recall5/10 and NDCG10','Completeness separately','fusion metrics','failures','latency','metered usage'], 'promotion':'All24dev calls must be valid. Among action,anchored,role_choice,constraint at depth20, select at most1 with mean Recall5 delta>=0.05 versus original direct Jev, at least2 query wins, and wins>losses. Rank by delta, then fixed variant order. No prompt editing. Depth factor excluded from selection.','confirmation':'Only if selected: one separate6-query panel with original and selected variants in same call/pool. Primary Recall5; report all selected variants, secondary metrics and multiple-testing selection. No further rounds.','max_dev_calls':24,'max_confirmation_calls':6,'max_total_calls':30,'max_input_tokens':1920000,'usage_bound_usd':'0.08064','reserve_all_in_usd':'0.10','model':'jev-1.13.0','rate_input_per_million':'0.042','output_free':True,'requests_sha256':digest(packets),'exposure':'Dev first6 reused initial public cases; confirmation6 disjoint earlier rerank cohorts, but public data historically exposed. Not genuine blind test. Types, objectives and candidate-depth effects are separate; screen results subject to selection/multiple testing.','credits':'Existing user-reported free credit only, USD5 stated by user; not independently verified. No topup. Metered usage != cash charge, USD4 cap unchanged.'}
 p['protocol_sha256']=digest(p)
 for name,value in [('varied-screen-plan.json',p),('prepared-varied-screen.json',packets)]:
  path=HERE/name
  if path.exists():assert json.loads(path.read_text())==value
  else:path.write_text(json.dumps(value,indent=2))
 print(json.dumps({'protocol_sha256':p['protocol_sha256'],'dev_calls':24,'conditional_confirmation_calls':6,'max_calls':30,'usage_bound_usd':'0.08064','reserved_usd':'0.10','max_payload_bytes':max(len(json.dumps(x['request']).encode()) for x in dev_packets+[x for xs in confirm_packets.values() for x in xs]),'network_calls':0}))
if __name__=='__main__':main()

