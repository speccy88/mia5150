"""Fixed equal-budget retrieval-pool screen. Preparation never reads labels or keys."""
import argparse,fcntl,hashlib,json,signal,sys,time,urllib.request,urllib.error
from decimal import Decimal
from pathlib import Path
from jev_probe import HERE,ROOT,save,post
from jev_auth_check import NoRedirect
from probe_contract import digest
from varied_prompts import request,parse
RUN='hybrid-pool-8-v1';ARMS=['bm25','vector','hybrid'];RATE=Decimal('.042')

def hybrid(bm,vector):
 result=[]
 for i in range(max(len(bm),len(vector))):
  for ranking in (bm,vector):
   if i<len(ranking) and ranking[i] not in result:result.append(ranking[i])
   if len(result)==20:return result
 raise ValueError('Candidate shortfall')

def cached_requests():
 cache={};uncertain=set()
 for path in sorted(HERE.glob('*/results.json')):
  if path.parent.name==RUN:continue
  obj=json.loads(path.read_text())
  for row in obj.get('rows',[]):
   h=row.get('request_sha256')
   if not h:continue
   if row.get('status')=='ok':cache.setdefault(h,{'source':str(path.relative_to(ROOT)),'row':row})
   else:uncertain.add(h)
 return cache,uncertain

def prepare():
 excluded={q['id'] for q in json.loads((ROOT/'manifests/scorer-queries.json').read_text())}
 for name,key in [('next-confirmation-8.json','queries'),('fusion-confirmation-plan.json','queries'),('request-clause-protocol.json','queries'),('varied-screen-plan.json','confirmation_queries')]:
  excluded.update(q['id'] for q in json.loads((HERE/name).read_text())[key])
 queries=[{'id':q['id'],'query':q['query']} for q in map(json.loads,(ROOT/'data/queries/test.jsonl').read_text().splitlines()) if q['id'] not in excluded]
 queries=sorted(queries,key=lambda q:hashlib.sha256(('mia5150-equal-hybrid-pool-v1:'+q['id']).encode()).hexdigest())[:8]
 files=[ROOT/'results/public-full-bm25.json',ROOT/'results/public-full-skillret_embed_06.json'];bm,vector=[json.loads(p.read_text())['rankings'] for p in files]
 pools={q['id']:{'bm25':bm[q['id']][:20],'vector':vector[q['id']][:20],'hybrid':hybrid(bm[q['id']][:100],vector[q['id']][:100])} for q in queries}
 assert all(len(v)==len(set(v))==20 for p in pools.values() for v in p.values())
 needed={d for p in pools.values() for v in p.values() for d in v};headers={};h=hashlib.sha256()
 with (ROOT/'data/skills/test.jsonl').open('rb') as f:
  for line in f:
   h.update(line);s=json.loads(line)
   if s['id'] in needed:headers[s['id']]={k:s[k] for k in ('id','name','description')}
 assert h.hexdigest()=='1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81'
 items=[]
 for i,q in enumerate(queries):
  for arm in ARMS[i%3:]+ARMS[:i%3]:
   pool=pools[q['id']][arm];items.append({'id':q['id'],'arm':arm,'pool':pool,'request':request(q,[headers[d] for d in pool],['original'])})
 cache,uncertain=cached_requests();reuse=[]
 for x in items:
  h=digest(x['request'])
  if h in uncertain and h not in cache:raise ValueError('Uncertain identical prior request; no blind retry')
  if h in cache:reuse.append({'id':x['id'],'arm':x['arm'],'request_sha256':h,'source':cache[h]['source']})
 plan={'version':RUN,'queries':queries,'query_selection':'First8 by saltedSHA256 mia5150-equal-hybrid-pool-v1:, excluding scorer128, earlier selective/fusion/clause cohorts and untouched prompt-confirmation6. No label access during selection or pool construction; historically exposed public data, not a blind test.','candidate_budget':20,'pools':pools,'construction':'BM25 top20; vector SkillRet-Embedding0.6B top20; hybrid stable round-robin BM25 then vector from cached top100, deduplicate by skill ID, stop at exactly20. Unreranked hybrid control uses that same round-robin order. No outcome-dependent fill or tuning.','retrieval_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'corpus_sha256':'1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81','rubric':'Exact original four-level Score as in prior screen; all arms identical rubric and full public name/description evidence. Returned score descending, original candidate order ties.','model':'jev-1.13.0','call_order':'Rotate bm25,vector,hybrid across query index to reduce fixed temporal ordering bias; one20-question request per arm/query.','primary':['true macro Recall@5','true macro Recall@10'],'secondary':['NDCG@10','pool recall20','oracle Recall5/10 and NDCG10','per-query gains/ties/losses','completeness separately','batch latency and token costs'],'interpretation':'Hybrid candidates are combined BEFORE Jev; this is not the earlier vector/Jev rank-fusion pipeline. Direct Jev is final ranking; no additional fusion. Include all8 queries per arm; invalid/unattempted rerank is empty and zero metrics, no fallback substitution. No adaptive follow-up.','max_calls':24,'max_input_tokens':1536000,'metered_usage_bound_usd':'0.064512','reserve_usd':'0.10','shared_cap_usd':'4','commitments_before_usd':'1.87','commitments_after_reservation_usd':'1.97','requests_sha256':digest(items),'identical_cache_matches':reuse,'endpoint':'https://api.typesafe.ai/v1/systemone','outgoing_fields':['public canonical task','candidate name','candidate description'],'no_topup':True,'no_downloads_or_gpu':True}
 plan['protocol_sha256']=digest(plan)
 for name,value in [('hybrid-pool-plan.json',plan),('prepared-hybrid-pool.json',items)]:
  path=HERE/name
  if path.exists():assert json.loads(path.read_text())==value
  else:save(path,value)
 return plan,items

def main():
 arg=argparse.ArgumentParser();arg.add_argument('--prepare-only',action='store_true');args=arg.parse_args();plan,items=prepare()
 if args.prepare_only:print(json.dumps({'protocol_sha256':plan['protocol_sha256'],'queries':[q['id'] for q in plan['queries']],'max_calls':24,'cached_matches':len(plan['identical_cache_matches']),'typed_questions':480,'usage_bound_usd':'0.064512','reserve_usd':'0.10','labels_read':False}));return 0
 from secret_store import load_key
 with (HERE/'budget.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lp=HERE/'budget-ledger.json';ledger=json.loads(lp.read_text());out=HERE/RUN
  if out.exists():raise ValueError('One-shot run already exists')
  if Decimal(ledger['cap_usd'])!=4 or Decimal(ledger['committed_usd'])!=Decimal('.87') or Decimal(ledger['external_spend_usd'])!=1:raise ValueError('Unexpected ledger')
  cache,uncertain=cached_requests();key=load_key('jev');out.mkdir();ledger['committed_usd']='0.97';ledger['runs'][RUN]={'reserved_usd':'0.10','status':'reserved','protocol_sha256':plan['protocol_sha256']};ledger['paid_tests_paused']=True;save(lp,ledger)
  save(out/'disclosure.json',{'endpoint':plan['endpoint'],'outgoing_fields':plan['outgoing_fields'],'authorization':'New user-requested BM25+vector candidates then Jev, fixed24calls/0.10reservation within shared4 cap. No topup.','max_calls':24,'reserve_usd':'0.10','usage_bound_usd':'0.064512'})
  rows=[];attempts=0;tokens=0;halt=False
  def deadline(signum,frame):raise TimeoutError('Bounded screen deadline')
  signal.signal(signal.SIGALRM,deadline);signal.alarm(600)
  try:
   for item in items:
    h=digest(item['request']);row={k:item[k] for k in ['id','arm','pool']};row.update(request_sha256=h,status='not_attempted',ranking=[])
    if not halt:
     if h in cache:
      old=cache[h]['row']
      if 'outputs' not in old or 'original' not in old['outputs']:raise ValueError('Matching cache needs supported mapping; no duplicate call')
      row.update(status='ok',ranking=old['outputs']['original']['ranking'],scores=old['outputs']['original']['scores'],cache_source=cache[h]['source'],new_call=False,input_tokens=0,api_seconds=0)
     else:
      if h in uncertain or attempts>=24:raise ValueError('No retry/call cap')
      attempts+=1;row.update(status='reserved_before_transmission',attempt=attempts,new_call=True);save(out/'progress.json',{'rows':rows+[row],'attempts':attempts});start=time.monotonic()
      try:
       response=post(key,item['request'],urllib.request.build_opener(NoRedirect).open);outputs,n=parse(response,item['pool'],['original']);tokens+=n
       if tokens>1536000:raise ValueError('Token cap')
       row.update(status='ok',ranking=outputs['original']['ranking'],scores=outputs['original']['scores'],input_tokens=n,model=response['model'])
       cache[h]={'source':str((out/'results.json').relative_to(ROOT)),'row':{'outputs':outputs}}
      except BaseException as exc:
       if isinstance(exc,urllib.error.HTTPError):row['http_status']=exc.code;exc.close()
       row.update(status='failed_or_uncertain',error_type=type(exc).__name__);halt=True
      row['api_seconds']=time.monotonic()-start
    rows.append(row);save(out/'progress.json',{'rows':rows,'attempts':attempts})
  finally:
   signal.alarm(0);del key
   result={'protocol_sha256':plan['protocol_sha256'],'rows':rows,'attempts':attempts,'known_input_tokens':tokens,'known_usage_usd':str(Decimal(tokens)*RATE/1000000),'reserved_usd':'0.10','reserve_released':False,'cash_charge_usd':None,'shared_commitments_usd':'1.97'};save(out/'results.json',result)
   ledger['runs'][RUN]['status']='finished_or_stopped';ledger['paid_tests_paused']=True;ledger['pause_reason']='Authorized fixed hybrid-pool screen completed/stopped; other paid work remains paused.';save(lp,ledger)
  print(json.dumps({k:v for k,v in result.items() if k!='rows'}));return int(halt)
if __name__=='__main__':
 try:raise SystemExit(main())
 except (Exception,KeyboardInterrupt):print('Stopped safely; no sensitive error output, retries or refunds.');raise SystemExit(1)
