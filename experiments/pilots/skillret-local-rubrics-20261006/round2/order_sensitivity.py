"""One frozen reversal on two cached cases. No search, labels or retries."""
import argparse,fcntl,json,signal,time,urllib.error,urllib.request
from decimal import Decimal
from jev_probe import HERE,save,post
from jev_auth_check import NoRedirect,check_key
from secret_store import load_key
from selective_jev import parse,fusion
from hybrid_retrieval import score_order
from probe_contract import digest,MODEL
RUN='order-sensitivity-2-v1';RESERVE=Decimal('0.02');RATE=Decimal('0.042')
def prepare():
 original=json.loads((HERE/'prepared-fusion-confirmation.json').read_text())
 assert len(original)==8 and [x['id'] for x in original[:2]]==['q-00091','q-03583']
 results=json.loads((HERE/'fusion-confirmation-8-v1/results.json').read_text());byid={r['id']:r for r in results['rows']}
 if results['protocol_sha256']!='e8133175358110f7e96a8cbd52dc6c83378f844d6e83c5f1248641b206e1c261':raise ValueError('Wrong source protocol')
 items=[]
 for source in original[:2]:
  old=byid[source['id']];assert old['status']=='ok' and old['request_sha256']==digest(source['request'])
  req=json.loads(json.dumps(source['request']));req['questions']={f's{i}':source['request']['questions'][f's{19-i}'] for i in range(20)}
  if len(json.dumps(req).encode())>60000:raise ValueError('Payload bound')
  items.append({'id':source['id'],'baseline':source['candidate_ids'],'permuted_ids':source['candidate_ids'][::-1],'request':req,'original_scores':old['scores'],'original_jev_ranking':old['jev_ranking'],'original_fusion':old['ranking']})
 protocol={'version':RUN,'selection':'First2 cases in already-frozen fusion-confirmation panel order, independent of gains/losses: q-00091 and q-03583.','permutation':'Reverse all20 candidate questions once. Task/rubric/model unchanged; map new scores back to skill IDs. Tie breaks remain ORIGINAL dense order, not permuted order.','ids':[r['id'] for r in items],'requests_sha256':digest(items),'model':MODEL,'max_calls':2,'max_input_tokens':128000,'usage_bound_usd':'0.005376','reserve_usd':'0.02','question':'Do mapped scores, Jev ranking and final RRF change under this one request-order perturbation?','caveat':'Operational robustness only. No quality labels. A changed answer could reflect temporal/backend nondeterminism as well as order; without a repeated original-order control we cannot isolate cause. Two cases do not establish general invariance.'}
 protocol['protocol_sha256']=digest(protocol);return items,protocol
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();items,protocol=prepare()
 if args.prepare_only:
  path=HERE/'order-sensitivity-plan.json'
  if path.exists():assert json.loads(path.read_text())==protocol
  else:save(path,protocol)
  save(HERE/'prepared-order-sensitivity.json',items);print(json.dumps(protocol,indent=2));return 0
 assert json.loads((HERE/'order-sensitivity-plan.json').read_text())==protocol and json.loads((HERE/'prepared-order-sensitivity.json').read_text())==items
 with (HERE/'budget.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out=HERE/RUN
  if out.exists():raise ValueError('No repeat')
  lp=HERE/'budget-ledger.json';ledger=json.loads(lp.read_text());committed=Decimal(ledger['committed_usd']);external=Decimal(ledger['external_spend_usd'])
  if Decimal(ledger['cap_usd'])!=4 or committed!=Decimal('0.75') or external!=1 or committed+external+RESERVE>4:raise ValueError('Unexpected ledger')
  key=load_key('jev');rows=[];attempts=0;tokens=0;halt=False
  try:
   auth=check_key(key)
   if not auth.get('authenticated'):print(json.dumps(auth));return 2
   out.mkdir();ledger['committed_usd']=str(committed+RESERVE);ledger['runs'][RUN]={'reserved_usd':str(RESERVE),'status':'reserved','protocol_sha256':protocol['protocol_sha256']};save(lp,ledger)
   save(out/'disclosure.json',{'protocol_sha256':protocol['protocol_sha256'],'approved_public_fields':['query','candidate.name','candidate.description'],'model':MODEL,'endpoint':'https://api.typesafe.ai/v1/systemone','authorization':'Explicit final overnight two-case one-permutation check, maxUSD0.02 additional, then stop new paid tests.','price_source':'https://docs.typesafe.ai/models','rate_per_million':'0.042','output_free':True,'price_verified_utc_date':'2026-10-06','usage_bound_usd':'0.005376','reserved_usd':str(RESERVE)})
   def deadline(signum,frame):raise TimeoutError('Deadline')
   signal.signal(signal.SIGALRM,deadline);signal.alarm(80)
   try:
    for item in items:
     row={'id':item['id'],'status':'not_attempted'}
     if not halt:
      attempts+=1;row.update(attempt=attempts,request_sha256=digest(item['request']),status='reserved_before_transmission');save(out/'progress.json',{'rows':rows+[row],'attempts':attempts});t=time.monotonic()
      try:
       response=post(key,item['request'],urllib.request.build_opener(NoRedirect).open);unused,scores,n=parse(response,item['permuted_ids']);tokens+=n
       byskill=dict(zip(item['permuted_ids'],scores));mapped=[byskill[d] for d in item['baseline']];rank=score_order(item['baseline'],mapped)
       row.update(status='ok',model=MODEL,input_tokens=n,mapped_scores=mapped,jev_ranking=rank,fusion=fusion(item['baseline'],rank))
      except BaseException as e:
       if isinstance(e,urllib.error.HTTPError):row['http_status']=e.code;e.close()
       row.update(status='failed_or_uncertain',error_type=type(e).__name__);halt=True
      row['api_seconds']=time.monotonic()-t
     rows.append(row)
   finally:signal.alarm(0)
   result={'rows':rows,'attempts':attempts,'known_input_tokens':tokens,'known_usage_usd':str(Decimal(tokens)*RATE/1000000),'reserved_usd':str(RESERVE),'reserve_released':False,'protocol_sha256':protocol['protocol_sha256'],'labels_read':False};save(out/'results.json',result)
   ledger['runs'][RUN]['status']='finished_or_stopped';save(lp,ledger)
   print(json.dumps({'attempts':attempts,'statuses':[r['status'] for r in rows],'known_usage_usd':result['known_usage_usd'],'committed_total_usd':'1.77'}))
  finally:del key
 return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except (Exception,KeyboardInterrupt):print('Stopped safely; no sensitive output, retries or refund.');raise SystemExit(1)
