"""One explicitly authorized, frozen8-query fixed-RRF confirmation probe."""
import argparse,fcntl,hashlib,json,signal,time,urllib.error,urllib.request
from decimal import Decimal
from jev_probe import HERE,ROOT,save,post
from jev_auth_check import NoRedirect,check_key
from secret_store import load_key
from selective_jev import parse,fusion
from probe_contract import build_request,digest,MODEL
PIN='e8133175358110f7e96a8cbd52dc6c83378f844d6e83c5f1248641b206e1c261'
RUN='fusion-confirmation-8-v1';RESERVE=Decimal('0.10');RATE=Decimal('0.042')
def prepare():
 p=json.loads((HERE/'fusion-confirmation-plan.json').read_text());assert p.pop('protocol_sha256')==PIN and digest(p)==PIN and len(p['queries'])==8
 needed={d for pool in p['candidate_ids'].values() for d in pool};headers={};h=hashlib.sha256()
 with (ROOT/'data/skills/test.jsonl').open('rb') as f:
  for line in f:
   h.update(line);s=json.loads(line)
   if s['id'] in needed:headers[s['id']]={k:s[k] for k in ('id','name','description')}
 assert h.hexdigest()=='1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81'
 items=[]
 for q in p['queries']:
  pool=p['candidate_ids'][q['id']];packet=build_request(q,[headers[d] for d in pool]);packet['questions']={k:v for k,v in packet['questions'].items() if k.startswith('s')}
  items.append({'id':q['id'],'candidate_ids':pool,'request':packet})
 return items
def run_batch(items,key,out,opener):
 if len(items)!=8 or len({x['id'] for x in items})!=8:raise ValueError('Exactly8 distinct queries required')
 rows=[];attempts=0;tokens=0;halt=False
 for item in items:
  pool=item['candidate_ids'];row={'id':item['id'],'candidate_ids':pool,'ranking':[],'status':'not_attempted'}
  if not halt:
   attempts+=1;row.update(status='reserved_before_transmission',attempt=attempts,request_sha256=digest(item['request']));save(out/'progress.json',{'rows':rows+[row],'attempts':attempts,'reserved_usd':str(RESERVE)})
   t=time.monotonic()
   try:
    response=post(key,item['request'],opener);jev,scores,n=parse(response,pool);tokens+=n
    row.update(status='ok',ranking=fusion(pool,jev),jev_ranking=jev,scores=scores,input_tokens=n,model=MODEL)
   except BaseException as e:
    if isinstance(e,urllib.error.HTTPError):row['http_status']=e.code;e.close()
    row.update(status='failed_or_uncertain',error_type=type(e).__name__);halt=True
   row['api_seconds']=time.monotonic()-t
  rows.append(row)
 result={'rows':rows,'attempts':attempts,'known_input_tokens':tokens,'known_usage_usd':str(Decimal(tokens)*RATE/1000000),'protocol_sha256':PIN,'reserved_usd':str(RESERVE),'reserve_released':False,'no_retries':True,'labels_read':False}
 save(out/'results.json',result);return result
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();items=prepare()
 if args.prepare_only:
  save(HERE/'prepared-fusion-confirmation.json',items);print(json.dumps({'queries':8,'max_calls':8,'input_tokens_bound':512000,'usage_bound_usd':'0.021504','reservation_usd':'0.10','payload_bytes':[len(json.dumps(x['request']).encode()) for x in items],'network_calls':0}));return 0
 if digest(items)!=digest(json.loads((HERE/'prepared-fusion-confirmation.json').read_text())):raise ValueError('Prepared requests changed')
 with (HERE/'budget.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out=HERE/RUN
  if out.exists():raise ValueError('No repeat')
  lp=HERE/'budget-ledger.json';ledger=json.loads(lp.read_text());committed=Decimal(ledger['committed_usd']);external=Decimal(ledger['external_spend_usd'])
  if Decimal(ledger['cap_usd'])!=4 or committed!=Decimal('0.65') or external!=1 or committed+external+RESERVE>4:raise ValueError('Unexpected budget')
  key=load_key('jev')
  try:
   auth=check_key(key)
   if not auth.get('authenticated'):print(json.dumps(auth));return 2
   out.mkdir();ledger['committed_usd']=str(committed+RESERVE);ledger['runs'][RUN]={'reserved_usd':str(RESERVE),'status':'reserved','protocol_sha256':PIN};save(lp,ledger)
   save(out/'disclosure.json',{'model':MODEL,'endpoint':'https://api.typesafe.ai/v1/systemone','approved_public_fields':['query','candidate.name','candidate.description'],'max_calls':8,'max_input_tokens':512000,'rate_usd_per_million':'0.042','price_source':'https://docs.typesafe.ai/models','price_verified_utc_date':'2026-10-06','usage_bound_usd':'0.021504','reserved_usd':str(RESERVE),'authorization':'User resumed bounded research 2026-10-06 01:01UTC, delegated ONE next small frozen probe with maxUSD0.10 additional. Supersedes previous review pause for this run only. No topup or other provider.','requests_sha256':digest(items)})
   def deadline(signum,frame):raise TimeoutError('Deadline')
   signal.signal(signal.SIGALRM,deadline);signal.alarm(240)
   try:r=run_batch(items,key,out,urllib.request.build_opener(NoRedirect).open)
   finally:signal.alarm(0)
   ledger['runs'][RUN]['status']='finished_or_stopped';save(lp,ledger)
   print(json.dumps({'attempts':r['attempts'],'known_usage_usd':r['known_usage_usd'],'statuses':[x['status'] for x in r['rows']],'committed_total_usd':'1.75'}))
  finally:del key
 return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except (Exception,KeyboardInterrupt):print('Stopped. No sensitive output; reservations retained, no retries.');raise SystemExit(1)
