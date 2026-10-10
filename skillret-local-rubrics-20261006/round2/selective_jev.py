"""Frozen selective-escalation probe; no labels imported until separate scoring."""
import argparse,fcntl,hashlib,json,math,signal,time,urllib.error,urllib.request
from decimal import Decimal
from jev_probe import HERE,ROOT,save,post
from jev_auth_check import NoRedirect,check_key
from secret_store import load_key
from probe_contract import build_request,digest,MODEL
PIN='429b77b7c086b7bfc9166fe5be0855d00950d68aad421494f10bce3784abeef6'
RESERVE=Decimal('0.05');RATE=Decimal('0.042');RUN='selective-jev-8-v1'
def prepare():
 p=json.loads((HERE/'selective-probe-plan.json').read_text());assert p.pop('protocol_sha256')==PIN and digest(p)==PIN
 assert len(p['rows'])==8 and sum(r['escalate'] for r in p['rows'])==4
 needed={d for r in p['rows'] if r['escalate'] for d in r['candidate_ids']};headers={};h=hashlib.sha256()
 with (ROOT/'data/skills/test.jsonl').open('rb') as f:
  for line in f:
   h.update(line);s=json.loads(line)
   if s['id'] in needed:headers[s['id']]={k:s[k] for k in ('id','name','description')}
 assert h.hexdigest()=='1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81'
 items=[]
 for row in p['rows']:
  packet=None
  if row['escalate']:
   packet=build_request({'id':row['id'],'query':row['query']},[headers[d] for d in row['candidate_ids']])
   packet['questions']={k:v for k,v in packet['questions'].items() if k.startswith('s')}
  items.append({'id':row['id'],'candidate_ids':row['candidate_ids'],'escalate':row['escalate'],'gate_triggered':row['gate_triggered'],'quota_excluded':row['quota_excluded'],'request':packet})
 return items
def fusion(pool,jev):
 if len(pool)!=20 or set(pool)!=set(jev) or len(jev)!=20:raise ValueError('Pool mismatch')
 scores={d:1/(61+i) for i,d in enumerate(pool)}
 for i,d in enumerate(jev):scores[d]+=1/(61+i)
 return sorted(pool,key=lambda d:(-scores[d],pool.index(d)))
def parse(response,pool):
 if response.get('model')!=MODEL:raise ValueError('Wrong model')
 tokens=response['usage']['input_tokens']
 if isinstance(tokens,bool) or not isinstance(tokens,int) or not 0<=tokens<=64000:raise ValueError('Invalid usage')
 scores=[]
 for i in range(20):
  a=response['answers'][f's{i}'];v=a['score']
  if a['type']!='score' or isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=3:raise ValueError('Invalid score')
  scores.append(v)
 jev=[pool[i] for i in sorted(range(20),key=lambda i:(-scores[i],i))]
 return jev,scores,tokens
def run_batch(items,key,out,opener):
 if len(items)!=8 or sum(r['escalate'] for r in items)>4:raise ValueError('Call cap changed')
 rows=[];attempts=0;tokens=0;halt=False
 for item in items:
  pool=item['candidate_ids'];row={'id':item['id'],'candidate_ids':pool,'ranking':pool,'no_failure_fallback_ranking':pool,'escalate':item['escalate'],'status':'dense_no_call'}
  if item['escalate']:
   if halt:row.update(status='not_attempted_after_failure',no_failure_fallback_ranking=[])
   else:
    attempts+=1;row.update(status='reserved_before_transmission',attempt=attempts,request_sha256=digest(item['request']))
    save(out/'progress.json',{'rows':rows+[row],'attempts':attempts,'reserved_usd':str(RESERVE)})
    t=time.monotonic()
    try:
     response=post(key,item['request'],opener);jev,scores,n=parse(response,pool);tokens+=n
     rank=fusion(pool,jev);row.update(status='ok',ranking=rank,no_failure_fallback_ranking=rank,jev_ranking=jev,scores=scores,input_tokens=n,model=MODEL)
    except BaseException as e:
     if isinstance(e,urllib.error.HTTPError):row['http_status']=e.code;e.close()
     row.update(status='failed_or_uncertain',error_type=type(e).__name__,no_failure_fallback_ranking=[]);halt=True
    row['api_seconds']=time.monotonic()-t
  rows.append(row)
 result={'rows':rows,'attempts':attempts,'known_input_tokens':tokens,'known_usage_usd':str(Decimal(tokens)*RATE/1000000),'protocol_sha256':PIN,'reserved_usd':str(RESERVE),'reserve_released':False,'no_retries':True,'labels_read':False}
 save(out/'results.json',result);return result
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();items=prepare()
 if args.prepare_only:
  save(HERE/'prepared-selective-requests.json',items);print(json.dumps({'queries':8,'calls':4,'max_input_tokens':256000,'usage_bound_usd':'0.010752','reserve_usd':'0.05','payload_bytes':[len(json.dumps(i['request']).encode()) for i in items if i['escalate']],'network_calls':0}));return 0
 if digest(items)!=digest(json.loads((HERE/'prepared-selective-requests.json').read_text())):raise ValueError('Prepared requests changed')
 with (HERE/'budget.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out=HERE/RUN
  if out.exists():raise ValueError('No one-shot reruns')
  lp=HERE/'budget-ledger.json';ledger=json.loads(lp.read_text());committed=Decimal(ledger['committed_usd']);external=Decimal(ledger['external_spend_usd'])
  if Decimal(ledger['cap_usd'])!=4 or committed!=Decimal('0.60') or external!=1 or committed+external+RESERVE>4:raise ValueError('Unexpected budget state')
  key=load_key('jev')
  try:
   auth=check_key(key)
   if not auth.get('authenticated'):print(json.dumps(auth));return 2
   out.mkdir();ledger['committed_usd']=str(committed+RESERVE);ledger['runs'][RUN]={'reserved_usd':str(RESERVE),'status':'reserved','protocol_sha256':PIN};save(lp,ledger)
   save(out/'disclosure.json',{'endpoint':'https://api.typesafe.ai/v1/systemone','model':MODEL,'max_calls':4,'max_input_tokens_total':256000,'usage_bound_usd':'0.010752','reserved_all_in_usd':str(RESERVE),'published_rate_usd_per_million':'0.042','output_price':0,'price_source':'https://docs.typesafe.ai/models','price_verified_utc_date':'2026-10-06','authorization':'Explicit delegated approval of ONE frozen selective probe, four calls, additional USD0.05 within totalUSD4. Public queries/name/description only.','requests_sha256':digest(items)})
   def deadline(signum,frame):raise TimeoutError('Deadline')
   signal.signal(signal.SIGALRM,deadline);signal.alarm(120)
   try:result=run_batch(items,key,out,urllib.request.build_opener(NoRedirect).open)
   finally:signal.alarm(0)
   ledger['runs'][RUN]['status']='finished_or_stopped';save(lp,ledger)
   print(json.dumps({'attempts':result['attempts'],'known_usage_usd':result['known_usage_usd'],'statuses':[r['status'] for r in result['rows']],'committed_total_usd':'1.65'}))
  finally:del key
 return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except (Exception,KeyboardInterrupt):print('Stopped; no sensitive output. Existing reservation remains. No retries.');raise SystemExit(1)
