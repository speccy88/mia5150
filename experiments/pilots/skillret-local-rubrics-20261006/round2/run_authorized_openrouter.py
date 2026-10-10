"""Execute the expressly authorized fixed round; no interactive approval repetition."""
import json,fcntl,signal,urllib.request
from decimal import Decimal
from openrouter_probe import HERE,RUN,RESERVE,PIN,packets,run_batch,quote
from openrouter_preflight import credits
from secret_store import load_key
from jev_auth_check import NoRedirect
from jev_probe import save
from probe_contract import digest
def main():
 with (HERE/'budget.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  out=HERE/RUN
  if out.exists():raise ValueError('Existing one-shot artifacts; no repeat')
  items=packets();cached=json.loads((HERE/'prepared-openrouter-requests.json').read_text())
  if digest(items)!=digest(cached):raise ValueError('Prepared requests changed')
  lp=HERE/'budget-ledger.json';ledger=json.loads(lp.read_text())
  committed=Decimal(ledger['committed_usd']);external=Decimal(ledger['external_spend_usd'])
  if Decimal(ledger['cap_usd'])!=4 or committed!=Decimal('0.10') or external!=1 or committed+external+RESERVE>4:raise ValueError('Unexpected ledger; reassess before spending')
  opener=urllib.request.build_opener(NoRedirect).open;pricing=quote(opener);key=load_key('openrouter')
  try:
   balance=credits(key,opener)
   if not balance['balance_verified'] or not balance['covers_reservation']:print(json.dumps(balance));return 2
   out.mkdir();ledger['committed_usd']=str(committed+RESERVE);ledger['runs'][RUN]={'reserved_usd':str(RESERVE),'status':'reserved','protocol_sha256':PIN};save(lp,ledger)
   save(out/'disclosure.json',{'pricing':pricing,'credit_verified':True,'approved_public_fields':['query','candidate.name','candidate.description'],'key_saved_by_probe':False,'max_calls':12,'reserve_usd':str(RESERVE),'authorization':'User approved OpenRouter experiments under shared USD4 cap; delegated continuation approved fixed12 query probe with saved key and USD0.50 reservation. No additional topup.','requests_sha256':digest(items)})
   def deadline(signum,frame):raise TimeoutError('Deadline')
   signal.signal(signal.SIGALRM,deadline);signal.alarm(660)
   try:result=run_batch(key,items,out,opener)
   finally:signal.alarm(0)
   ledger['runs'][RUN]['status']='finished_or_stopped';save(lp,ledger)
   print(json.dumps({'attempts':result['attempts'],'known_usage_usd':result['known_reported_cost_usd'],'reserved_usd':str(RESERVE),'committed_total_usd':str(committed+external+RESERVE),'statuses':[r['status'] for r in result['rows']]}))
  finally:del key
 return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except (Exception,KeyboardInterrupt):print('Stopped with no sensitive output. Any existing reservation remains; no automatic retries.');raise SystemExit(1)
