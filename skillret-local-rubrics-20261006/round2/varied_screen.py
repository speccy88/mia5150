"""One authorized, frozen, bounded screen. No retries or automatic continuation."""
import fcntl,json,signal,statistics,sys,time,urllib.error,urllib.request
from decimal import Decimal
from pathlib import Path
from jev_probe import HERE,ROOT,save,post
from jev_auth_check import NoRedirect
from secret_store import load_key
from varied_prompts import parse,VARIANTS
from probe_contract import digest
sys.path.insert(0,str(ROOT))
from harness import qrels,evaluate
RUN='varied-screen-v3';PIN='d832c146580a51e466d8e887c6a9901979bbefce844c553f18205189822c407e'
RESERVE=Decimal('.10');RATE=Decimal('.042')

def analyze(rows,ids,pools,labels,variants,depths=False):
 labels={q:labels[q] for q in ids};arms={'BM25':{q:pools[q] for q in ids}}
 for v in variants:
  for suffix in ('direct','fusion'):
   arm={q:[] for q in ids}
   for row in rows:
    if row['id'] in arm and row['depth']==20 and v in row['variants'] and row['status']=='ok':arm[row['id']]=row['outputs'][v]['ranking' if suffix=='direct' else 'fusion']
   arms[v+'_'+suffix]=arm
 if depths:
  arms['BM25_depth10']={q:pools[q][:10] for q in ids}
  for suffix in ('direct','fusion'):
   arm={q:[] for q in ids}
   for row in rows:
    if row['id'] in arm and row['depth']==10 and row['status']=='ok':arm[row['id']]=row['outputs']['original']['ranking' if suffix=='direct' else 'fusion']
   arms['original_depth10_'+suffix]=arm
 arms['Pool_oracle']={q:sorted(pools[q],key=lambda d:(-labels[q].get(d,0),pools[q].index(d))) for q in ids}
 metrics={name:evaluate(labels,ranks) for name,ranks in arms.items()}
 comparisons={}
 for v in variants[1:]:
  ds=[metrics[v+'_direct']['per_query'][q]['Recall@5']-metrics['original_direct']['per_query'][q]['Recall@5'] for q in ids]
  comparisons[v]={'mean_delta_recall5':statistics.mean(ds),'wins':sum(d>1e-12 for d in ds),'ties':sum(abs(d)<=1e-12 for d in ds),'losses':sum(d< -1e-12 for d in ds),'per_query_delta':dict(zip(ids,ds))}
 return {'metrics':metrics,'comparisons':comparisons,'ids':ids,'positive_counts':{q:sum(v>0 for v in labels[q].values()) for q in ids}}

def promote(analysis,rows):
 if len(rows)!=24 or any(r['status']!='ok' for r in rows):return []
 eligible=[v for v,c in analysis['comparisons'].items() if c['mean_delta_recall5']>=.05 and c['wins']>=2 and c['wins']>c['losses']]
 return sorted(eligible,key=lambda v:(-analysis['comparisons'][v]['mean_delta_recall5'],VARIANTS.index(v)))[:1]

def main():
 plan=json.loads((HERE/'varied-screen-plan.json').read_text());claimed=plan.pop('protocol_sha256')
 packets=json.loads((HERE/'prepared-varied-screen.json').read_text())
 assert claimed==PIN and digest(plan)==PIN and digest(packets)==plan['requests_sha256']
 assert len(packets['dev'])==24 and plan['max_total_calls']==30
 with (HERE/'budget.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  out=HERE/RUN;lp=HERE/'budget-ledger.json';ledger=json.loads(lp.read_text())
  if out.exists():raise ValueError('No reruns')
  if Decimal(ledger['cap_usd'])!=4 or Decimal(ledger['committed_usd'])!=Decimal('.77') or Decimal(ledger['external_spend_usd'])!=1:raise ValueError('Unexpected shared budget')
  if Decimal(ledger['committed_usd'])+Decimal(ledger['external_spend_usd'])+RESERVE>4:raise ValueError('Cap')
  key=load_key('jev');out.mkdir();ledger['committed_usd']='0.87'
  ledger['runs'][RUN]={'reserved_usd':str(RESERVE),'status':'reserved','protocol_sha256':PIN}
  # The general pause remains enabled. Explicit authorization applies only here.
  ledger['paid_tests_paused']=True;ledger['pause_reason']='Only the explicitly authorized frozen varied-screen-v3 may run; all other paid work paused.';save(lp,ledger)
  rows=[];attempts=0;tokens=0;halt=False;selected=[];error=None
  save(out/'disclosure.json',{'authorization':'Fred explicitly resumed varied Jev/BM25 prompt tests; at most USD0.10 additional, within shared USD4 cap. User reports USD5 signup credits; no balance verification or top-up.','endpoint':'https://api.typesafe.ai/v1/systemone','approved_public_fields':['canonical query','candidate name','candidate description'],'max_calls':30,'max_input_tokens':1920000,'usage_bound_usd':'0.08064','reserved_usd':'0.10','protocol_sha256':PIN,'all_other_paid_work_paused':True})
  def deadline(signum,frame):raise TimeoutError('Screen deadline')
  def batch(items,phase):
   nonlocal attempts,tokens,halt
   for item in items:
    row={k:item[k] for k in ('id','depth','tag','variants','pool')};row.update(phase=phase,status='not_attempted')
    if not halt:
     if attempts>=30:raise ValueError('Call cap')
     attempts+=1;row.update(attempt=attempts,request_sha256=digest(item['request']),status='reserved_before_transmission')
     save(out/'progress.json',{'rows':rows+[row],'attempts':attempts});start=time.monotonic()
     try:
      response=post(key,item['request'],urllib.request.build_opener(NoRedirect).open)
      outputs,n=parse(response,item['pool'],item['variants']);tokens+=n
      if tokens>1920000:raise ValueError('Token cap')
      row.update(status='ok',outputs=outputs,input_tokens=n,model=response['model'])
     except BaseException as exc:
      if isinstance(exc,urllib.error.HTTPError):row['http_status']=exc.code;exc.close()
      row.update(status='failed_or_uncertain',error_type=type(exc).__name__);halt=True
     row['api_seconds']=time.monotonic()-start
    rows.append(row);save(out/'progress.json',{'rows':rows,'attempts':attempts})
  signal.signal(signal.SIGALRM,deadline);signal.alarm(900)
  try:
   batch(packets['dev'],'dev')
   # Predictions are durable before any development label evaluation.
   labels=qrels('test');ids=[q['id'] for q in plan['dev_queries']]
   development=analyze(rows,ids,plan['pools'],labels,VARIANTS,True);save(out/'development.json',development)
   selected=promote(development,rows);save(out/'selection.json',{'selected':selected,'rule':plan['promotion'],'made_after_dev_before_confirmation':True})
   if selected:
    batch(packets['confirmation'][selected[0]],'confirmation')
    ids=[q['id'] for q in plan['confirmation_queries']]
    save(out/'confirmation.json',analyze([r for r in rows if r['phase']=='confirmation'],ids,plan['pools'],labels,['original',*selected]))
  except BaseException as exc:error=type(exc).__name__
  finally:
   signal.alarm(0);del key
   result={'rows':rows,'attempts':attempts,'known_input_tokens':tokens,'known_usage_usd':str(Decimal(tokens)*RATE/1000000),'reserved_usd':str(RESERVE),'reserve_released':False,'selected':selected,'protocol_sha256':PIN,'error_type':error,'no_retries':True,'cash_charge_usd':None,'cash_note':'Metered use may consume user-reported USD5 signup credit. Cash charge not verified.'};save(out/'results.json',result)
   ledger['runs'][RUN]['status']='finished_or_stopped';ledger['paid_tests_paused']=True;ledger['pause_reason']='Authorized varied prompt screen completed or stopped; no automatic further paid experiments.';save(lp,ledger)
  print(json.dumps({k:v for k,v in result.items() if k!='rows'}));return 1 if error or halt else 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except (Exception,KeyboardInterrupt):print('Stopped safely; no sensitive exception text, retries, or refunds.');raise SystemExit(1)
