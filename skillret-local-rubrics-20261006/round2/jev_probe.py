"""User-run, one-shot paid probe. No key storage; --prepare-only never networks."""
import argparse,getpass,hashlib,json,os,sys,time,urllib.request,urllib.error,warnings
from pathlib import Path
from decimal import Decimal
from probe_contract import build_request,parse_response,digest,MODEL
from jev_auth_check import NoRedirect,check_key
from secret_store import obtain_key
ENDPOINT='https://api.typesafe.ai/v1/systemone'
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
PIN='0a2c3d1aa6de20d8b350ef68afacdac190eb986a09c989f77a605ff408b3036e'
RESERVE=Decimal('0.10');RATE=Decimal('0.042');MAX_CALLS=12
def save(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def prepare(root=ROOT):
 panel=json.loads((root/'round2/exploration-freeze.json').read_text());claimed=panel.pop('protocol_sha256')
 if claimed!=PIN or digest(panel)!=PIN or len(panel['queries'])!=12:raise ValueError('Frozen protocol changed')
 needed={d for row in panel['candidate_ids'].values() for d in row};headers={};hasher=hashlib.sha256()
 with (root/'data/skills/test.jsonl').open('rb') as f:
  for line in f:
   hasher.update(line);s=json.loads(line)
   if s['id'] in needed:headers[s['id']]={k:s[k] for k in ('id','name','description')}
 if hasher.hexdigest()!='1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81':raise ValueError('Pinned corpus mismatch')
 packets=[]
 for q in panel['queries']:
  candidates=[headers[d] for d in panel['candidate_ids'][q['id']]]
  # Whole-query rejection, never silent truncation or changed candidates.
  try:packet=build_request(q,candidates);error=None
  except ValueError:packet=None;error='payload_bound_or_schema_rejected'
  packets.append({'id':q['id'],'candidates':candidates,'request':packet,'preflight_error':error})
 return packets
def post(key,packet,opener):
 request=urllib.request.Request(ENDPOINT,data=json.dumps(packet).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
 with opener(request,timeout=20) as response:
  raw=response.read(1024*1024+1)
 if len(raw)>1024*1024:raise ValueError('Response bound exceeded')
 return json.loads(raw)
def run_batch(key,packets,out,opener,clock=time.monotonic):
 if len(packets)>MAX_CALLS:raise ValueError('Request cap exceeded')
 rows=[];calls=0;usage=0;start=clock();halt=False
 for item in packets:
  row={'id':item['id'],'candidate_ids':[c['id'] for c in item['candidates']],'rankings':{'noul':[],'score':[]},'status':'not_attempted'}
  if item['preflight_error']:row['status']=item['preflight_error']
  elif not halt:
   if clock()-start>300:halt=True;row['status']='deadline_before_request'
   else:
    calls+=1;row.update(status='reserved_before_transmission',request_sha256=digest(item['request']),attempt=calls)
    save(out/'progress.json',{'rows':rows+[row],'attempts':calls,'reserved_all_in_usd':str(RESERVE)})
    t=clock()
    try:
     response=post(key,item['request'],opener);rankings,tokens=parse_response(response,item['candidates']);usage+=tokens
     if usage>MAX_CALLS*64000:raise ValueError('Token cap exceeded')
     row.update(status='ok',rankings=rankings,input_tokens=tokens,model=MODEL,
       scores={'noul':[response['answers'][f'n{i}']['noul'] for i in range(20)],'score':[response['answers'][f's{i}']['score'] for i in range(20)]})
    except BaseException as e:
     if isinstance(e,urllib.error.HTTPError):e.close()
     # Never serialize exceptions, response bodies, headers, requests or keys.
     row['status']='failed_or_uncertain';row['error_type']=type(e).__name__;halt=True
    row['seconds']=clock()-t
  rows.append(row);save(out/'progress.json',{'rows':rows,'attempts':calls,'reserved_all_in_usd':str(RESERVE)})
 result={'rows':rows,'attempts':calls,'known_input_tokens':usage,'known_usage_usd_at_pinned_rate':str(Decimal(usage)*RATE/Decimal(1000000)),
  'reserved_all_in_usd':str(RESERVE),'reserve_released':False,'protocol_sha256':PIN,'no_automatic_retries':True,
  'claim':'Exploratory exposed public development. Unattempted/failed queries retain denominator12. Known usage excludes uncertain calls; reservation remains until billing reconciliation.'}
 save(out/'results.json',result);return result
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
 packets=prepare();ready=sum(x['request'] is not None for x in packets)
 if args.prepare_only:
  save(HERE/'prepared-public-requests.json',packets);print(json.dumps({'ready_queries':ready,'total_queries':12,'network_calls':0,'max_post_calls':12,'max_input_tokens_total':768000}));return 0
 if not sys.stdin.isatty():print('Use an interactive Ubuntu terminal. No piped or redirected key input.');return 2
 if not ready:print('No valid requests; nothing sent.');return 2
 import fcntl
 with (HERE/'budget.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:print('Another budget-controlled task is active; nothing sent.');return 2
  out=HERE/'jev-typed-12-v1'
  if out.exists():print('This one-shot run already has an artifact directory. No repeat or retry allowed.');return 2
  ledger_path=HERE/'budget-ledger.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'cap_usd':'4.00','committed_usd':'0','external_spend_usd':'0','runs':{}}
  print(f'TypeSafe DIRECT Jev {MODEL}; destination {ENDPOINT}. Not OpenRouter or Jev Router.')
  print(f'{ready}/12 valid queries; up to12 POST calls,40 typed questions each,64k input tokens/call and768k total. No automatic retries.')
  print('Outgoing: PUBLIC SkillRet canonical query text and each frozen candidate NAME/DESCRIPTION, plus fixed relevance rubrics. No qrels, gold IDs, private project files, local files mentioned by benchmark text, credentials in payloads, checkpoints or model weights.')
  print('Usage bound at $0.042/input-million, output free: $0.032256. Reserve $0.10 ALL-IN from the shared $4 round; no credit purchase or auto-refill. Software cannot inspect your balance/tax terms.')
  print('Confirm in your TypeSafe account that this exact price applies, existing credit covers the batch, and all mandatory charges fit $0.10. If not, cancel. No automatic top-up is authorized.')
  external=Decimal(input('Other provider spend/reservations in this $4 round, USD (0 if none): ').strip())
  if not external.is_finite() or external<0:raise ValueError('Invalid external spend')
  external=max(external,Decimal(ledger['external_spend_usd']))
  committed=Decimal(ledger['committed_usd'])
  if Decimal(ledger['cap_usd'])!=Decimal('4') or committed+external+RESERVE>Decimal('4'):print('Shared cap would be exceeded; nothing sent.');return 2
  if input('Type RUN 0.10 to approve this exact public-data transfer and batch; anything else cancels: ').strip()!='RUN 0.10':print('Cancelled.');return 2
  key=obtain_key('jev','TypeSafe API key (hidden): ')
  auth=check_key(key)
  if not auth.get('authenticated'):print(json.dumps(auth));return 1
  out.mkdir();ledger['committed_usd']=str(committed+RESERVE);ledger['external_spend_usd']=str(external);ledger['runs']['jev-typed-12-v1']={'reserved_usd':str(RESERVE),'status':'reserved','protocol_sha256':PIN};save(ledger_path,ledger)
  save(out/'disclosure.json',{'endpoint':ENDPOINT,'model':MODEL,'approved_public_fields':['query','candidate.name','candidate.description'],'key_saved':False,'pricing_usd_per_million_input':str(RATE),'output_price':0,'max_calls':12,'reserve_usd':str(RESERVE),'user_billing_and_transfer_confirmation':True})
  import signal
  def deadline(signum,frame):raise TimeoutError('Batch deadline')
  signal.signal(signal.SIGALRM,deadline);signal.alarm(360)
  try:result=run_batch(key,packets,out,urllib.request.build_opener(NoRedirect).open)
  finally:signal.alarm(0);del key
  ledger['runs']['jev-typed-12-v1']['status']='finished_or_stopped';save(ledger_path,ledger)
  print(json.dumps({'results_directory':str(out),'attempts':result['attempts'],'known_usage_usd':result['known_usage_usd_at_pinned_rate'],'reserved_usd':'0.10','key_saved':False}));return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except (Exception,KeyboardInterrupt):print('Stopped safely. No error body or credential displayed. Any existing reservation remains; inspect local receipts before further paid work.');raise SystemExit(1)
