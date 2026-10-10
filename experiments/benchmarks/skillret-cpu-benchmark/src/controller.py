"""Serial recovery controller with process-group deadline and durable stage gates."""
import os,sys,time,json,subprocess,signal,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'protocol';P.mkdir(exist_ok=True)
DEADLINE=datetime.datetime(2026,10,2,11,52,tzinfo=datetime.timezone.utc).timestamp()
child=None

def save(obj):
 p=P/'controller_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def stop(*args):
 global child
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 save({'status':'stopped','deadline_utc':'2026-10-02T11:52:00Z','stopped_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});sys.exit(0)
for s in [signal.SIGTERM,signal.SIGINT]:signal.signal(s,stop)
extension=int(json.loads((P/'extension_choice.json').read_text())['query_count']) if (P/'extension_choice.json').exists() else 64
assert extension in [48,64]
stages=[]
for m in ['bm25','bge-small','e5-small','embeddinggemma-int8']:stages.append((f'retrieval64_{m}', '.venv', ['src/run_retrieval.py',m],f'results/retrieval64/{m}_run.json'))
for mode,n in [('quality',32),('timing',32),('quality',extension)]:
 for m in ['julia','gliner','verdict','laya']:
  v='.venv-julia' if m=='julia' else '.venv-gliner' if m=='gliner' else '.venv';stages.append((f'{mode}{n}_{m}',v,['src/run_decision.py',m,mode,str(n)],f'results/{mode}{n}/{m}_run.json'))
(P/'recovery_plan.json').write_text(json.dumps({'deadline_utc':'2026-10-02T11:52:00Z','deliver_by_utc':'2026-10-02T12:12:00Z','frozen_cohort':'First64/32 from original outcome-independent SHA-ordered256','threads':4,'serial':True,'stage_checkpoint_policy':'Wait for Library checkpoint acknowledgement after every stage','stages':[{'stage':a,'command':[b+'/bin/python',*c],'receipt':d} for a,b,c,d in stages]},indent=2))
for stage,venv,args,receipt in stages:
 if time.time()>=DEADLINE:stop()
 if (R/receipt).exists() and json.loads((R/receipt).read_text()).get('status')=='completed':
  if (P/(stage+'.checkpoint_ack')).exists():continue
  save({'status':'awaiting_durable_checkpoint','stage':stage,'deadline_utc':'2026-10-02T11:52:00Z'})
  while not (P/(stage+'.checkpoint_ack')).exists():
   if time.time()>=DEADLINE:stop()
   time.sleep(5)
  continue
 if stage=='quality64_julia' and not (P/'approve_quality64').exists():
  save({'status':'awaiting_budget_decision','stage':stage,'deadline_utc':'2026-10-02T11:52:00Z','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  while not (P/'approve_quality64').exists():
   if (P/'finish_at32').exists():save({'status':'completed_common32','deadline_utc':'2026-10-02T11:52:00Z'});sys.exit(0)
   if time.time()>=DEADLINE:stop()
   time.sleep(5)
 logfile=R/'logs'/f'{stage}.log';started=time.time()
 with logfile.open('a') as f:
  child=subprocess.Popen([str(R/venv/'bin/python'),'-u',*args],cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1'})
  while child.poll() is None:
   save({'status':'running','stage':stage,'controller_pid':os.getpid(),'child_pid':child.pid,'started_utc':datetime.datetime.fromtimestamp(started,datetime.timezone.utc).isoformat(),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.time()-started,'deadline_utc':'2026-10-02T11:52:00Z','log':str(logfile)})
   if time.time()>=DEADLINE:stop()
   time.sleep(5)
 if child.returncode!=0:
  save({'status':'stage_failed','stage':stage,'exit_code':child.returncode,'log':str(logfile),'deadline_utc':'2026-10-02T11:52:00Z'});sys.exit(child.returncode)
 ack=P/(stage+'.checkpoint_ack')
 while not ack.exists():
  save({'status':'awaiting_durable_checkpoint','stage':stage,'controller_pid':os.getpid(),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_utc':'2026-10-02T11:52:00Z'})
  if time.time()>=DEADLINE:stop()
  time.sleep(5)
save({'status':f'completed_common{extension}','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_utc':'2026-10-02T11:52:00Z'})
