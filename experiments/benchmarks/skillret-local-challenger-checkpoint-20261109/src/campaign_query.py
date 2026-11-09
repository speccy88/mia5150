"""Bounded serialized query campaign; never uses official test outcomes."""
from common import *
import subprocess,sys,time,os
PYTHON=sys.executable

def compatible_completed_stage(saved,requested,script,args):
 if saved==requested:return True
 if script not in ('train_query.py','train_reranker.py') or '--name' not in args:return False
 if len(saved)!=len(requested)+2 or saved[:-2]!=requested or saved[-2]!='--resume':return False
 family='query' if script=='train_query.py' else 'reranker'
 trial=str(args[args.index('--name')+1])
 summary=STUDY/'results'/family/trial/'summary.json'
 if not summary.exists():return False
 result=read(summary)
 return result['status']=='complete' and result.get('resumed_from_step',0)>0

def run(name,script,args=(),max_seconds=7200):
 argv=[PYTHON,str(STUDY/'src'/script),*map(str,args)]
 public=[Path(argv[0]).name]+[Path(x).name if x.startswith('/') else x for x in argv[1:]]
 existing=sorted((CACHE/'runs').glob(name+'*'))
 for folder in existing:
  receipt=folder/'exit.json';command=folder/'command.json'
  if receipt.exists() and command.exists() and read(receipt)['exit_code']==0 and compatible_completed_stage(read(command)['argv'],public,script,args):
   print('Reusing completed stage receipt',folder.name,flush=True);return
 attempt=name if not(CACHE/'runs'/name).exists() else name+'-retry-'+str(len(existing)+1)
 log=CACHE/(attempt+'.log');env=dict(os.environ,CHALLENGER_MAX_SECONDS=str(max_seconds))
 with log.open('w') as f:
  p=subprocess.run([PYTHON,str(STUDY/'src/guard.py'),attempt,*argv],env=env,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,('Stage failed; preserve receipt and inspect',attempt,p.returncode)

def main():
 # This orchestrator queues only dependencies; receipts distinguish queued from executed.
 deadline=time.monotonic()+15000
 while not(CACHE/'vectors/train-queries/complete.json').exists():
  if (CACHE/'runs/index-full-01/exit.json').exists():assert read(CACHE/'runs/index-full-01/exit.json')['exit_code']==0,'Index stopped; do not start training'
  assert time.monotonic()<deadline,'Index dependency wait limit'
  time.sleep(15)
 # Completion receipts are written before model process shutdown.
 while True:
  import subprocess
  memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
  if memory<500:break
  time.sleep(3)
 run('query-resume-parity-01','resume_parity.py',[],1800)
 run('query-objective-smoke-01','train_query.py',['--name','objective-smoke','--steps','4','--smoke'],600)
 smoke=read(STUDY/'results/query/objective-smoke/summary.json');steps=rows(STUDY/'results/query/objective-smoke/steps.jsonl');median=float(np.median([x['seconds'] for x in steps]));budget={'measured_optimizer_step_median_seconds':median,'pilot_steps_each':100,'pilot_time_limit_seconds':max(900,int(median*100*2+300)),'full_maximum_steps':1200,'full_time_limit_seconds':max(3600,int(median*1200*2+1800)),'checkpoint_storage_limit_bytes':1024**3,'maximum_passes':2,'smoke_discarded':True};save(STUDY/'results/query/admission.json',budget)
 trials=[]
 for name,lr in [('pilot-3e-6',3e-6),('pilot-1e-6',1e-6),('pilot-1e-5',1e-5)]:
  run('query-'+name+'-01','train_query.py',['--name',name,'--lr',lr,'--steps','100'],budget['pilot_time_limit_seconds']);best=read(STUDY/'results/query'/name/'best.json');trials.append({'name':name,'lr':lr,**best})
 # Earliest trial wins exact ties, per declared LR ordering; base weight0 remains a real candidate.
 winner=max(trials,key=lambda x:x['ndcg10']);baseline=read(STUDY/'results/frozen-selection.json')['evaluation']['aggregate']['NDCG@10'];save(STUDY/'results/query/pilot_comparison.json',{'trials':trials,'selected':winner,'baseline_ndcg10':baseline})
 name=winner['name']
 if winner['ndcg10']>baseline+1e-12:
  full_name='full-selected-lr';run('query-full-selected-lr-01','train_query.py',['--name',full_name,'--lr',winner['lr'],'--steps','1200'],budget['full_time_limit_seconds'])
  if read(STUDY/'results/query'/full_name/'best.json')['ndcg10']>winner['ndcg10']+1e-12:name=full_name
 best=read(STUDY/'results/query'/name/'best.json');folder=CACHE/'checkpoints/query'/name/'best';assert sha(folder/'adapter_model.safetensors')==best['adapter_sha256'];chosen={'trial':name,'relative_cache_path':str(folder.relative_to(CACHE)),'adapter_sha256':best['adapter_sha256'],'mixture_weight':best['weight'],'best_step':best['step'],'selected_before_confirmation':True,'baseline_selection_ndcg10':baseline,'selection_ndcg10':best['ndcg10']};save(STUDY/'results/query/SELECTION_LOCK.json',chosen)
 save(STUDY/'results/query/campaign_complete.json',{'status':'selected','confirmation_opened':False,'selected':chosen})
 print('Query campaign completed; final official test remains locked',flush=True)
if __name__=='__main__':main()
