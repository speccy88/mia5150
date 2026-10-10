"""Bounded local finite-score diagnostic. Existing cached models; localhost only."""
import argparse,hashlib,json,math,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2';sys.path.insert(0,str(ROOT))
from local_server import server
from harness import evaluate,qrels
from probe_contract import digest
from jev_probe import save
sys.path.insert(0,'/home/user/code/routing-pilot/source')
from jevk5 import JevK5GGUF
MODELS={
 'jevk5':{'path':'/home/user/code/routing-pilot/jevk5/models/jevk5-4b-v0.3-Q5_K_M.gguf','sha256':'b1df9869dd14c0f5b243fa956dd985f0141c3d23ec4847da4c503c73c1050601','temperature':1.22},
 'plumb':{'path':'/home/user/code/routing-pilot/plumb/models/plumb-4b-v5-Q4_K_M.gguf','sha256':'d605acccbcf18e283d01f483fccfc9815335f6c46d1030a84e57acb34dae32c9','temperature':2.07},
 'qwen35':{'path':'/home/user/code/category-pilot/multimodal/models/Qwen3.5-4B-Q4_K_M.gguf','sha256':'00fe7986ff5f6b463e6245582114603938a70800d1fb69ef11a4','temperature':1.0}}
# Exact Qwen hash from its existing manifest; assigned explicitly to avoid abbreviation.
MODELS['qwen35']['sha256']='00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'

def prepare():
 p=json.loads((HERE/'prepared-varied-screen.json').read_text())
 ids=list(dict.fromkeys(x['id'] for x in p['dev']))[:2];items=[]
 for q in ids:
  for v in ('original','constraint'):
   source=next(x for x in p['dev'] if x['id']==q and x['depth']==20 and v in x['variants'])
   items.append({'id':q,'variant':v,'pool':source['pool'],'state':source['request']['state'],'questions':[source['request']['questions'][f'{v}_{i}'] for i in range(20)]})
 hashes={n:hashlib.sha256((Path('/home/user/code/routing-pilot/source/jevk5')/n).read_bytes()).hexdigest() for n in ['__init__.py','gguf.py','prompt.py']}
 plan={'version':'local-rubric-screen-v1','ids':ids,'selection':'First two queries of existing six-query development panel, fixed order; already exposed. No confirmation cohort or new labels used for selection.','variants':['original','constraint'],'models':MODELS,'adapter_source_sha256':hashes,'prepared_items':items,'max_forward_passes_per_model':80,'max_forward_passes_total':240,'max_model_seconds':600,'context_tokens':2048,'top_logprob_count':256,'cache_prompt':False,'readout':'Existing native JevK5/SemIf four-letter Score distribution; same prompt and finite-score readout on all three models. Temperatures reused unchanged from prior local routing implementation; not fitted to SkillRet. Missing option letter invalidates row, no probability-floor substitution accepted.','claim':'Two-query diagnostic, not confirmation, not hosted TypeSafe Jev. Direct ranking and RRF60 separate; strict zero ranking on invalid query-rubric output; all denominators retained.','api_calls':0,'downloads':0}
 plan['protocol_sha256']=digest(plan);dest=HERE/'local-rubric-plan.json'
 if dest.exists():assert json.loads(dest.read_text())==plan
 else:save(dest,plan)
 return plan

class Bounded(JevK5GGUF):
 def _post(self,path,payload):
  if self.url!='http://127.0.0.1:18089' or path not in ('/tokenize','/completion'):raise ValueError('Local endpoint only')
  if path=='/completion':
   self.calls+=1
   if self.calls>80 or len(payload['prompt'])+1>2048 or payload['n_predict']!=1:raise ValueError('Frozen inference bound')
  r=super()._post(path,payload)
  if path=='/tokenize' and len(r['tokens'])+1>2048:raise ValueError('Context bound; no truncation')
  return r

def rank(pool,scores):
 if len(pool)!=20 or len(scores)!=20 or len(set(pool))!=20 or any(not math.isfinite(s) or not 0<=s<=3 for s in scores):raise ValueError('Invalid scores')
 direct=[pool[i] for i in sorted(range(20),key=lambda i:(-scores[i],i))]
 fs={d:1/(61+pool.index(d))+1/(61+direct.index(d)) for d in pool}
 return direct,sorted(pool,key=lambda d:(-fs[d],pool.index(d)))

def main():
 arg=argparse.ArgumentParser();arg.add_argument('--prepare-only',action='store_true');arg.add_argument('--model',choices=list(MODELS));args=arg.parse_args();plan=prepare()
 if args.prepare_only:print(json.dumps({'protocol_sha256':plan['protocol_sha256'],'models':list(MODELS),'ids':plan['ids'],'max_local_forwards':240,'network_calls':0}));return
 if not args.model:raise ValueError('Choose one model')
 model=MODELS[args.model];out=HERE/('local-rubric-'+args.model+'-v1');out.mkdir(exist_ok=False)
 def deadline(signum,frame):raise TimeoutError('Bounded model time')
 signal.signal(signal.SIGALRM,deadline);signal.alarm(600);rows=[];halt=False;started=time.monotonic()
 try:
  with open(model['path'],'rb') as f:
   actual=hashlib.file_digest(f,'sha256').hexdigest();os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
  if actual!=model['sha256']:raise ValueError('Model pin mismatch')
  client=Bounded(url='http://127.0.0.1:18089',temperature=model['temperature'],top_k=256,timeout_s=60);client.calls=0
  with server(model['path'],'local-rubric-'+args.model,2048):
   for item in plan['prepared_items']:
    row={'id':item['id'],'variant':item['variant'],'pool':item['pool'],'status':'not_attempted','direct':[],'fusion':[],'answers':[]};t=time.monotonic()
    if not halt:
     try:
      for question in item['questions']:
       missing=client.missing;before=time.monotonic();answer=client.decide(item['state'],question)
       row['answers'].append({'answer':answer,'seconds':time.monotonic()-before,'missing_letters':client.missing-missing});save(out/'progress.json',{'rows':rows+[row],'forward_calls':client.calls})
       if client.missing!=missing:raise ValueError('Missing option letter; no approximate replacement accepted')
      row['direct'],row['fusion']=rank(item['pool'],[x['answer']['score'] for x in row['answers']]);row['status']='ok'
     except Exception as exc:row.update(status='failed',error_type=type(exc).__name__);halt=True
    row['wall_seconds']=time.monotonic()-t;rows.append(row);save(out/'progress.json',{'rows':rows,'forward_calls':client.calls});print(args.model,row['id'],row['variant'],row['status'],round(row['wall_seconds'],3),flush=True)
  result={'model':args.model,'model_sha256':actual,'protocol_sha256':plan['protocol_sha256'],'rows':rows,'forward_calls':client.calls,'total_seconds_including_load':time.monotonic()-started,'hosted_api_calls':0,'cost_usd':0};save(out/'results.json',result)
  labels=qrels('test');labels={q:labels[q] for q in plan['ids']};pool={i['id']:i['pool'] for i in plan['prepared_items']};systems={'BM25':pool,'Oracle':{q:sorted(v,key=lambda d:(-labels[q].get(d,0),v.index(d))) for q,v in pool.items()}}
  for v in plan['variants']:
   for mode in ['direct','fusion']:systems[v+'_'+mode]={r['id']:r[mode] for r in rows if r['variant']==v}
  save(out/'evaluation.json',{'metrics':{k:evaluate(labels,v) for k,v in systems.items()},'note':plan['claim']})
 finally:signal.alarm(0)
if __name__=='__main__':main()
