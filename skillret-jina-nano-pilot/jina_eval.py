import pathlib,sys,os,json,time,signal,hashlib
sys.dont_write_bytecode=True
r=pathlib.Path.home()/'code/jina-nano-pilot';e=pathlib.Path.home()/'code/embeddinggemma2-pilot';o=pathlib.Path(os.environ['PILOT_RESULT']);os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HOME=str(r/'hf-cache'),HF_MODULES_CACHE=str(r/'hf-modules'))
def save(n,x):(o/(n+'.json')).write_text(json.dumps(x,indent=2))
def stage(n):save('stage',{'stage':n})
def timeout(a,b):raise TimeoutError('Embedding exceeded 60 seconds')
signal.signal(signal.SIGALRM,timeout);stage('import');start=time.perf_counter();cpu=time.process_time()
import torch,numpy as np,importlib.metadata
from transformers import AutoConfig,AutoModel
save('import',{'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu});torch.set_num_threads(1);torch.set_num_interop_threads(1);assert torch.cuda.get_device_capability()==(7,5)
for x in json.loads((r/'review-manifest.json').read_text()):
 if x['file'].endswith('.py'):assert hashlib.sha256((r/'model'/x['file']).read_bytes()).hexdigest()==x['sha256']
stage('load');start=time.perf_counter();cpu=time.process_time();config=AutoConfig.from_pretrained(r/'model',trust_remote_code=True,local_files_only=True);config._attn_implementation='eager'
model=AutoModel.from_pretrained(r/'model',config=config,trust_remote_code=True,local_files_only=True,dtype=torch.float32).to(device='cuda',dtype=torch.float32).eval();model.set_adapter(['retrieval']);torch.cuda.synchronize();save('load',{'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu});assert all(p.device.type=='cuda' and p.dtype==torch.float32 for p in model.parameters());assert model.config._attn_implementation=='eager'
save('runtime',{'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'dtype':'float32','attention':model.config._attn_implementation,'active_adapters':model.active_adapters,'max_tokens':1024,'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'packages':{n:importlib.metadata.version(n) for n in ['transformers','peft','huggingface-hub','safetensors']},'config':model.config.to_dict()})
for p in (r/'model').rglob('*.safetensors'):
 with open(p,'rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
torch.cuda.reset_peak_memory_stats();shapes=[]
def hook(module,args,kwargs):
 if 'input_ids' in kwargs:shapes.append(list(kwargs['input_ids'].shape))
handle=model.register_forward_pre_hook(hook,with_kwargs=True)
def embed(text,prompt,ident):
 stage(ident);prefix='Query: ' if prompt=='query' else 'Document: ';raw=len(model.tokenizer(prefix+text,truncation=False)['input_ids']);shapes.clear();torch.cuda.synchronize();start=time.perf_counter();cpu=time.process_time();signal.alarm(60)
 try:
  output=model.encode(texts=[text],task='retrieval',prompt_name=prompt,max_length=1024);v=output[0].float().cpu().numpy();torch.cuda.synchronize()
 finally:signal.alarm(0)
 wall=time.perf_counter()-start;used=time.process_time()-cpu;assert v.shape==(768,) and np.isfinite(v).all() and abs(float(np.linalg.norm(v))-1)<1e-4;assert model.active_adapters==['retrieval'];assert shapes==[[1,min(raw,1024)]],(shapes,raw)
 with open(o/'calls.jsonl','a') as f:f.write(json.dumps({'id':ident,'prompt':prompt,'wall_seconds':wall,'cpu_seconds':used,'raw_tokens':raw,'used_tokens':shapes[0][1],'truncated':raw>1024})+'\n')
 return v
smoke=[]
for repeat in range(3):
 vectors=[embed(t,p,f'smoke/{repeat}/{i}') for i,(t,p) in enumerate([('What causes the northern lights?','query'),('The northern lights are caused by charged particles from the sun.','document'),('A bicycle has two wheels and pedals.','document')])];scores=[float(vectors[0]@v) for v in vectors[1:]];assert scores[0]>scores[1];smoke.append(scores);np.save(o/f'smoke-{repeat}.npy',vectors)
save('smoke',{'passed':True,'scores':smoke});print('SMOKE_PASS',flush=True)
small=json.loads((e/'fixtures/qwen-small-fixtures.json').read_text());dv=np.stack([embed(t,'document',f'small/document/{i}') for i,t in enumerate(small['documents'])]);np.save(o/'small-docs.npy',dv)
for repeat in range(3):np.save(o/f'small-queries-{repeat}.npy',np.stack([embed(t,'query',f'small/query/{repeat}/{i}') for i,t in enumerate(small['queries'])]))
print('SMALL_COMPLETE',flush=True);pilot=json.loads((e/'fixtures/skillret-pilot.json').read_text());np.save(o/'skillret-docs.npy',np.stack([embed(x['text'],'document',f'skillret/document/{i}') for i,x in enumerate(pilot['documents'])]))
for repeat in range(2):np.save(o/f'skillret-queries-{repeat}.npy',np.stack([embed(x['query'],'query',f'skillret/query/{repeat}/{i}') for i,x in enumerate(pilot['queries'])]))
save('torch-memory',{'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved()});save('loaded-libraries',sorted(set(x.split()[-1] for x in pathlib.Path('/proc/self/maps').read_text().splitlines() if any(s in x for s in ['libcuda','libcudnn','libcublas','libtorch_cuda']))));stage('done');print('PILOT_COMPLETE',flush=True)
