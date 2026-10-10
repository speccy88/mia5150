import pathlib,sys,os,json,time,signal,hashlib
r=pathlib.Path.home()/'code/embeddinggemma2-pilot';sys.path.insert(0,str(r/'overlay'));o=pathlib.Path(os.environ['PILOT_RESULT']);os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
def save(name,x):(o/(name+'.json')).write_text(json.dumps(x,indent=2))
def stage(name):save('stage',{'stage':name})
def timeout(a,b):raise TimeoutError('Embedding request exceeded 60 seconds')
signal.signal(signal.SIGALRM,timeout)
stage('import');start=time.perf_counter();cpu=time.process_time()
import torch,numpy as np,importlib.metadata
from sentence_transformers import SentenceTransformer
save('import',{'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu});torch.set_num_threads(1);torch.set_num_interop_threads(1);assert torch.cuda.get_device_capability()==(7,5)
stage('load');start=time.perf_counter();cpu=time.process_time()
model=SentenceTransformer(str(r/'model'),device='cuda',local_files_only=True,trust_remote_code=False,config_kwargs={'vision_config':None,'audio_config':None},model_kwargs={'dtype':torch.float32,'attn_implementation':'eager','device_map':'cuda'})
model.max_seq_length=512;torch.cuda.synchronize();save('load',{'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu})
native=model[0].auto_model;assert native.vision_tower is None and native.audio_tower is None;assert all(p.dtype==torch.float32 and p.device.type=='cuda' for p in native.parameters());parameters=sum(p.numel() for p in native.parameters());assert parameters<320_000_000
save('runtime',{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'parameters':parameters,'dtype':'float32','attention':'eager','max_tokens':512,'dimension':768,'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'prompts':model.prompts,'packages':{n:importlib.metadata.version(n) for n in ['transformers','sentence-transformers','tokenizers','huggingface-hub','safetensors']},'model_config':native.config.to_dict()})
for p in (r/'model').glob('*.safetensors'):
 with open(p,'rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
torch.cuda.reset_peak_memory_stats();timings=[]
def embed(text,prompt,ident):
 stage(ident);prefix=model.prompts[prompt];raw=len(model.tokenizer(prefix+text,add_special_tokens=True,truncation=False)['input_ids']);torch.cuda.synchronize();start=time.perf_counter();cpu=time.process_time();signal.alarm(60)
 try:
  v=model.encode([text],prompt_name=prompt,batch_size=1,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)[0];torch.cuda.synchronize()
 finally:signal.alarm(0)
 wall=time.perf_counter()-start;used=time.process_time()-cpu;assert v.shape==(768,) and np.isfinite(v).all() and abs(float(np.linalg.norm(v))-1)<1e-4
 row={'id':ident,'prompt':prompt,'wall_seconds':wall,'cpu_seconds':used,'raw_tokens':raw,'used_tokens':min(raw,512),'truncated':raw>512};timings.append(row)
 with open(o/'calls.jsonl','a') as f:f.write(json.dumps(row)+'\n')
 return v
smoke=[]
for repeat in range(3):
 vectors=[embed(t,p,f'smoke/{repeat}/{i}') for i,(t,p) in enumerate([('What causes the northern lights?','SearchQuery'),('The northern lights are caused by charged particles from the sun.','Document'),('A bicycle has two wheels and pedals.','Document')])]
 scores=[float(vectors[0]@v) for v in vectors[1:]];assert scores[0]>scores[1];smoke.append({'repeat':repeat,'scores':scores});np.save(o/f'smoke-{repeat}.npy',vectors)
save('smoke',{'passed':True,'cases':smoke});print('SMOKE_PASS',flush=True)
small=json.loads((r/'fixtures/qwen-small-fixtures.json').read_text());dv=np.stack([embed(t,'Document',f'small/document/{i}') for i,t in enumerate(small['documents'])]);np.save(o/'small-docs.npy',dv)
for repeat in range(3):
 qv=np.stack([embed(t,'SearchQuery',f'small/query/{repeat}/{i}') for i,t in enumerate(small['queries'])]);np.save(o/f'small-queries-{repeat}.npy',qv)
print('SMALL_COMPLETE',flush=True)
pilot=json.loads((r/'fixtures/skillret-pilot.json').read_text());dv=np.stack([embed(d['text'],'Document',f'skillret/document/{i}') for i,d in enumerate(pilot['documents'])]);np.save(o/'skillret-docs.npy',dv)
for repeat in range(2):
 qv=np.stack([embed(q['query'],'SearchQuery',f'skillret/query/{repeat}/{i}') for i,q in enumerate(pilot['queries'])]);np.save(o/f'skillret-queries-{repeat}.npy',qv)
save('torch-memory',{'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved()});save('loaded-libraries',sorted(set(x.split()[-1] for x in pathlib.Path('/proc/self/maps').read_text().splitlines() if any(s in x for s in ['libcuda','libcudnn','libcublas','libtorch_cuda']))));stage('done');print('PILOT_COMPLETE',flush=True)
