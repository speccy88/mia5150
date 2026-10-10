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
model.max_seq_length=1024;torch.cuda.synchronize();save('load',{'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu})
native=model[0].auto_model;assert native.vision_tower is None and native.audio_tower is None;assert all(p.dtype==torch.float32 and p.device.type=='cuda' for p in native.parameters());parameters=sum(p.numel() for p in native.parameters());assert parameters<320_000_000
save('runtime',{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'parameters':parameters,'dtype':'float32','attention':'eager','max_tokens':1024,'dimension':768,'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'prompts':model.prompts,'packages':{n:importlib.metadata.version(n) for n in ['transformers','sentence-transformers','tokenizers','huggingface-hub','safetensors']},'model_config':native.config.to_dict()})
for p in (r/'model').glob('*.safetensors'):
 with open(p,'rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
torch.cuda.reset_peak_memory_stats();timings=[]
def embed(text,prompt,ident):
 stage(ident);prefix=model.prompts[prompt];raw=len(model.tokenizer(prefix+text,add_special_tokens=True,truncation=False)['input_ids']);torch.cuda.synchronize();start=time.perf_counter();cpu=time.process_time();signal.alarm(60)
 try:
  v=model.encode([text],prompt_name=prompt,batch_size=1,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)[0];torch.cuda.synchronize()
 finally:signal.alarm(0)
 wall=time.perf_counter()-start;used=time.process_time()-cpu;assert v.shape==(768,) and np.isfinite(v).all() and abs(float(np.linalg.norm(v))-1)<1e-4
 row={'id':ident,'prompt':prompt,'wall_seconds':wall,'cpu_seconds':used,'raw_tokens':raw,'used_tokens':min(raw,1024),'truncated':raw>1024};timings.append(row)
 with open(o/'calls.jsonl','a') as f:f.write(json.dumps(row)+'\n')
 return v
pilot=json.loads((r/'fixtures/skillret-pilot.json').read_text());index=next(i for i,q in enumerate(pilot['queries']) if q['id']=='q-02348');query=pilot['queries'][index];shapes=[]
def capture(module,args,kwargs):
 if kwargs.get('input_ids') is not None:shapes.append(list(kwargs['input_ids'].shape))
 elif kwargs.get('inputs_embeds') is not None:shapes.append(list(kwargs['inputs_embeds'].shape[:2]))
handle=native.language_model.register_forward_pre_hook(capture,with_kwargs=True)
vectors=np.stack([embed(query['query'],'SearchQuery',f'sensitivity/{repeat}') for repeat in range(3)]);handle.remove();np.save(o/'query-1024.npy',vectors)
assert len(shapes)==3 and all(x==[1,836] for x in shapes),shapes
old=r/'evaluation-compatible';docs=np.load(old/'skillret-docs.npy');original=np.load(old/'skillret-queries-0.npy');updated=original.copy();updated[index]=vectors[0];np.save(o/'skillret-queries-sensitive.npy',updated)
ids=[d['id'] for d in pilot['documents']];before=[ids[j] for j in np.argsort(-(docs@original[index]),kind='stable')];after=[ids[j] for j in np.argsort(-(docs@vectors[0]),kind='stable')];gold=set(pilot['qrels'][query['id']]);save('sensitivity',{'query_id':query['id'],'index':index,'actual_input_shapes':shapes,'before_top20':before[:20],'after_top20':after[:20],'before_recall':{k:len(set(before[:k])&gold)/len(gold) for k in [1,5,10,20]},'after_recall':{k:len(set(after[:k])&gold)/len(gold) for k in [1,5,10,20]},'query_cosine_before_after':float(original[index]@vectors[0]),'repeat_max_absolute_difference':float(np.max(np.abs(vectors-vectors[0]))),'changed_rows':1,'original_scores_preserved':True})
save('torch-memory',{'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved()});stage('done');print('SENSITIVITY_COMPLETE',flush=True)
