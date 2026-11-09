"""One CUDA resident model, exact baseline RoPE/pooling, query-only LoRA."""
from common import *
import time
import torch
from transformers import AutoConfig,AutoModel,AutoModelForCausalLM,AutoTokenizer

def setup(seed=42):
 import random
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_math_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(True);torch.backends.cuda.enable_cudnn_sdp(False)
 from transformers.integrations import sdpa_attention
 sdpa_attention.use_gqa_in_sdpa=lambda attention_mask,key:False
 assert torch.cuda.get_device_capability()==(7,5)

def load(kind='embedding',adapter=None,train=False,seed=42):
 setup(seed);p=CACHE/'models'/kind;manifest=read(STUDY/'manifests'/('embedding_source.json' if kind=='embedding' else 'reranker_source.json'))
 for n,meta in manifest['files'].items():assert sha(p/n)==meta['sha256'],('Model integrity',n)
 cfg=AutoConfig.from_pretrained(p,local_files_only=True,trust_remote_code=False)
 raw=read(p/'config.json');rp=raw.get('rope_parameters');theta=rp['rope_theta'] if rp else raw['rope_theta'];cfg.rope_theta=theta;cfg.rope_scaling=None;cfg.use_cache=False
 cls=AutoModel if kind=='embedding' else AutoModelForCausalLM
 m=cls.from_pretrained(p,config=cfg,local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa',device_map='cuda',low_cpu_mem_usage=True).eval()
 backbone=m if kind=='embedding' else m.model
 expected=1/(theta**(torch.arange(0,cfg.head_dim,2,device='cuda',dtype=torch.float32)/cfg.head_dim))
 assert torch.allclose(backbone.rotary_emb.inv_freq,expected,rtol=1e-5,atol=1e-8)
 tok=AutoTokenizer.from_pretrained(p,local_files_only=True,padding_side='left');tok.truncation_side='right'
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 targets=[n for n,v in m.named_modules() if n.endswith(('.q_proj','.v_proj'))];assert len(targets)==2*cfg.num_hidden_layers
 if train or adapter:
  from peft import LoraConfig,get_peft_model,PeftModel
  if adapter:
   adapter_cfg=read(Path(adapter)/'adapter_config.json');assert adapter_cfg['base_model_name_or_path']==manifest['repository'] and adapter_cfg['revision']==manifest['revision'],'Adapter belongs to a different pinned encoder role'
   m=PeftModel.from_pretrained(m,str(adapter),is_trainable=train)
  else:m=get_peft_model(m,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none'))
  for n,p in m.named_parameters():
   if 'lora_' in n:p.data=p.data.float()
  if train:m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});m.enable_input_require_grads();m.train()
 for f in (CACHE/'models'/kind).glob('*.safetensors'):
  with f.open('rb') as stream:os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 return m,tok

def vector(m,tok,text,budget=32768,training=False):
 x=tok(text,return_tensors='pt',truncation=True,max_length=budget).to('cuda')
 # Left padding ensures the last position is the last nonpadding token.
 assert bool(x.attention_mask[0,-1]);h=m(**x,use_cache=False).last_hidden_state[:,-1].float()
 return torch.nn.functional.normalize(h,p=2,dim=-1)[0]

def fingerprint(m):
 h=hashlib.sha256()
 for n,p in m.named_parameters():
  if 'lora_' in n:continue
  h.update(n.replace('base_model.model.','').replace('.base_layer','').encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def save_adapter(m,path):
 import tempfile,shutil
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);staging=Path(tempfile.mkdtemp(prefix='adapter-staging-',dir=path.parent));m.save_pretrained(staging,safe_serialization=True)
 cfg=read(staging/'adapter_config.json');kind='embedding' if m.get_base_model().__class__.__name__=='Qwen3Model' else 'reranker'
 pin=read(STUDY/'manifests'/('embedding_source.json' if kind=='embedding' else 'reranker_source.json'))
 cfg['base_model_name_or_path']=pin['repository'];cfg['revision']=pin['revision'];save(staging/'adapter_config.json',cfg);path.mkdir(parents=True,exist_ok=True)
 for name in ('adapter_config.json','adapter_model.safetensors'):(staging/name).replace(path/name)
 shutil.rmtree(staging)
 return sha(path/'adapter_model.safetensors')

def index(split='train'):
 m,tok=load();s,q,g=data(split);items=[fulltext(x) for x in s];ids=[x['id'] for x in s]
 encode_cache(m,tok,items,ids,f'{split}-documents',32768)

def encode_cache(m,tok,texts,ids,name,budget=32768):
 adapter_hash=None
 if hasattr(m,'peft_config'):
  h=hashlib.sha256()
  for n,p in m.named_parameters():
   if 'lora_' in n:h.update(n.encode());h.update(p.detach().float().cpu().numpy().tobytes())
  adapter_hash=h.hexdigest()
 p=CACHE/'vectors'/name;p.mkdir(parents=True,exist_ok=True);meta={'ids':ids,'text_hashes':[digest(t) for t in texts],'budget':budget,'adapter_tensor_sha256':adapter_hash,'model_revision':MODEL_REV,'source':read(STUDY/'manifests/embedding_source.json'),'runtime':{'torch':torch.__version__,'transformers':__import__('transformers').__version__},'engine_sha256':sha(__file__)}
 if (p/'contract.json').exists():
  old=read(p/'contract.json');old.setdefault('adapter_tensor_sha256',None)
  # Source edits unrelated to encoding cannot alter saved vectors; runtime and all scientific inputs must agree.
  assert {k:v for k,v in old.items() if k!='engine_sha256'}=={k:v for k,v in meta.items() if k!='engine_sha256'}
 else:save(p/'contract.json',meta)
 f=p/'vectors.npy';a=np.lib.format.open_memmap(f,mode='r+' if f.exists() else 'w+',dtype=np.float32,shape=(len(texts),1024));journal=p/'journal.jsonl';done={}
 if journal.exists():
  for x in rows(journal):
   i=x['index'];assert i not in done and x['id']==ids[i];assert digest(a[i].tobytes().hex())==x['vector_sha256'];assert abs(float(np.linalg.norm(a[i]))-1)<1e-5;done[i]=x
 start=time.perf_counter();new_count=0;limit=int(os.environ.get('CHALLENGER_INDEX_NEW_LIMIT','0'))
 order=list(range(len(texts)))
 if name=='train-documents':
  counts=read(STUDY/'results/data_audit.json')['document_counts'];order.sort(key=lambda i:(-counts[i],i))
 for i in order:
  if i in done:continue
  t=time.perf_counter()
  with torch.inference_mode():v=vector(m,tok,texts[i],budget).cpu().numpy()
  dt=time.perf_counter()-t;new_count+=1;assert np.isfinite(v).all();a[i]=v;a.flush();record={'index':i,'id':ids[i],'seconds':dt,'vector_sha256':digest(v.tobytes().hex())};append(journal,record);done[i]=record
  if len(done)%100==0:print(name,len(done),'/',len(texts),'last_s',round(dt,3),'elapsed_s',round(time.perf_counter()-start,1),flush=True)
  if limit and new_count>=limit:print('Bounded indexing pilot complete',len(done),flush=True);return a
 save(p/'complete.json',{'status':'complete','count':len(done),'array_sha256':sha(f),'active_seconds':sum(x['seconds'] for x in done.values()),'journal_sha256':sha(journal)})
 return a

def load_vectors(name):
 p=CACHE/'vectors'/name;assert read(p/'complete.json')['array_sha256']==sha(p/'vectors.npy')
 return np.load(p/'vectors.npy',mmap_mode='r'),read(p/'contract.json')['ids']

if __name__=='__main__':
 import sys
 if sys.argv[1]=='index':index(sys.argv[2] if len(sys.argv)>2 else 'train')
