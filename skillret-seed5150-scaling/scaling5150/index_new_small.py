"""Bounded offline native EG2/Jina indexes, one model at a time, batch1 FP32."""
import argparse,json,os,sys,time,hashlib,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'scaling5150';EG=Path('/home/user/code/embeddinggemma2-pilot');JINA=Path('/home/user/code/jina-nano-pilot')
def main():
 a=argparse.ArgumentParser();a.add_argument('role',choices=['eg2','jina']);a.add_argument('--pilot',action='store_true');args=a.parse_args()
 role={'eg2':'eg2_text270m_fp32','jina':'jina_nano239m_fp32'}[args.role];out=HERE/(('embedding-pilot-' if args.pilot else 'index-')+role);out.mkdir(exist_ok=False);stage_started=time.monotonic()
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1');sys.dont_write_bytecode=True
 if args.role=='eg2':sys.path.insert(0,str(EG/'overlay'))
 else:os.environ.update(HF_HOME=str(JINA/'hf-cache'),HF_MODULES_CACHE=str(JINA/'hf-modules'))
 start=time.perf_counter();import torch,numpy as np
 torch.set_num_threads(1);torch.set_num_interop_threads(1);assert torch.cuda.get_device_capability()==(7,5)
 manifest=json.loads((HERE/'manifest.json').read_text());texts=json.loads((HERE/'documents.json').read_text());source_skills=[json.loads(x) for x in (ROOT/'data/skills/test.jsonl').read_text().splitlines()];ids=[s['id'] for s in source_skills];del source_skills
 queries=manifest['queries'];ids=(ids[:16]+sorted(ids[16:],key=lambda d:(-len(texts[d]),d))[:16]) if args.pilot else ids;queries=queries[:2] if args.pilot else queries
 modelroot=EG if args.role=='eg2' else JINA;limit=512 if args.role=='eg2' else 1024
 if args.role=='eg2':
  from sentence_transformers import SentenceTransformer
  model=SentenceTransformer(str(EG/'model'),device='cuda',local_files_only=True,trust_remote_code=False,config_kwargs={'vision_config':None,'audio_config':None},model_kwargs={'dtype':torch.float32,'attn_implementation':'eager','device_map':'cuda'});model.max_seq_length=limit;native=model[0].auto_model;assert native.vision_tower is None and native.audio_tower is None
 else:
  from transformers import AutoConfig,AutoModel
  for x in json.loads((JINA/'review-manifest.json').read_text()):
   if x['file'].endswith('.py'):assert hashlib.sha256((JINA/'model'/x['file']).read_bytes()).hexdigest()==x['sha256']
  config=AutoConfig.from_pretrained(JINA/'model',trust_remote_code=True,local_files_only=True);config._attn_implementation='eager';model=AutoModel.from_pretrained(JINA/'model',config=config,trust_remote_code=True,local_files_only=True,dtype=torch.float32).to(device='cuda',dtype=torch.float32).eval();model.set_adapter(['retrieval']);native=model
 assert all(p.dtype==torch.float32 and p.device.type=='cuda' for p in native.parameters());torch.cuda.synchronize();startup=time.perf_counter()-start
 for p in (modelroot/'model').rglob('*.safetensors'):
  with p.open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 def timeout(signum,frame):raise TimeoutError('Single encode60seconds')
 signal.signal(signal.SIGALRM,timeout);torch.cuda.reset_peak_memory_stats();calls=[]
 def encode(text,query,ident):
  if args.role=='eg2':prompt='SearchQuery' if query else 'Document';prefix=model.prompts[prompt];tok=model.tokenizer
  else:prompt='query' if query else 'document';prefix='Query: ' if query else 'Document: ';tok=model.tokenizer
  raw=len(tok(prefix+text,add_special_tokens=True,truncation=False)['input_ids']);torch.cuda.synchronize();start=time.perf_counter();cpu=time.process_time();signal.alarm(60)
  try:
   if args.role=='eg2':v=model.encode([text],prompt_name=prompt,batch_size=1,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)[0]
   else:v=model.encode(texts=[text],task='retrieval',prompt_name=prompt,max_length=limit)[0].float().cpu().numpy()
   torch.cuda.synchronize()
  finally:signal.alarm(0)
  assert v.shape==(768,) and np.isfinite(v).all() and abs(float(np.linalg.norm(v))-1)<1e-4
  row={'id':ident,'query':query,'seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'raw_tokens':raw,'used_tokens':min(raw,limit),'truncated':raw>limit};calls.append(row)
  with (out/'calls.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  return v
 encode('warmup',False,'warmup')
 # Prior270document vectors are reusable only for identical text/native config.
 pilot=json.loads((EG/'fixtures/skillret-pilot.json').read_text());prev=EG/'evaluation-compatible' if args.role=='eg2' else JINA/'evaluation';old_docs=np.load(prev/'skillret-docs.npy');old_queries=np.load(prev/'skillret-queries-0.npy');dold={d['id']:(d['text'],old_docs[i]) for i,d in enumerate(pilot['documents'])};qold={q['id']:(q['query'],old_queries[i]) for i,q in enumerate(pilot['queries'])}
 pilot_path=HERE/('embedding-pilot-'+role)
 if not args.pilot and (pilot_path/'manifest.json').exists():
  pm=json.loads((pilot_path/'manifest.json').read_text());pd=np.load(pilot_path/'docs.npy');pq=np.load(pilot_path/'queries.npy')
  for i,d in enumerate(pm['document_ids']):dold[d]=(texts[d],pd[i])
  for i,qid in enumerate(pm['query_ids']):qold[qid]=(next(q['query'] for q in queries if q['id']==qid),pq[i])
 dv=np.lib.format.open_memmap(out/'docs.npy',mode='w+',dtype=np.float32,shape=(len(ids),768));qv=np.lib.format.open_memmap(out/'queries.npy',mode='w+',dtype=np.float32,shape=(len(queries),768));reused_docs=0;reused_q=0;qtimes=[];qraw=[]
 for i,d in enumerate(ids):
  if time.monotonic()-stage_started>1200:raise TimeoutError('Full index bounded at20minutes')
  if d in dold and dold[d][0]==texts[d]:dv[i]=dold[d][1];reused_docs+=1
  else:dv[i]=encode(texts[d],False,d)
  if i%100==0:dv.flush();print(role,i,len(ids),flush=True)
 dv.flush()
 for i,q in enumerate(queries):
  if q['id'] in qold and qold[q['id']][0]==q['query']:qv[i]=qold[q['id']][1];reused_q+=1;qtimes.append(None);qraw.append(None)
  else:qv[i]=encode(q['query'],True,q['id']);qtimes.append(calls[-1]['seconds']);qraw.append(calls[-1]['raw_tokens'])
 qv.flush();info={'role':role,'view_sha256':manifest['text_sha256'],'document_ids':ids,'query_ids':[q['id'] for q in queries],'startup_seconds':startup,'document_forward_seconds':sum(x['seconds'] for x in calls if not x['query'] and x['id']!='warmup'),'query_seconds':qtimes,'query_raw_tokens':qraw,'reused_document_vectors':reused_docs,'reused_query_vectors':reused_q,'cache_timing_note':'Reused vectors have no fresh forward latency; null query timings are not zero-cost inference. Full indexing cost is incomplete when prior vectors reused.','new_forward_calls':len(calls)-1,'dtype':'FP32','attention':'eager','batch':1,'max_tokens':limit,'parameters':sum(p.numel() for p in native.parameters()),'max_allocated_vram_bytes':torch.cuda.max_memory_allocated(),'max_reserved_vram_bytes':torch.cuda.max_memory_reserved(),'truncated_new_inputs':sum(x['truncated'] for x in calls),'model_manifest_sha256':hashlib.sha256((modelroot/'model-manifest.json').read_bytes()).hexdigest()};(out/'manifest.json').write_text(json.dumps(info,indent=2));print(json.dumps({k:v for k,v in info.items() if k not in ['document_ids','query_ids']}))
if __name__=='__main__':main()
