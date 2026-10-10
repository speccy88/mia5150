"""Pinned full-text SKILLRET baseline. Only cached local assets; no API credentials."""
import argparse, hashlib, importlib.metadata, json, os, pathlib, platform, subprocess, time
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=pathlib.Path(__file__).resolve().parent
PREFIX='Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: '
DATA_REV='6583d7d2ed07644d0fb8938ed8178f3a7dc42a12'
MODEL_REV='67f1594c46c40e13a5eb2f4092cc85c98c9ed32d'
CODE_REV='0cc508e6bfcfff1fe6352d90ff5957ba38a64bba'
HASHES={'skills':'1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81','queries':'4d07a35f40e597f8468c6dab480c95a1f9b40ecbde80274081ba2e7c643d0f6d','qrels':'edbfae0b8109de460d54e27e6235b1b47316f34956744db0571ad1b4ddb860af'}
TARGETS={'NDCG':[.7902,.8112,.8187],'Recall':[.8238,.8774,.9016],'Completeness':[.7083,.7894,.8265],'MAP':[.7409,.7538,.7570],'Precision':[.2591,.1401,.0962]}

def save(path,value):
    path=pathlib.Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def text_digest(texts):
    digest=hashlib.sha256()
    for text in texts:
        encoded=text.encode('utf-8');digest.update(len(encoded).to_bytes(8,'big'));digest.update(encoded)
    return digest.hexdigest()

def load_assets(base):
    rows={}
    for subset in HASHES:
        path=base/'data'/subset/'test.jsonl'
        assert sha(path)==HASHES[subset],f'Pinned {subset} hash differs'
        with path.open() as f:rows[subset]=[json.loads(s) for s in f if s.strip()]
    skills,queries,qrels=(rows[k] for k in ('skills','queries','qrels'))
    assert tuple(map(len,(skills,queries,qrels)))==(6006,4392,7187)
    assert len({s['id'] for s in skills})==6006 and len({q['id'] for q in queries})==4392
    assert all(s.get('skill_md','').strip() for s in skills),'Full text required; no description fallback'
    skill_ids={s['id'] for s in skills};query_ids={q['id'] for q in queries};labels={q:{} for q in query_ids}
    for r in qrels:
        assert r['skill_id'] in skill_ids and r['query_id'] in query_ids and r['relevance']==1
        assert r['skill_id'] not in labels[r['query_id']]
        labels[r['query_id']][r['skill_id']]=1
    assert all(labels.values())
    for q in queries:
        if 'skill_ids' in q:assert set(q['skill_ids'])==set(labels[q['id']])
    model_path=base/'models/skillret_embed_06'
    manifest=json.loads((model_path/'manifest.json').read_text());assert manifest['revision']==MODEL_REV
    for name,meta in manifest['files'].items():assert sha(model_path/name)==meta['sha256'],f'Model integrity: {name}'
    texts=[' | '.join((s.get(k) or '').strip() for k in ('name','description','skill_md')) for s in skills]
    # Only canonical query text reaches encoder. qrels remain evaluator-only.
    query_texts=[PREFIX+q['query'] for q in queries]
    return skills,queries,labels,texts,query_texts,model_path,manifest

def protocol(manifest,max_length):
    import inspect
    return {'model':'ThakiCloud/SKILLRET-Embedding-0.6B','model_revision':MODEL_REV,'dataset':'ThakiCloud/SKILLRET','dataset_revision':DATA_REV,'official_code_revision':CODE_REV,'test_file_sha256':HASHES,'model_files':manifest['files'],'document_format':'stripped name | stripped description | stripped skill_md','query_prefix':PREFIX,'max_seq_length':max_length,'tokenizer_add_special_tokens':True,'truncation_side':'right','pooling':'lasttoken','padding_side':'left','normalize_embeddings':True,'similarity':'dot product of unit float32 vectors','weight_dtype':'float16','normalization_dtype':'float32','attention':'sdpa; explicit repeat_kv; memory-efficient CUDA kernel; math fallback disabled','use_cache':False,'batch_size':1,'top_k':50,'ties':'catalogue source order, stable descending sort','cpu_threads':1,'encoder_source_sha256':hashlib.sha256((inspect.getsource(load_model)+inspect.getsource(encode_one)).encode()).hexdigest(),'runtime':{n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','tokenizers','accelerate','safetensors']}}

def token_audit(tok,texts,qtexts,max_length):
    counts=[]
    for i,text in enumerate(texts):
        counts.append(len(tok.encode(text,add_special_tokens=True,truncation=False)))
        if i%1000==0:print('token audit',i,len(texts),flush=True)
    qc=[len(tok.encode(t,add_special_tokens=True,truncation=False)) for t in qtexts]
    import numpy as np
    def stats(values):return {'min':min(values),'median':float(np.median(values)),'p95':float(np.percentile(values,95)),'max':max(values),'total':sum(values),'over_8192':sum(n>8192 for n in values),'over_32768':sum(n>32768 for n in values),'truncated_count':sum(n>max_length for n in values),'effective_total':sum(min(n,max_length) for n in values)}
    return {'documents':stats(counts),'queries':stats(qc),'document_raw_tokens':counts,'query_raw_tokens':qc,'max_seq_length':max_length,'truncation':'right, tokenizer token limit; full texts retained in source'}

def load_model(path):
    import torch
    from transformers import AutoModel,AutoConfig
    from transformers.integrations import sdpa_attention
    # Transformers 4.57 selects native GQA on torch>=2.5, but SM75 efficient
    # attention needs equal head counts. Its own repeat_kv branch is equivalent.
    sdpa_attention.use_gqa_in_sdpa=lambda attention_mask,key:False
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.80)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_math_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(True)
    started=time.perf_counter()
    cfg=AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    # Saved Transformers 5 configuration nests theta; Transformers 4 otherwise
    # silently supplies Qwen3's default 10000 instead of the trained 1000000.
    raw_config=json.loads((path/'config.json').read_text())
    rp=raw_config['rope_parameters'];assert rp=={'rope_theta':1000000,'rope_type':'default'}
    cfg.rope_theta=rp['rope_theta'];cfg.rope_scaling=None
    model=AutoModel.from_pretrained(path,config=cfg,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True,device_map='cuda').eval()
    expected=1.0/(1000000**(torch.arange(0,128,2,dtype=torch.float32,device='cuda')/128))
    assert torch.allclose(model.rotary_emb.inv_freq.float(),expected,rtol=1e-5,atol=1e-8),'Effective rotary frequencies differ from pinned model'
    for p in path.glob('*.safetensors'):
        with p.open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    return model,time.perf_counter()-started

def encode_one(model,tok,text,max_length):
    import torch
    started=time.perf_counter();inputs=tok(text,truncation=True,max_length=max_length,return_tensors='pt').to('cuda')
    with torch.inference_mode():
        hidden=model(**inputs,use_cache=False).last_hidden_state[:,-1].float()
        vector=torch.nn.functional.normalize(hidden,p=2,dim=1)[0].cpu().numpy()
    assert bool(__import__('numpy').isfinite(vector).all()),'nonfinite embedding'
    assert abs(float(__import__('numpy').linalg.norm(vector))-1)<1e-5
    return vector,time.perf_counter()-started

def cached_encode(model,tok,texts,cache,kind,max_length,order=None,only=None,cooldown=0.0):
    import numpy as np
    array_path=cache/(kind+'.npy');journal_path=cache/(kind+'-journal.jsonl')
    if journal_path.exists() and not array_path.exists():raise RuntimeError('Cache journal exists without vector array; refusing unsafe resume')
    if array_path.exists():vectors=np.lib.format.open_memmap(array_path,mode='r+')
    else:vectors=np.lib.format.open_memmap(array_path,mode='w+',dtype='float32',shape=(len(texts),1024))
    assert vectors.shape==(len(texts),1024) and vectors.dtype==np.float32,'Cache shape/dtype mismatch'
    done={}
    if journal_path.exists():
        for line in journal_path.read_text().splitlines():
            row=json.loads(line);index=row['index']
            if not isinstance(index,int) or isinstance(index,bool) or not 0<=index<len(texts) or index in done:raise RuntimeError('Cache journal index invalid or duplicated')
            v=vectors[index]
            if not np.isfinite(v).all() or abs(float(np.linalg.norm(v))-1)>1e-5 or hashlib.sha256(v.tobytes()).hexdigest()!=row['sha256']:raise RuntimeError(f'Cache integrity failure at {kind} {index}')
            if not isinstance(row['seconds'],(int,float)) or not 0<=row['seconds']<float('inf'):raise RuntimeError('Invalid cached timing')
            done[index]=row
    indices=list(range(len(texts))) if order is None else order
    if only is not None:indices=indices[:only]
    with journal_path.open('a') as journal:
        for index in indices:
            if index in done:continue
            v,seconds=encode_one(model,tok,texts[index],max_length);vectors[index]=v;vectors.flush()
            row={'index':index,'seconds':seconds,'utc':time.time(),'sha256':hashlib.sha256(v.tobytes()).hexdigest()}
            journal.write(json.dumps(row)+'\n');journal.flush();os.fsync(journal.fileno());done[index]=row
            if len(done)%50==0 or len(done)<5:print(kind,'cached',len(done),'/',len(texts),'last seconds',round(seconds,3),flush=True)
            if cooldown:time.sleep(cooldown)
    return vectors,done

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=pathlib.Path,default=pathlib.Path('/home/user/code/mia5150-local'));p.add_argument('--output',type=pathlib.Path,default=ROOT);p.add_argument('--stage',choices=('audit','feasibility','full'),default='audit');p.add_argument('--max-length',type=int,default=32768);p.add_argument('--forward-cooldown',type=float,default=.05,help='Safety pause after new forwards; excluded from active encoding latency and cache identity')
    a=p.parse_args();out=a.output;out.mkdir(exist_ok=True,parents=True)
    if not 0<=a.forward_cooldown<=10:raise ValueError('Invalid forward cooldown')
    if a.stage!='audit' and os.environ.get('SKILLRET_RESOURCE_GUARD')!='1':raise RuntimeError('Run through resource_guard.py with fresh host telemetry')
    skills,queries,labels,texts,qtexts,mp,manifest=load_assets(a.base)
    prot=protocol(manifest,a.max_length)
    prot.update(document_text_sha256=text_digest(texts),query_text_sha256=text_digest(qtexts),effective_rope_theta=1000000)
    contract=hashlib.sha256(json.dumps(prot,sort_keys=True).encode()).hexdigest();cache=out/'cache'/contract;cache.mkdir(parents=True,exist_ok=True)
    save(out/'protocol.json',prot);save(cache/'protocol.json',prot)
    env={'python':platform.python_version(),'platform':platform.platform(),'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','tokenizers','accelerate','safetensors','pytrec-eval-terrier']}}
    save(out/'environment.json',env)
    panel=sorted([q['id'] for q in queries],key=lambda q:(hashlib.sha256(q.encode()).hexdigest(),q))[:200]
    save(out/'results/pilot_200_ids.json',panel);save(out/'results/smoke_10_ids.json',panel[:10])
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(mp,local_files_only=True,trust_remote_code=False,padding_side='left')
    tok.truncation_side='right'
    audit_path=cache/'token_audit.json'
    if audit_path.exists():audit=json.loads(audit_path.read_text())
    else:audit=token_audit(tok,texts,qtexts,a.max_length);save(audit_path,audit)
    save(out/'results/token_audit.json',audit)
    print(json.dumps({'cache':str(cache),'documents':audit['documents'],'queries':audit['queries']}),flush=True)
    if a.stage=='audit':return
    model,startup=load_model(mp)
    # Longest first makes fit failures visible before an expensive catalogue pass.
    order=sorted(range(len(texts)),key=lambda i:(-audit['document_raw_tokens'][i],i))
    vectors,doc_done=cached_encode(model,tok,texts,cache,'documents',a.max_length,order,3 if a.stage=='feasibility' else None,cooldown=a.forward_cooldown)
    if a.stage=='feasibility':
        import torch
        save(out/'results/feasibility.json',{'status':'passed','document_indices':order[:3],'raw_tokens':[audit['document_raw_tokens'][i] for i in order[:3]],'max_length':a.max_length,'startup_seconds':startup,'documents_cached':len(doc_done),'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2});return
    qvectors,q_done=cached_encode(model,tok,qtexts,cache,'queries',a.max_length,cooldown=a.forward_cooldown)
    assert len(doc_done)==6006 and len(q_done)==4392
    import numpy as np, torch
    memory={'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'torch_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
    del model;torch.cuda.empty_cache()
    rankings={};scores_by_q={};latencies={};ids=[s['id'] for s in skills]
    dense_path=out/'results/dense_top50.jsonl'
    with dense_path.open('w') as f:
        for i,q in enumerate(queries):
            t=time.perf_counter();scores=vectors@qvectors[i];order=np.argsort(-scores,kind='stable')[:50];search=time.perf_counter()-t
            r=[ids[j] for j in order];score=[float(scores[j]) for j in order];rankings[q['id']]=r;scores_by_q[q['id']]=score
            latencies[q['id']]={'query_encode_seconds':q_done[i]['seconds'],'dot_search_seconds':search,'total_seconds':q_done[i]['seconds']+search}
            f.write(json.dumps({'query_id':q['id'],'candidates':[{'skill_id':s,'score':v} for s,v in zip(r,score)]})+'\n')
    from evaluate import evaluate
    ev=evaluate(labels,rankings);agg=ev['aggregate']
    deltas={f'{m}@{k}':agg[f'{m}@{k}']-target for m,vals in TARGETS.items() for k,target in zip((5,10,15),vals)}
    max_dev=max(abs(deltas[f'NDCG@{k}']) for k in (5,10,15))
    gate='validated' if max_dev<=.005 and a.max_length==32768 else 'investigation_required'
    latency_values=[r['total_seconds'] for r in latencies.values()]
    result={'status':'completed','scope':'full_4392_queries_6006_fulltext_skills','baseline_gate':gate,'max_abs_ndcg_deviation':max_dev,'official_targets':TARGETS,'target_deltas':deltas,'evaluation':ev,'rankings':rankings,'scores':scores_by_q,'protocol':prot,'cache_contract_sha256':contract,'startup_seconds_this_run':startup,'document_encoding_seconds':sum(v['seconds'] for v in doc_done.values()),'latencies':latencies,'retrieval_median_seconds':float(np.median(latency_values)),'retrieval_p95_seconds':float(np.percentile(latency_values,95)),'memory':memory,'jev_executed':False}
    result['execution_safety']={'forward_cooldown_seconds_final_run':a.forward_cooldown,'latency_definition':'active query tokenization + GPU forward/normalization/transfer + CPU dot search; excludes inter-forward safety cooldown, cache flush, startup, and document indexing','wall_time_evidence':'runs/*/telemetry.jsonl; includes guarded stopped attempts separately'}
    save(out/'results/baseline_skillret_06b.json',result)
    save(out/'results/baseline_gate.json',{'status':gate,'max_abs_ndcg_deviation':max_dev,'threshold':.005,'full_query_count':4392,'full_skill_count':6006,'contract_sha256':contract})
    print(json.dumps({'status':gate,'aggregate':agg,'target_deltas':deltas}),flush=True)

if __name__=='__main__':main()
