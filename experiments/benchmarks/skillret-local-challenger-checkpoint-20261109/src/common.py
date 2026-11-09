"""Immutable inputs, atomic artifacts, and exact-ID evaluation for the challenger."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import hashlib, json, math, importlib.util
from pathlib import Path
import numpy as np
STUDY=Path(os.environ.get('SKILLRET_STUDY_ROOT',Path(__file__).resolve().parents[1])).resolve()
SRC=Path(os.environ.get('SKILLRET_EXECUTION_SOURCE',STUDY/'src')).resolve()
REPO=STUDY.parents[2]
CACHE=Path(os.environ.get('SKILLRET_CHALLENGER_CACHE',Path.home()/'.cache/mia-local-challenger'))
OLD=STUDY.parent/'skillret-embedding-jev-fusion'
_legacy_data=Path.home()/'.cache/mia-bge-lora/data'
TRAIN_CACHE=Path(os.environ.get('SKILLRET_TRAIN_CACHE',_legacy_data if _legacy_data.exists() else CACHE/'data'))
PREFIX='Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: '
DATA_REV='6583d7d2ed07644d0fb8938ed8178f3a7dc42a12'
MODEL_REV='67f1594c46c40e13a5eb2f4092cc85c98c9ed32d'
SCALING_SIZES=(50,100,250,500,1000,2000,5000,6006)
SCALING_PANEL='experiments/benchmarks/skill-discovery-oauth-v1-20261103/data/test_queries.json'
SCALING_POOLS='experiments/benchmarks/skillret-qwen3-embedding-8b-q4km-fulltext/results/candidate_pools.json'
def scaling_rerankers(primary):
 specs={'released-reranker':('base',False),'trained-reranker':('base',True)}
 if primary=='trained-combined':specs.update({'released-combined':('mixture-query',False),'trained-combined':('mixture-query',True)})
 return specs
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x):return hashlib.sha256((x if isinstance(x,str) else json.dumps(x,sort_keys=True)).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def record_path(p):
 p=Path(p)
 return p if p.exists() or p.suffix=='.gz' else Path(str(p)+'.gz')
def records_exist(p):return record_path(p).is_file()
def iter_rows(p):
 import gzip
 p=record_path(p);opener=gzip.open if p.suffix=='.gz' else open
 with opener(p,'rt',encoding='utf-8') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def rows(p):return list(iter_rows(p))
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def write_rows(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
 with t.open('w') as f:
  for r in x:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 t.replace(p)
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def data(split):
 prefix='train-' if split=='train' else ''
 s,q,r=[rows(TRAIN_CACHE/(prefix+k+'.jsonl')) for k in ('skills','queries','qrels')]
 gold={x['id']:set() for x in q};ids={x['id'] for x in s};assert len(ids)==len(s) and len(gold)==len(q)
 assert all('/' not in sid and '\\' not in sid for sid in ids)
 for x in r:
  assert x['query_id'] in gold and x['skill_id'] in ids and x['relevance']==1
  assert x['skill_id'] not in gold[x['query_id']];gold[x['query_id']].add(x['skill_id'])
 assert all(gold.values())
 for x in q:assert set(x['skill_ids'])==gold[x['id']]
 return s,q,gold

def fulltext(s):return ' | '.join((s.get(k) or '').strip() for k in ('name','description','skill_md'))
def evaluate(gold,rankings):
 per={}
 for q,g in gold.items():
  r=rankings.get(q,[]);assert len(r)==len(set(r));n=len(g);z={}
  for k in (5,10,15):
   dcg=sum((s in g)/math.log2(i+2) for i,s in enumerate(r[:k]));ideal=sum(1/math.log2(i+2) for i in range(min(k,n)));z[f'NDCG@{k}']=dcg/ideal
  for k in (1,5,10,20,50):
   hit=sum(s in g for s in r[:k]);z[f'Recall@{k}']=hit/n
   if k in (5,10):
    z[f'Precision@{k}']=hit/k;z[f'Completeness@{k}']=float(hit==n);seen=0;ap=0
    for i,s in enumerate(r[:k],1):
     if s in g:seen+=1;ap+=seen/i
    z[f'MAP@{k}']=ap/n;z[f'MRR@{k}']=next((1/i for i,s in enumerate(r[:k],1) if s in g),0.)
  z['Hit@1']=float(bool(r) and r[0] in g);per[q]=z
 return {'aggregate':{k:math.fsum(x[k] for x in per.values())/len(per) for k in next(iter(per.values()))},'per_query':per,'query_count':len(per),'missing_query_count':len(set(gold)-set(rankings))}
def retrieve(v,docs,ids,k=50):
 scores=np.asarray(docs@v,dtype=np.float32);order=np.argsort(-scores,kind='stable')[:k]
 return [ids[i] for i in order],[float(scores[i]) for i in order]
def mix_query_vectors(base,adapted,weight):
 if weight==0:return base
 if weight==1:return adapted
 assert 0<weight<1
 mixed=(1-weight)*base+weight*adapted
 return mixed/np.linalg.norm(mixed,axis=-1,keepdims=True)
def append(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
def summary(vals):return {'min':min(vals),'median':float(np.median(vals)),'p95':float(np.percentile(vals,95)),'max':max(vals),'total':sum(vals)}
