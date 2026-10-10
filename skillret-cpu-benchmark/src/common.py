"""Shared, label-independent retrieval and separately evaluated IR metrics."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
os.environ['TOKENIZERS_PARALLELISM']='false'
import json,csv,re,time,hashlib,platform,sys,resource
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/processed';RESULTS=ROOT/'results';RESULTS.mkdir(exist_ok=True)
METRICS=['mrr5','recall1','recall5','hit1','hit5','ndcg5','complete5','candidate_recall','candidate_hit']
def readjl(p):
 with Path(p).open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def writecsv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ms(t):return (time.perf_counter()-t)*1000
def hw():
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:4])
 return {'cpu':next((x.split(':',1)[1].strip() for x in open('/proc/cpuinfo') if x.startswith('model name')),'unknown'),'visible_cpus':os.cpu_count(),'affinity':allowed[:4],'threads':4,'python':sys.version,'platform':platform.platform(),'device':'CPU'}
def metrics(ids,gold):
 gold=set(gold);rel=[int(x in gold) for x in ids[:5]];first=next((i+1 for i,r in enumerate(rel) if r),None);ideal=sum(1/np.log2(i+2) for i in range(min(5,len(gold))))
 return {'mrr5':1/first if first else 0.,'recall1':(rel[0] if rel else 0)/len(gold),'recall5':sum(rel)/len(gold),'hit1':float(bool(rel and rel[0])),'hit5':float(any(rel)),'ndcg5':sum(r/np.log2(i+2) for i,r in enumerate(rel))/ideal,'complete5':float(sum(rel)==len(gold)),'candidate_recall':len(set(ids)&gold)/len(gold),'candidate_hit':float(bool(set(ids)&gold))}
def rank(pool,scores,k=20):return pool[np.argsort(-np.asarray(scores),kind='stable')[:k]]
def rrf(pool,bm,vec):
 s=np.zeros(len(pool),dtype=np.float32)
 for a in [bm,vec]:
  order=np.argsort(-a,kind='stable')[:100];s[order]+=1/(60+np.arange(1,len(order)+1))
 return rank(pool,s)
class BM25:
 def __init__(self,texts):
  t=time.perf_counter();self.vocab={};lengths=[];row=[];col=[];vals=[]
  for i,text in enumerate(texts):
   counts=Counter(re.findall(r'\w+',text.casefold(),flags=re.UNICODE));lengths.append(sum(counts.values()))
   for word,count in counts.items():
    if word not in self.vocab:self.vocab[word]=len(self.vocab)
    row.append(i);col.append(self.vocab[word]);vals.append(count)
  self.tf=sparse.csr_matrix((vals,(row,col)),shape=(len(texts),len(self.vocab)),dtype=np.float32);self.lengths=np.array(lengths,dtype=np.float32);self.seconds=time.perf_counter()-t
 def score(self,text,pool):
  terms=[self.vocab[x] for x in re.findall(r'\w+',text.casefold(),flags=re.UNICODE) if x in self.vocab]
  if not terms:return np.zeros(len(pool))
  tf=self.tf[pool][:,terms].toarray();df=(tf>0).sum(0);idf=np.log(1+(len(pool)-df+.5)/(df+.5));dl=self.lengths[pool];norm=1.2*(1-.75+.75*dl/max(dl.mean(),1e-9));return (tf*2.2/(tf+norm[:,None])*idf).sum(1)
class Corpus:
 def __init__(self):
  self.skills=readjl(DATA/'skills.jsonl');self.queries=readjl(DATA/'queries.jsonl');self.ids=[x['skill_id'] for x in self.skills];assert self.ids==sorted(self.ids)
  self.index={s:i for i,s in enumerate(self.ids)};self.texts=[s['text'] for s in self.skills];self.full=np.arange(len(self.ids),dtype=np.int32);self.sizes=[10,100,1000,len(self.ids)]
  self.pools={r['query_id']:np.array([self.index[x] for x in r['candidate_skill_ids']],dtype=np.int32) for r in readjl(DATA/'catalogs.jsonl')}
  self.major=np.array([s['major'] for s in self.skills]);self.bm=BM25(self.texts);self.latids=json.loads((DATA/'latency_query_ids.json').read_text())[:32]
 def pool(self,qid,n):return self.full if n==len(self.full) else np.sort(self.pools[qid][:n])
 def route(self,emb,pool):
  labels=np.unique(self.major[pool]);groups=[pool[self.major[pool]==x] for x in labels];centroids=np.array([emb[g].mean(0) for g in groups]);centroids/=np.maximum(np.linalg.norm(centroids,axis=1,keepdims=True),1e-12);return labels,groups,centroids
def summarize(rows,latency):
 out=[]
 for model,method,size in sorted({(r['model'],r['method'],r['catalog_size']) for r in rows}):
  rs=[r for r in rows if (r['model'],r['method'],r['catalog_size'])==(model,method,size)];ls=[r for r in latency if (r['model'],r['method'],r['catalog_size'])==(model,method,size)];v=[r['total_ms'] for r in ls]
  out.append({'model':model,'method':method,'catalog_size':size,'queries':len(rs),**{m:float(np.mean([r[m] for r in rs])) for m in METRICS},'latency_queries':len({r['query_id'] for r in ls}),'latency_repeats':len({r['repeat'] for r in ls}), 'p50_ms':float(np.percentile(v,50)) if v else None,'p95_ms':float(np.percentile(v,95)) if v else None,'mean_ms':float(np.mean(v)) if v else None,'serial_qps':1000/float(np.mean(v)) if v else None})
 return out
