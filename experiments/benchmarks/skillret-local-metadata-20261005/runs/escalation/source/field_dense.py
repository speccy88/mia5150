"""Six-channel uniform field cosine: three TF-IDF + three Qwen channels."""
from harness import *
from neural import Embed
import numpy as np,time
from collections import Counter
import scipy.sparse as sp,re

def sparse(texts,vocab=None,idf=None):
 counts=[Counter(re.findall(r'(?u)\b\w\w+\b',t.lower())) for t in texts]
 if vocab is None:vocab={w:i for i,w in enumerate(sorted({w for c in counts for w in c}))}
 rows=[];cols=[];vals=[]
 for i,c in enumerate(counts):
  for w,v in c.items():
   if w in vocab:rows.append(i);cols.append(vocab[w]);vals.append(v)
 x=sp.csr_matrix((np.asarray(vals,dtype=np.float32),(rows,cols)),shape=(len(texts),len(vocab)))
 if idf is None:idf=np.log((len(texts)+1)/(np.asarray((x>0).sum(0)).ravel()+1))+1
 x=x.multiply(idf).tocsr();norm=np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel());return sp.diags(1/np.maximum(norm,1e-12))@x,vocab,idf

queries=json.loads((R/'manifests/scorer-queries.json').read_text());model=Embed('qwen_embed_06');qvec,qstats=model.encode([q['query'] for q in queries],True,1);combined=None;channels={}
for field in ['name','description','skill_md']:
 ids=[];texts=[]
 with (R/'data/skills/test.jsonl').open() as f:
  for line in f:
   row=json.loads(line);ids.append(row['id']);texts.append(row.get(field) or '')
 cached=R/f'cache/field-{field}.npy';start=time.perf_counter()
 if cached.exists():vectors=np.load(cached);stats={'reused_cache':True}
 else:vectors,stats=model.encode(texts,batch=1);np.save(cached,vectors)
 dense=qvec@vectors.T;x,vocab,idf=sparse(texts);q,_,_=sparse([q['query'] for q in queries],vocab,idf);lex=(q@x.T).toarray();score=(dense+lex)/6
 combined=score if combined is None else combined+score
 rankings={qu['id']:[ids[i] for i in np.argsort(-dense[j],kind='stable')[:100]] for j,qu in enumerate(queries)}
 channels[field]={'seconds':time.perf_counter()-start,'document_stats':stats};save(R/f'results/field-dense-{field}.json',{'rankings':rankings,'metrics':evaluate(json.loads((R/'manifests/evaluator-qrels.json').read_text()),rankings),'statistics':channels[field]})
 del x,q,texts,vectors,dense,lex
rankings={qu['id']:[ids[i] for i in np.argsort(-combined[j],kind='stable')[:100]] for j,qu in enumerate(queries)}
save(R/'results/field-six-uniform.json',{'rankings':rankings,'metrics':evaluate(json.loads((R/'manifests/evaluator-qrels.json').read_text()),rankings),'channels':channels,'note':'S11 six-channel uniform cosine method adaptation to corrected dataset; Qwen FP16 native query prompt, each field capped1024, not original older-pool result or learned MLP.'})
