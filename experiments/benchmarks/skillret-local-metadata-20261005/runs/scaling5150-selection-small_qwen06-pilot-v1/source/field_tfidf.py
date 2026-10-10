"""Field-aware TF-IDF ablation following S11's cosine field combination."""
from harness import *
from collections import Counter
import re,numpy as np,scipy.sparse as sp,time
def tokenize(t):return re.findall(r'(?u)\b\w\w+\b',t.lower())
def matrix(texts,vocab=None,idf=None):
 counts=[Counter(tokenize(t)) for t in texts]
 if vocab is None:vocab={w:i for i,w in enumerate(sorted({w for c in counts for w in c}))}
 row=[];col=[];val=[]
 for i,c in enumerate(counts):
  for w,n in c.items():
   if w in vocab:row.append(i);col.append(vocab[w]);val.append(n)
 x=sp.csr_matrix((np.array(val,dtype=np.float32),(row,col)),shape=(len(texts),len(vocab)))
 if idf is None:idf=np.log((len(texts)+1)/(np.array((x>0).sum(0)).ravel()+1))+1
 x=x.multiply(idf).tocsr();norm=np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel());x=sp.diags(1/np.maximum(norm,1e-12))@x
 return x,vocab,idf
skills=load('skills','test');queries=json.loads((R/'manifests/scorer-queries.json').read_text());labels=json.loads((R/'manifests/evaluator-qrels.json').read_text());combined=np.zeros((len(queries),len(skills)));ids=[s['id'] for s in skills];statistics={}
for field in ['name','description','skill_md','flattened']:
 start=time.perf_counter();texts=[view(s,'full') if field=='flattened' else (s.get(field) or '') for s in skills];x,vocab,idf=matrix(texts);q,_,_=matrix([q['query'] for q in queries],vocab,idf);scores=(q@x.T).toarray();rankings={qu['id']:[ids[i] for i in np.argsort(-scores[j],kind='stable')[:100] if scores[j,i]>0] for j,qu in enumerate(queries)};statistics[field]={'seconds':time.perf_counter()-start,'vocabulary':len(vocab)}
 save(R/f'results/tfidf-{field}.json',{'rankings':rankings,'metrics':evaluate(labels,rankings),'timing':statistics[field]})
 if field!='flattened':combined+=scores/3
rankings={qu['id']:[ids[i] for i in np.argsort(-combined[j],kind='stable')[:100] if combined[j,i]>0] for j,qu in enumerate(queries)};save(R/'results/tfidf-fields.json',{'rankings':rankings,'metrics':evaluate(labels,rankings),'fields':statistics,'note':'Sparse field-cosine uniform ablation. Not a full six-channel dense+sparse MLP reproduction.'});print(evaluate(labels,rankings)['aggregate'])
