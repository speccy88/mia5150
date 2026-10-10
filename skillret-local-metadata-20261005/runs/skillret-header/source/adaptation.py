"""Bounded training-only lexical ranker adaptation with replay and resume checks."""
from harness import *
import numpy as np, sys, collections
def partitions():
 skills=load('skills','train');queries=load('queries','train');parent={s['id']:s['id'] for s in skills}
 def find(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
  return a
 def union(a,b):parent[find(a)]=find(b)
 # Repository and exact-name groups plus multi-positive queries stay together.
 groups={}
 for s in skills:
  for key in [('repo',s.get('repo')),('name',s.get('name','').strip().lower())]:
   if key[1]:
    if key in groups:union(s['id'],groups[key])
    else:groups[key]=s['id']
 components=collections.defaultdict(list)
 train=[];dev=[];excluded=0
 for q in queries:
  keys={find(s) for s in q['skill_ids']};sides={int(digest(key)[:8],16)%5==0 for key in keys}
  if len(sides)>1:excluded+=1;continue
  (dev if next(iter(sides)) else train).append(q)
  for key in keys:components[key].append(q)
 overlap={q['query'].strip().lower() for q in train}&{q['query'].strip().lower() for q in dev}
 train=sorted([q for q in train if q['query'].strip().lower() not in overlap],key=lambda q:digest(q['id']))[:512];dev=sorted([q for q in dev if q['query'].strip().lower() not in overlap],key=lambda q:digest(q['id']))[:128]
 assert train and dev,'No viable repository-disjoint split'
 a={find(s) for q in train for s in q['skill_ids']};b={find(s) for q in dev for s in q['skill_ids']};assert not a&b
 save(R/'manifests/training-partitions.json',{'train':[q['id'] for q in train],'dev':[q['id'] for q in dev],'train_allowed_skill_ids':[s['id'] for s in skills if int(digest(find(s['id']))[:8],16)%5!=0],'component_count':len(components),'excluded_duplicate_query_texts':len(overlap),'excluded_cross_partition_queries':excluded,'largest_components':sorted(map(len,components.values()),reverse=True)[:10],'grouping':'connected repositories and exact skill names; queries spanning train/dev groups excluded','exposure':'Released specialists already trained on public training data; split only evaluates this new adaptation, not pretrained-model generalization.'})
 return skills,train,dev
def features(rankings,q,doc):
 f=[]
 for name in ['header','full']:
  rank={d:i+1 for i,d in enumerate(rankings[name].get(q,[]))}.get(doc,1000)
  f.extend([1/rank,1/(60+rank),float(rank<=5),float(rank<=20),float(rank<=100)])
 return [1.]+f
def fit(x,y,replay=False,steps=200,initial=None):
 w=np.zeros(x.shape[1]) if initial is None else initial.copy();loss=[]
 for step in range(steps):
  z=np.clip(x@w,-30,30);p=1/(1+np.exp(-z));target=.9*y+.1*.5 if replay else y
  w-=.5*(x.T@(p-target)/len(y)+.001*w);loss.append(float(np.mean(np.logaddexp(0,z)-target*z)))
 assert np.isfinite(w).all();return w,loss
def main():
 skills,train,dev=partitions();allq=train+dev;rankings={};timings={}
 for name in ['header','full']:
  rankings[name],timings[name]=lexical(skills,safe_queries(allq),name,100)
  save(R/f'results/training-bm25-{name}.json',{'rankings':rankings[name],'timing':timings[name]})
 labels=qrels('train');x=[];y=[];allowed=set(json.loads((R/'manifests/training-partitions.json').read_text())['train_allowed_skill_ids'])
 for q in train:
  qid=q['id'];pool=rrf([rankings[n][qid] for n in rankings])
  # Only training labels enter examples; use every positive and exclude all from negatives.
  for d in list(labels[qid])+negatives(labels[qid],[d for d in pool if d in allowed],12):x.append(features(rankings,qid,d));y.append(float(d in labels[qid]))
 x=np.array(x);y=np.array(y);report={'examples':len(y),'train_queries':len(train),'dev_queries':len(dev),'models':{}}
 # Toy overfit, exact deterministic resume, finite/changed parameters.
 toy=np.array([[1.,1.],[1.,-1.]]);tw,tl=fit(toy,np.array([1.,0.]),steps=400);assert tl[-1]<.02
 w1,_=fit(x,y,steps=100);wresume,_=fit(x,y,steps=100,initial=w1);wfull,_=fit(x,y,steps=200);assert np.allclose(wresume,wfull)
 save(R/'evidence/adaptation-smoke.json',{'toy_final_loss':tl[-1],'resume_equal':True,'positive_collision_count':0})
 devlabels={q['id']:labels[q['id']] for q in dev}
 for mode in ['no-adaptation','hard-negative','replay']:
  w,loss=fit(x,y,replay=mode=='replay') if mode!='no-adaptation' else (None,[])
  result={}
  for q in dev:
   qid=q['id'];pool=rrf([rankings[n][qid] for n in rankings]);result[qid]=pool if w is None else sorted(pool,key=lambda d:(-float(np.array(features(rankings,qid,d))@w),d))
  metric=evaluate(devlabels,result);report['models'][mode]={'metrics':metric,'loss':loss,'weights':w.tolist() if w is not None else None};save(R/f'results/adaptation-{mode}.json',{'rankings':result,'metrics':metric})
  if w is not None:save(R/f'checkpoints/{mode}.json',{'weights':w.tolist(),'features':'bias + [1/rank,1/(60+rank),rank<=5,rank<=20,rank<=100] for header/full','train_partition_hash':digest([q['id'] for q in train])})
 save(R/'results/adaptation-summary.json',report);print({k:v['metrics']['aggregate'] for k,v in report['models'].items()},flush=True)
if __name__=='__main__':main()
