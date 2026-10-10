"""Correctly labelled anchoring control; reuses frozen training retrieval."""
from adaptation import *
skills,train,dev=partitions();labels=qrels('train');rankings={n:json.loads((R/f'results/training-bm25-clean-{n}.json').read_text())['rankings'] for n in ['header','full']};x=[];y=[];allowed=set(json.loads((R/'manifests/training-partitions.json').read_text())['train_allowed_skill_ids'])
for q in train:
 qid=q['id'];pool=rrf([rankings[n][qid] for n in rankings])
 for d in list(labels[qid])+negatives(labels[qid],[d for d in pool if d in allowed],12):x.append(features(rankings,qid,d));y.append(float(d in labels[qid]))
x=np.array(x);y=np.array(y);anchor=np.zeros(x.shape[1]);anchor[0]=-1.5;anchor[2]=60.;anchor[7]=60.;w=anchor.copy();loss=[]
for step in range(200):
 z=np.clip(x@w,-30,30);p=1/(1+np.exp(-z));w-=.5*(x.T@(p-y)/len(y)+.01*(w-anchor));loss.append(float(np.mean(np.logaddexp(0,z)-y*z)+.005*np.sum((w-anchor)**2)))
result={}
for q in dev:
 qid=q['id'];pool=rrf([rankings[n][qid] for n in rankings]);result[qid]=sorted(pool,key=lambda d:(-float(np.array(features(rankings,qid,d))@w),d))
metric=evaluate({q['id']:labels[q['id']] for q in dev},result);save(R/'results/adaptation-anchoring.json',{'rankings':result,'metrics':metric,'loss':loss,'weights':w.tolist(),'anchor':anchor.tolist(),'note':'L2 anchoring to untrained RRF score weights. Earlier file called replay was label smoothing, not replay; retained as historical mislabelled run.'});save(R/'checkpoints/anchoring.json',{'weights':w.tolist(),'anchor':anchor.tolist()});print(metric['aggregate'],flush=True)
