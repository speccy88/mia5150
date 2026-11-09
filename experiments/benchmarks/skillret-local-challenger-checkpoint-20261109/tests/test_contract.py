import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import *

def test_multi_positive_metric_denominators_and_missing():
 g={'a':{'s1','s2'},'b':{'s3'}};r={'a':['s1','x','s2']};e=evaluate(g,r)
 assert e['query_count']==2 and e['missing_query_count']==1
 assert e['per_query']['a']['Recall@1']==.5 and e['per_query']['a']['Hit@1']==1
 assert e['aggregate']['Completeness@5']==.5
 assert abs(e['per_query']['a']['MAP@5']-(1+2/3)/2)<1e-12
 assert e['aggregate']['Precision@5']==.2

def test_ties_preserve_source_order():
 d=np.array([[1,0],[1,0],[0,1]],np.float32);v=np.array([1,0],np.float32)
 assert retrieve(v,d,['b','a','c'])[0]==['b','a','c']

def test_multi_positive_softmax_gradient():
 import torch
 v=torch.tensor([.4,.2,.3],requires_grad=True);pos=torch.tensor([0,2]);loss=-torch.log_softmax(v,0)[pos].mean();loss.backward()
 expected=torch.softmax(v.detach(),0)-torch.tensor([.5,0,.5]);assert torch.allclose(v.grad,expected)

def test_partitions_all_positives_and_source_groups():
 m=read(STUDY/'manifests/split_membership.json');s,q,g=data('train');parts=m['skill_partition'];seen=set()
 for p in ('train','selection','confirmation'):
  ids=m[p+'_query_ids'];assert not(seen&set(ids));seen.update(ids)
  for qid in ids:assert {parts[sid] for sid in g[qid]}=={p}
 assert len(m['selection_query_ids'])==len(m['confirmation_query_ids'])==512
 repos={}
 for x in s:
  repo=x.get('repo','').casefold().strip().rstrip('/')
  if repo:
   if repo in repos:assert repos[repo]==parts[x['id']]
   repos[repo]=parts[x['id']]

def test_metric_reproduction_without_inference():
 _,_,g=data('test');b=read(OLD/'results/baseline_skillret_06b.json');r=evaluate(g,b['rankings'])
 for k,v in b['evaluation']['aggregate'].items():
  if k in r['aggregate']:assert abs(r['aggregate'][k]-v)<1e-12

def test_fusion_baseline_and_scale_invariance():
 from train_reranker import normalized_fusion
 a=normalized_fusion([.9,.7,.3],[1,4,-3],0);assert list(a)==[0,1,2]
 a=normalized_fusion([.9,.7,.3],[1,4,-3],.5);b=normalized_fusion([90,70,30],[15,21,7],.5);assert list(a)==list(b)
