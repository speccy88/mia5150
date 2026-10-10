import unittest, tempfile, pathlib
from harness import *
class Contract(unittest.TestCase):
 def test_multigold(self):
  a=evaluate({'q':{'a':1,'d':1}},{'q':['a','b','c','d']})['aggregate'];self.assertEqual(a['Recall@5'],1);self.assertEqual(a['Completeness@5'],1);self.assertEqual(a['MRR@5'],1)
 def test_missing_duplicate_cutoff(self):
  a=evaluate({'q':{'a':1,'d':1},'missing':{'a':1}},{'q':['a','a']})['aggregate'];self.assertEqual(a['Recall@5'],.25)
  self.assertEqual(evaluate({'q':{'g':1}},{'q':['a','b','c','d','e','g']})['aggregate']['MRR@5'],0)
 def test_official_parity(self):
  for r in [['a','b','c','d'],['d','c','b','a'],[],['x','y']]:
   labels={'q':{'a':1,'d':1}};mine=evaluate(labels,{'q':r})['aggregate'];official=official_metrics(labels,{'q':r})
   for key,value in official.items():self.assertAlmostEqual(mine[key],value,places=4)
 def test_no_leak(self):self.assertEqual(safe_queries([{'id':'q','query':'text','skill_ids':['a'],'k':1}]),[{'id':'q','query':'text'}])
 def test_negative_collisions(self):self.assertEqual(negatives({'a','b'},['a','c','b','c','d']),['c','d'])
 def test_finalset(self):self.assertEqual(set_metrics(['a','b'],['a'])['exact'],0)
 def test_rrf_duplicate_does_not_vote_twice(self):
  self.assertEqual(rrf([['a','a','b'],['b']]),rrf([['a','b'],['b']]))
 def test_failures_keep_denominator(self):
  a=evaluate({'ok':{'x':1},'fail':{'y':1}},{'ok':['x'],'fail':[]})
  self.assertEqual(a['query_count'],2);self.assertEqual(a['empty_rate'],.5);self.assertEqual(a['aggregate']['NDCG@10'],.5)
 def test_mmr_relevance_gate(self):
  import numpy as np
  self.assertEqual(mmr(['a','b'],{'a':.9,'b':.1},{'a':np.array([1.,0.]),'b':np.array([0.,1.])},.8,5,.5),['a'])
 def test_dsr_budget_unique(self):
  import numpy as np
  from structures import dsr
  r=dsr(['a','b','c'],np.array([.9,.8,.7]),np.array([[1.,0.],[1.,0.],[0.,1.]]),np.array([1.,0.]),2)
  self.assertEqual(len(r),2);self.assertEqual(len(set(r)),2)
 def test_domain_fallback_retains_global(self):
  import numpy as np
  from structures import route
  scores=np.array([.9,.01,.8,.8]);groups=['a','a','b','b']
  self.assertNotIn(0,route(scores,groups,1));self.assertIn(0,route(scores,groups,1,1))
 def test_empty_finalset_does_not_activate(self):
  self.assertEqual(set_metrics([],['a'])['exact'],0);self.assertEqual(set_metrics([],['a'])['recall'],0)
if __name__=='__main__':unittest.main()
