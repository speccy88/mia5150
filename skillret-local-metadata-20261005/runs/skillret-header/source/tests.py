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
if __name__=='__main__':unittest.main()
