import unittest
from varied_screen import promote,analyze
class Screen(unittest.TestCase):
 def test_promotion_and_incomplete(self):
  a={'comparisons':{'action':{'mean_delta_recall5':.1,'wins':2,'losses':1},'anchored':{'mean_delta_recall5':.1,'wins':2,'losses':1},'constraint':{'mean_delta_recall5':.2,'wins':1,'losses':0}}}
  rows=[{'status':'ok'} for _ in range(24)]
  self.assertEqual(promote(a,rows),['action'])
  rows[0]['status']='failed_or_uncertain';self.assertEqual(promote(a,rows),[])
 def test_true_recall_and_failure_denominator(self):
  pools={'q':['a','b'],'r':['c','d']};labels={'q':{'a':1,'z':1},'r':{'c':1}}
  a=analyze([],['q','r'],pools,labels,['original'])
  self.assertEqual(a['metrics']['BM25']['aggregate']['Recall@5'],.75)
  self.assertEqual(a['metrics']['BM25']['aggregate']['Completeness@5'],.5)
  self.assertEqual(a['metrics']['original_direct']['aggregate']['Recall@5'],0)
  self.assertEqual(a['metrics']['original_direct']['query_count'],2)
if __name__=='__main__':unittest.main()
