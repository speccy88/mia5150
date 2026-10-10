import unittest
from protocol import library,render
class Libraries(unittest.TestCase):
 def test_nested_complete_and_order_independent(self):
  ids=[f'd{i:03d}' for i in range(200)];gold=['d199','d198','d197'];previous=set()
  for n in [10,50,100,200]:
   r=library(ids,gold,'q',n);self.assertEqual(r,library(ids[::-1],gold[::-1],'q',n));self.assertEqual(len(r),n);self.assertEqual(len(set(r)),n);self.assertTrue(previous<=set(r));self.assertTrue(set(gold)<=set(r));previous=set(r)
 def test_no_gold_first_display_rule(self):
  ids=list(map(str,range(100)));a=library(ids,['99'],'q',100);b=library(ids,['1','3'],'q',100);self.assertEqual(a,b)
 def test_reject_invalid(self):
  for ids,gold,n in [(['a','a'],['a'],1),(['a'],['z'],1),(['a','b'],['a','b'],1)]:
   with self.assertRaises(ValueError):library(ids,gold,'q',n)
 def test_once_capped_view(self):
  self.assertEqual(len(render({'name':'a','description':'b'*2000})),1024)
if __name__=='__main__':unittest.main()
