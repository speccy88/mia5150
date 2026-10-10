import unittest
from candidate_budget import pools
class Tests(unittest.TestCase):
 def test_exact_budget_novel_order(self):
  d=[str(i) for i in range(100)];a,b=pools(d,d[10:]+d[:10]);self.assertEqual(a,d[:30]);self.assertEqual(b,d[:30]);self.assertEqual(len(set(b)),30)
 def test_complementary_ids_keep_dense20(self):
  d=[str(i) for i in range(100)];a,b=pools(d,['x'+str(i) for i in range(100)]);self.assertEqual(b[:20],d[:20]);self.assertEqual(b[20:],['x'+str(i) for i in range(10)])
 def test_shortfall_refused(self):
  with self.assertRaises(ValueError):pools([str(i) for i in range(30)],[str(i) for i in range(20)])
