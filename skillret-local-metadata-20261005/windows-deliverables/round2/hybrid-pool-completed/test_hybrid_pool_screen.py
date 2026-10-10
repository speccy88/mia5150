import unittest
from hybrid_pool_screen import hybrid
class Pools(unittest.TestCase):
 def test_disjoint_balanced(self):
  a=[f'a{i}' for i in range(100)];b=[f'b{i}' for i in range(100)];p=hybrid(a,b)
  self.assertEqual(p,[d for i in range(10) for d in (a[i],b[i])])
 def test_identical_pool_and_overlap(self):
  a=list(map(str,range(100)));self.assertEqual(hybrid(a,a),a[:20])
  p=hybrid(a,a[5:]+a[:5]);self.assertEqual(len(p),20);self.assertEqual(len(set(p)),20)
 def test_shortfall(self):
  with self.assertRaises(ValueError):hybrid(['a'],['a'])
if __name__=='__main__':unittest.main()
