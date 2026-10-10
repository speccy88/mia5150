import unittest
from local_rubric_screen import rank,Bounded
class LocalContracts(unittest.TestCase):
 def test_rank_ties_and_invalid(self):
  p=[str(i) for i in range(20)];self.assertEqual(rank(p,[1.5]*20),(p,p))
  with self.assertRaises(ValueError):rank(p,[float('nan')]*20)
  with self.assertRaises(ValueError):rank(p[:19],[1]*19)
 def test_local_endpoint_and_call_bounds(self):
  client=Bounded(url='https://example.com');client.calls=0
  with self.assertRaises(ValueError):client._post('/completion',{'prompt':[1],'n_predict':1})
  client.url='http://127.0.0.1:18089';client.calls=80
  with self.assertRaises(ValueError):client._post('/completion',{'prompt':[1],'n_predict':1})
  client.calls=0
  with self.assertRaises(ValueError):client._post('/completion',{'prompt':[1]*2048,'n_predict':1})
if __name__=='__main__':unittest.main()
