import unittest
from rerank_local import rank,native_prompt
class Tests(unittest.TestCase):
 def test_ties_and_invalid(self):
  self.assertEqual(rank(['b','a'],[1.,1.]),['b','a'])
  with self.assertRaises(ValueError):rank(['a'],[float('nan')])
 def test_native_prompt_preserves_document(self):
  doc='x'*12000
  self.assertIn(doc,native_prompt('query',doc))
if __name__=='__main__':unittest.main()
