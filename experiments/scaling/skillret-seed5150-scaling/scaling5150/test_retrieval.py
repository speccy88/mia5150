import unittest
from retrieval import sorted_scores,roundrobin,field_tfidf
class Retrieval(unittest.TestCase):
 def test_ties(self):self.assertEqual(sorted_scores(['z','a','b'],[1,1,0]),['a','z','b'])
 def test_roundrobin(self):self.assertEqual(roundrobin(['a','b','c'],['a','d','e'],4),['a','b','d','c'])
 def test_fields_and_unknown_query(self):
  t={'b':'Mailer | Send email messages','a':'Calculator | Compute arithmetic'}
  self.assertEqual(field_tfidf('email',['a','b'],t)[0][0],'b');self.assertEqual(field_tfidf('zzzz',['b','a'],t)[0],['a','b'])
if __name__=='__main__':unittest.main()
