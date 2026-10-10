import unittest
from probe_contract import *
class Contract(unittest.TestCase):
 def setUp(self):self.c=[{'id':str(i),'name':f'skill{i}','description':'Public skill header'} for i in range(20)]
 def test_no_labels_or_hidden_evidence(self):
  with self.assertRaises(ValueError):build_request({'id':'q','query':'task','gold':['a']},self.c)
 def test_frozen_schema(self):
  r=build_request({'id':'q','query':'task'},self.c);self.assertEqual(len(r['questions']),40);self.assertEqual(r['state'],{'task':'task'});self.assertEqual(r['model'],MODEL)
 def test_no_truncation(self):
  self.c[0]['description']='x'*25000
  with self.assertRaises(ValueError):build_request({'id':'q','query':'task'},self.c)
 def test_budget_and_duplicate(self):
  l={'cap_usd':'.04','committed_usd':'0','requests':{}};l=reserve(l,'a','.03')
  with self.assertRaises(ValueError):reserve(l,'b','.02')
  with self.assertRaises(ValueError):reserve(l,'a','.001')
 def test_typed_response(self):
  a={**{f'n{i}':{'type':'noul','noul':i/20} for i in range(20)},**{f's{i}':{'type':'score','score':i/10} for i in range(20)}}
  response={'model':MODEL,'answers':a,'usage':{'input_tokens':123}}
  rankings,usage=parse_response(response,self.c);self.assertEqual(rankings['noul'][0],'19');self.assertEqual(usage,123)
  response['model']='other'
  with self.assertRaises(ValueError):parse_response(response,self.c)
 def test_missing_usage_rejected(self):
  with self.assertRaises(ValueError):parse_response({'model':MODEL},self.c)
if __name__=='__main__':unittest.main()
