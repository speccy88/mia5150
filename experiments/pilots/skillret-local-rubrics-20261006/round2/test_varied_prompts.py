import copy,unittest
from varied_prompts import request,parse,VARIANTS
from probe_contract import MODEL
class Contracts(unittest.TestCase):
 def setUp(self):
  self.pool=[str(i) for i in range(20)]
  self.response={'model':MODEL,'usage':{'input_tokens':10},'answers':{}}
  for v in VARIANTS:
   for i in range(20):
    a={'type':'score','score':1.5}
    if v=='action':a={'type':'noul','noul':.5}
    if v=='role_choice':a={'type':'choice','probabilities':{'irrelevant':.1,'topical':.2,'support':.3,'direct':.4}}
    self.response['answers'][f'{v}_{i}']=a
 def test_ties_and_mapping(self):
  result,n=parse(self.response,self.pool,VARIANTS)
  self.assertEqual(n,10)
  for v in VARIANTS:self.assertEqual(result[v]['ranking'],self.pool);self.assertEqual(result[v]['fusion'],self.pool)
  self.assertAlmostEqual(result['role_choice']['scores'][0],2.)
 def test_nonfinite_rejected(self):
  for value in (float('nan'),float('inf'),-1,4,True):
   r=copy.deepcopy(self.response);r['answers']['original_0']['score']=value
   with self.assertRaises(ValueError):parse(r,self.pool,['original'])
 def test_bad_probabilities(self):
  for probs in ({'direct':1},{'irrelevant':.1,'topical':.1,'support':.1,'direct':.1}):
   r=copy.deepcopy(self.response);r['answers']['role_choice_0']['probabilities']=probs
   with self.assertRaises(ValueError):parse(r,self.pool,['role_choice'])
 def test_missing_answer(self):
  del self.response['answers']['action_19']
  with self.assertRaises(KeyError):parse(self.response,self.pool,['action'])
 def test_depth_and_payload(self):
  cs=[{'id':str(i),'name':'test','description':'description'} for i in range(20)]
  p=request({'id':'q','query':'task'},cs,['original','action']);self.assertEqual(len(p['questions']),40)
  self.assertEqual(len(request({'id':'q','query':'task'},cs[:10],['original'])['questions']),10)
  with self.assertRaises(ValueError):request({'id':'q','query':'task'},cs[:9],['original'])
  cs[0]['description']='x'*60001
  with self.assertRaises(ValueError):request({'id':'q','query':'task'},cs,['original'])
 def test_answer_dictionary_order_does_not_change_results(self):
  for i in range(20):self.response['answers'][f'original_{i}']['score']=i/7
  expected=parse(self.response,self.pool,VARIANTS)
  self.response['answers']=dict(reversed(list(self.response['answers'].items())))
  self.assertEqual(parse(self.response,self.pool,VARIANTS),expected)
 def test_choice_key_order_does_not_change_expected_ordinal(self):
  expected=parse(self.response,self.pool,['role_choice'])
  for i in range(20):
   a=self.response['answers'][f'role_choice_{i}'];a['probabilities']=dict(reversed(list(a['probabilities'].items())))
  self.assertEqual(parse(self.response,self.pool,['role_choice']),expected)
 def test_permuted_candidates_map_back_to_skill_ids(self):
  for i in range(20):self.response['answers'][f'original_{i}']['score']=i/7
  expected=parse(self.response,self.pool,['original'])[0]['original']['ranking']
  reversed_response=copy.deepcopy(self.response)
  for i in range(20):reversed_response['answers'][f'original_{i}']=self.response['answers'][f'original_{19-i}']
  actual=parse(reversed_response,self.pool[::-1],['original'])[0]['original']['ranking']
  self.assertEqual(actual,expected)
  # Fusion deliberately anchors to original retrieval order; do not change that pool.
 def test_wrong_type_and_model_rejected(self):
  r=copy.deepcopy(self.response);r['answers']['action_0']['type']='score'
  with self.assertRaises(ValueError):parse(r,self.pool,['action'])
  r=copy.deepcopy(self.response);r['model']='other'
  with self.assertRaises(ValueError):parse(r,self.pool,['original'])
 def test_invalid_usage_rejected(self):
  for n in (-1,64001,True,1.5):
   r=copy.deepcopy(self.response);r['usage']['input_tokens']=n
   with self.assertRaises(ValueError):parse(r,self.pool,['original'])
 def test_finite_noul_and_choice_probabilities(self):
  for bad in (float('nan'),float('inf'),True,-.1,1.1):
   r=copy.deepcopy(self.response);r['answers']['action_0']['noul']=bad
   with self.assertRaises(ValueError):parse(r,self.pool,['action'])
   r=copy.deepcopy(self.response);r['answers']['role_choice_0']['probabilities']['direct']=bad
   with self.assertRaises(ValueError):parse(r,self.pool,['role_choice'])
if __name__=='__main__':unittest.main()
