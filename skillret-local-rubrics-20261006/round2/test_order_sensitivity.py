import json,unittest
from order_sensitivity import prepare,HERE
class Tests(unittest.TestCase):
 def test_reversal_changes_only_question_order_and_ids(self):
  items,protocol=prepare();original=json.loads((HERE/'prepared-fusion-confirmation.json').read_text())
  self.assertEqual(len(items),2)
  for new,old in zip(items,original):
   self.assertEqual(new['permuted_ids'],old['candidate_ids'][::-1]);self.assertEqual(new['request']['state'],old['request']['state']);self.assertEqual(new['request']['model'],old['request']['model'])
   for i in range(20):self.assertEqual(new['request']['questions'][f's{i}'],old['request']['questions'][f's{19-i}'])
  self.assertEqual(protocol['max_calls'],2)
