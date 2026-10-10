import io,json,tempfile,unittest,urllib.error
from pathlib import Path
from selective_jev import run_batch,fusion,MODEL
def items():return [{'id':str(i),'candidate_ids':[str(j) for j in range(20)],'escalate':i<4,'request':{}} for i in range(8)]
class SelectiveTests(unittest.TestCase):
 def test_four_calls_eight_outputs(self):
  calls=[]
  def opener(req,timeout):
   calls.append(1);return io.BytesIO(json.dumps({'model':MODEL,'usage':{'input_tokens':100},'answers':{f's{i}':{'type':'score','score':3-i/10} for i in range(20)}}).encode())
  with tempfile.TemporaryDirectory() as d:
   r=run_batch(items(),'DUMMY',Path(d),opener);self.assertEqual(len(calls),4);self.assertEqual(len(r['rows']),8)
   self.assertTrue(all(x['ranking']==x['candidate_ids'] for x in r['rows']));self.assertNotIn('DUMMY',json.dumps(r))
 def test_stop_failure_preserves_policy_dense(self):
  calls=[]
  def opener(req,timeout):calls.append(1);raise urllib.error.HTTPError(req.full_url,429,'DUMMY',{},io.BytesIO(b'DUMMY'))
  with tempfile.TemporaryDirectory() as d:
   r=run_batch(items(),'DUMMY',Path(d),opener);self.assertEqual(len(calls),1);self.assertTrue(all(x['ranking']==x['candidate_ids'] for x in r['rows']))
   self.assertEqual(r['rows'][0]['http_status'],429);self.assertEqual(r['rows'][1]['no_failure_fallback_ranking'],[]);self.assertEqual(r['rows'][4]['status'],'dense_no_call')
 def test_quota_rejected(self):
  x=items();x[4]['escalate']=True
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):run_batch(x,'DUMMY',Path(d),None)
