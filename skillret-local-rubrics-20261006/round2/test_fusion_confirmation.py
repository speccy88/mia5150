import io,json,tempfile,unittest,urllib.error
from pathlib import Path
from fusion_confirmation import run_batch,MODEL
def items():return [{'id':str(i),'candidate_ids':[str(j) for j in range(20)],'request':{}} for i in range(8)]
class Tests(unittest.TestCase):
 def test_eight_successes(self):
  calls=[]
  def opener(req,timeout):
   calls.append(1);return io.BytesIO(json.dumps({'model':MODEL,'usage':{'input_tokens':100},'answers':{f's{i}':{'type':'score','score':3-i/10} for i in range(20)}}).encode())
  with tempfile.TemporaryDirectory() as d:
   r=run_batch(items(),'DUMMY',Path(d),opener);self.assertEqual(len(calls),8);self.assertTrue(all(x['ranking']==x['candidate_ids'] for x in r['rows']));self.assertNotIn('DUMMY',json.dumps(r))
 def test_no_retry_fixed_failure_denominator(self):
  calls=[]
  def opener(req,timeout):calls.append(1);raise urllib.error.HTTPError(req.full_url,429,'DUMMY',{},io.BytesIO(b'DUMMY'))
  with tempfile.TemporaryDirectory() as d:
   r=run_batch(items(),'DUMMY',Path(d),opener);self.assertEqual(len(calls),1);self.assertEqual(len(r['rows']),8);self.assertTrue(all(x['ranking']==[] for x in r['rows']));self.assertEqual(r['rows'][0]['http_status'],429)
