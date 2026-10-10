import io,json,tempfile,unittest,urllib.error
from pathlib import Path
from openrouter_probe import parse,run_batch,MODEL,ENDPOINT,quote,ROUTE
def item():return {'id':'dummy-query','candidate_ids':[str(i) for i in range(20)],'request':{'model':MODEL}}
def response():return {'model':MODEL,'provider':'GMICloud','choices':[{'finish_reason':'stop','message':{'content':json.dumps({'ranking':[f's{i:02d}' for i in range(20)]})}}],'usage':{'prompt_tokens':100,'completion_tokens':100,'cost':0.001}}
class ProbeTests(unittest.TestCase):
 def test_parse_and_reject_malformed_or_wrong_route(self):
  self.assertEqual(parse(response(),item())['ranking'],item()['candidate_ids'])
  for change in ('provider','model'):
   r=response();r[change]='wrong'
   with self.assertRaises(ValueError):parse(r,item())
  r=response();r['choices'][0]['message']['content']='{"ranking":["s00"]}'
  with self.assertRaises(ValueError):parse(r,item())
 def test_success_caps_and_no_key_saved(self):
  seen=[]
  def opener(req,timeout):seen.append(req.full_url);return io.BytesIO(json.dumps(response()).encode())
  with tempfile.TemporaryDirectory() as d:
   result=run_batch('DUMMY-SECRET',[item() for _ in range(12)],Path(d),opener)
   self.assertEqual(result['attempts'],12);self.assertEqual(seen,[ENDPOINT]*12)
   self.assertNotIn('DUMMY-SECRET',json.dumps(result))
 def test_uncertain_failure_stops_and_keeps_denominator(self):
  seen=[]
  def opener(req,timeout):seen.append(1);raise urllib.error.HTTPError(req.full_url,429,'DUMMY-SECRET',{},io.BytesIO(b'DUMMY-SECRET'))
  with tempfile.TemporaryDirectory() as d:
   result=run_batch('DUMMY-SECRET',[item() for _ in range(12)],Path(d),opener)
   self.assertEqual(len(seen),1);self.assertEqual(len(result['rows']),12)
   self.assertNotIn('DUMMY-SECRET',json.dumps(result))
 def test_quote_price_change_stops(self):
  def opener(req,timeout):return io.BytesIO(json.dumps({'data':{'id':MODEL,'endpoints':[{'tag':ROUTE,'quantization':'fp8','model_id':MODEL,'pricing':{'prompt':'1','completion':'1'}}]}}).encode())
  with self.assertRaises(ValueError):quote(opener)
