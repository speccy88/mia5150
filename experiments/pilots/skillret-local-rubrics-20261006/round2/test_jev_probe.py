import io,json,tempfile,unittest,urllib.error
from pathlib import Path
from jev_probe import run_batch,ENDPOINT
from probe_contract import build_request,MODEL
class Probe(unittest.TestCase):
 def packets(self,n=12):
  c=[{'id':str(i),'name':'skill','description':'header'} for i in range(20)]
  return [{'id':f'q{i}','candidates':c,'request':build_request({'id':f'q{i}','query':'task'},c),'preflight_error':None} for i in range(n)]
 def response(self):return {'model':MODEL,'usage':{'input_tokens':100},'answers':{**{f'n{i}':{'type':'noul','noul':.5} for i in range(20)},**{f's{i}':{'type':'score','score':2.} for i in range(20)}},'irrelevant_secret_echo':'synthetic-placeholder'}
 def test_fixed_endpoint_cap_redaction(self):
  calls=[]
  with tempfile.TemporaryDirectory() as d:
   def opener(req,timeout):calls.append(req.full_url);self.assertEqual(req.get_method(),'POST');return io.BytesIO(json.dumps(self.response()).encode())
   result=run_batch('synthetic-placeholder',self.packets(),Path(d),opener)
   self.assertEqual(calls,[ENDPOINT]*12);self.assertEqual(result['known_input_tokens'],1200)
   self.assertNotIn('synthetic-placeholder',(Path(d)/'results.json').read_text())
 def test_error_never_retried_and_denominator_retained(self):
  calls=[]
  with tempfile.TemporaryDirectory() as d:
   def opener(req,timeout):calls.append(1);raise urllib.error.HTTPError(ENDPOINT,429,'synthetic-secret',{},None)
   result=run_batch('synthetic-placeholder',self.packets(),Path(d),opener)
   self.assertEqual(len(calls),1);self.assertEqual(len(result['rows']),12);self.assertNotIn('synthetic-secret',json.dumps(result));self.assertEqual(result['rows'][-1]['status'],'not_attempted')
 def test_oversized_batch_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):run_batch('synthetic',self.packets(13),Path(d),None)
 def test_missing_usage_stops(self):
  with tempfile.TemporaryDirectory() as d:
   result=run_batch('synthetic',self.packets(),Path(d),lambda *a,**kw:io.BytesIO(b'{}'))
   self.assertEqual(result['attempts'],1);self.assertEqual(result['rows'][0]['status'],'failed_or_uncertain')
if __name__=='__main__':unittest.main()
