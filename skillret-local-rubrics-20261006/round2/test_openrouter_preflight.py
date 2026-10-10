import io,json,unittest,urllib.error
from openrouter_preflight import credits,CREDITS
class CreditTests(unittest.TestCase):
 def test_allowed_numeric_fields_only(self):
  def opener(req,timeout):
   self.assertEqual(req.full_url,CREDITS);self.assertEqual(req.get_method(),'GET')
   return io.BytesIO(json.dumps({'data':{'total_credits':2,'total_usage':1,'secret':'DUMMY'}}).encode())
  r=credits('DUMMY',opener);self.assertTrue(r['covers_reservation']);self.assertNotIn('DUMMY',json.dumps(r))
 def test_denial_never_reflects_body(self):
  def opener(req,timeout):raise urllib.error.HTTPError(req.full_url,403,'DUMMY',{},io.BytesIO(b'DUMMY'))
  r=credits('DUMMY',opener);self.assertFalse(r['balance_verified']);self.assertEqual(r['http_status'],403);self.assertNotIn('DUMMY',json.dumps(r))
