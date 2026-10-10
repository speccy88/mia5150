import io,json,unittest,urllib.error
from unittest.mock import patch
from provider_auth_checks import check,main,ENDPOINTS,RUNPOD_LEGACY
class Checks(unittest.TestCase):
 def test_runpod_never_reads_body(self):
  class Response(io.BytesIO):
   status=200
   def read(self,*a):raise AssertionError('Pod body must not be read')
  def opener(request,timeout):self.assertEqual(request.full_url,ENDPOINTS['runpod']);self.assertEqual(request.get_method(),'GET');return Response()
  r=check('runpod','synthetic-placeholder',opener);self.assertTrue(r['authenticated_pod_read']);self.assertFalse(r['balance_verified'])
 def test_openrouter_whitelist(self):
  def opener(request,timeout):
   self.assertEqual(request.full_url,ENDPOINTS['openrouter']);self.assertEqual(request.get_method(),'GET')
   return io.BytesIO(json.dumps({'data':{'label':'synthetic-secret','creator_user_id':'private-id','limit':4,'limit_remaining':3,'usage':1,'is_free_tier':False}}).encode())
  r=check('openrouter','synthetic-placeholder',opener);self.assertEqual(r['limit_remaining'],3);self.assertNotIn('synthetic',json.dumps(r));self.assertNotIn('private-id',json.dumps(r))
 def test_redirected_input_refused(self):
  with patch('sys.stdin.isatty',return_value=False),patch('getpass.getpass') as prompt:self.assertEqual(main('runpod'),2);prompt.assert_not_called()
 def test_no_header_injection(self):
  with self.assertRaises(ValueError):check('runpod','x\r\ny')
 def test_403_is_not_invalid_key_claim_and_no_retry(self):
  calls=[]
  def opener(request,timeout):
   calls.append(request.full_url)
   raise urllib.error.HTTPError(request.full_url,403,'synthetic-secret',{},io.BytesIO(b'secret-body'))
  result=check('runpod','synthetic-placeholder',opener)
  self.assertIsNone(result['authenticated']);self.assertEqual(result['http_status'],403)
  self.assertEqual(len(calls),1);self.assertNotIn('synthetic-secret',json.dumps(result));self.assertNotIn('secret-body',json.dumps(result))
 def test_explicit_legacy_never_reads_body(self):
  class Response(io.BytesIO):
   status=200
   def read(self,*a):raise AssertionError('Pod body must not be read')
  def opener(request,timeout):self.assertEqual(request.full_url,RUNPOD_LEGACY);return Response()
  self.assertTrue(check('runpod','synthetic-placeholder',opener,legacy=True)['authenticated_pod_read'])
 def test_legacy_rejected_for_other_provider(self):
  with self.assertRaises(ValueError):check('openrouter','synthetic-placeholder',legacy=True)
 def test_403_diagnostics_never_reflect_arbitrary_text(self):
  for kind,body in [('text/html',b'<html>DUMMY-SECRET</html>'),('application/json',b'{"title":"Forbidden","detail":"DUMMY-SECRET","status":403}')]:
   def opener(request,timeout):raise urllib.error.HTTPError(request.full_url,403,'DUMMY-SECRET',{'Content-Type':kind},io.BytesIO(body))
   r=check('runpod','dummy',opener)
   self.assertNotIn('DUMMY-SECRET',json.dumps(r))
   self.assertEqual(r['diagnostic']['response_kind'],'html' if kind=='text/html' else 'json')
if __name__=='__main__':unittest.main()
