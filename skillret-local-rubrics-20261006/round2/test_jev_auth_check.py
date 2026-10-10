import io,json,unittest
from unittest.mock import patch
from jev_auth_check import check_key,main,ENDPOINT
class AuthCheck(unittest.TestCase):
 def test_safe_summary_and_endpoint(self):
  def opener(request,timeout):
   self.assertEqual(request.full_url,ENDPOINT);self.assertEqual(request.get_method(),'GET');self.assertEqual(timeout,20)
   return io.BytesIO(json.dumps({'models':[{'name':'jev-latest'}],'ignored':'server-detail'}).encode())
  result=check_key('synthetic-placeholder',opener);self.assertTrue(result['authenticated']);self.assertNotIn('synthetic',json.dumps(result));self.assertFalse(result['balance_verified'])
 def test_no_redirected_input(self):
  with patch('sys.stdin.isatty',return_value=False),patch('getpass.getpass') as prompt:
   self.assertEqual(main(),2);prompt.assert_not_called()
 def test_no_header_injection(self):
  with self.assertRaises(ValueError):check_key('synthetic\nheader')
if __name__=='__main__':unittest.main()
