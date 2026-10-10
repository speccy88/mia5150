"""USER-RUN one-shot TypeSafe authentication check. Never saves credentials."""
import getpass,json,sys,urllib.request,urllib.error,warnings
from secret_store import obtain_key
ENDPOINT='https://api.typesafe.ai/v1/models'
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,request,fp,code,msg,headers,newurl):
  raise urllib.error.HTTPError(request.full_url,code,'Redirect refused',headers,None)
def check_key(key,opener=None):
 if not key.strip() or '\n' in key or '\r' in key:raise ValueError('A nonempty single-line key is required')
 request=urllib.request.Request(ENDPOINT,headers={'Authorization':'Bearer '+key.strip(),'Accept':'application/json'},method='GET')
 if opener is None:opener=urllib.request.build_opener(NoRedirect).open
 try:
  with opener(request,timeout=20) as response:
   payload=json.loads(response.read(1024*1024))
  models=payload.get('models')
  if not isinstance(models,list):return {'authenticated':False,'status':'Unexpected response shape; no response contents displayed'}
  return {'authenticated':True,'model_count':len(models),'jev_listed':any(isinstance(m,dict) and str(m.get('name','')).startswith('jev') for m in models),'balance_verified':False,'paid_calls':0}
 except urllib.error.HTTPError as e:
  code=e.code;e.close();return {'authenticated':False,'http_status':code,'status':'Authentication check did not succeed; no response body displayed','paid_calls':0}
 except (urllib.error.URLError,TimeoutError,json.JSONDecodeError):
  return {'authenticated':False,'status':'Connection or response error; no sensitive details displayed','paid_calls':0}
def main():
 if not sys.stdin.isatty():
  print('Run this yourself in an interactive Ubuntu terminal; redirected/piped key input is refused.',file=sys.stderr);return 2
 print('TypeSafe direct Jev only. Destination: https://api.typesafe.ai/v1/models')
 print('Do not enter an OpenRouter or another provider key here. This checks authentication only: no inference, payment, file upload or configuration change.')
 print('Uses your provider-specific local store if configured; otherwise hidden input. This check never writes keys.')
 try:
  key=obtain_key('jev','TypeSafe Jev API key (hidden): ')
  result=check_key(key);del key;print(json.dumps(result,indent=2));return 0 if result['authenticated'] else 1
 except (getpass.GetPassWarning,EOFError,KeyboardInterrupt):
  print('\nSecure input unavailable or cancelled. Nothing saved.');return 2
 except (ValueError,OSError):
  print('Invalid key input. Nothing saved.');return 2
if __name__=='__main__':raise SystemExit(main())
