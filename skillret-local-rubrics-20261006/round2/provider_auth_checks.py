"""User-run read-only access checks; no credential files, settings or paid APIs."""
import getpass,json,math,sys,urllib.request,urllib.error,warnings
from jev_auth_check import NoRedirect
from secret_store import obtain_key
ENDPOINTS={'runpod':'https://api.runpod.io/v2/pods?limit=1','openrouter':'https://openrouter.ai/api/v1/key'}
RUNPOD_LEGACY='https://rest.runpod.io/v1/pods'
def check(provider,key,opener=None,legacy=False):
 if legacy and provider!='runpod':raise ValueError('Legacy mode is RunPod only')
 if not key.strip() or '\n' in key or '\r' in key:raise ValueError('Invalid input')
 if opener is None:opener=urllib.request.build_opener(NoRedirect).open
 endpoint=RUNPOD_LEGACY if legacy else ENDPOINTS[provider]
 request=urllib.request.Request(endpoint,headers={'Authorization':'Bearer '+key.strip(),'Accept':'application/json'},method='GET')
 try:
  with opener(request,timeout=20) as response:
   if provider=='runpod':
    # Authenticate Pod-read permission without reading pod bodies or env values.
    return {'provider':'runpod','authenticated_pod_read':response.status==200,'balance_verified':False,'write_permissions_tested':False,'paid_actions':0,'note':'No pod response body read. Check available balance in RunPod Billing before deployment.'}
   raw=response.read(1024*1024+1)
  if len(raw)>1024*1024:raise ValueError('Oversized status response')
  data=json.loads(raw).get('data')
  if not isinstance(data,dict):raise ValueError('Unexpected response')
  result={'provider':'openrouter','authenticated':True,'balance_verified':False,'paid_actions':0,'note':'Key allowance is not account cash balance. No inference or credit purchase.'}
  for name in ('limit','limit_remaining','usage'):
   v=data.get(name)
   if v is None:result[name]=None
   elif not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v):result[name]=v
  for name in ('is_free_tier','is_management_key'):
   if isinstance(data.get(name),bool):result[name]=data[name]
  return result
 except urllib.error.HTTPError as e:
  code=e.code;diagnostic={}
  if provider=='runpod' and code==403:
   kind=(e.headers.get('Content-Type','') if e.headers else '').split(';')[0].strip().lower()
   diagnostic['response_kind']='json' if kind in ('application/json','application/problem+json') else 'html' if kind=='text/html' else 'other_or_unknown'
   if diagnostic['response_kind']=='json':
    try:
     raw=e.read(4097)
     obj=json.loads(raw) if len(raw)<=4096 else {}
     if isinstance(obj,dict):
      for name,allowed in [('title',('Forbidden','Unauthorized')),('detail',('access denied','missing bearer token'))]:
       if obj.get(name) in allowed:diagnostic[name]=obj[name]
      if obj.get('status')==403:diagnostic['status']=403
    except Exception:pass
   diagnostic['interpretation']='HTML response may come from a gateway; source unconfirmed.' if diagnostic['response_kind']=='html' else 'Authorization or gateway denial; valid key and saved permission state remain unconfirmed.'
  e.close()
  return {'provider':provider,'authenticated':None,'access_check_succeeded':False,'http_status':code,'paid_actions':0,'diagnostic':diagnostic,'note':'Access denied; a 403 alone does not establish an invalid key. No permissions changed or arbitrary response text displayed.' if code==403 else 'Read-only access check did not succeed; key validity is unconfirmed. No response body displayed.'}
 except (urllib.error.URLError,TimeoutError,ValueError):
  return {'provider':provider,'authenticated':None,'access_check_succeeded':False,'paid_actions':0,'note':'Connection or response error; key validity is unconfirmed. No sensitive details displayed.'}
def main(provider,legacy=False):
 if not sys.stdin.isatty():print('Run yourself in an interactive Ubuntu terminal; piped input refused.');return 2
 print(f'{provider} only. Read-only GET {RUNPOD_LEGACY if legacy else ENDPOINTS[provider]}')
 if legacy:print('Explicit legacy v1 compatibility check. No automatic fallback, retry, or permission change.')
 print('Uses your provider-specific local store if configured; otherwise hidden input. No key writes, paid inference, pod creation, purchase, redirects or retries.')
 if provider=='runpod':print('Checks pod-read access only; pod bodies and account secret values are not read. Balance/write permissions remain unverified.')
 else:print('Shows numeric key allowance/usage only; never key labels, identifiers or raw responses. Account balance may differ.')
 try:
  key=obtain_key(provider,f'{provider} API key (hidden): ')
  result=check(provider,key,legacy=legacy);del key;print(json.dumps(result,indent=2));return 0 if result.get('authenticated',result.get('authenticated_pod_read',False)) else 1
 except (Exception,KeyboardInterrupt):print('\nCancelled or secure input unavailable. Nothing saved; no sensitive details displayed.');return 2
