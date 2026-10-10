"""Explicitly user-authorized read-only checks; never displays credentials."""
import json
from secret_store import STORE,NAMES,directory,load_key
from jev_auth_check import check_key
from provider_auth_checks import check,ENDPOINTS
def main():
 try:directory(STORE)
 except Exception:
  print(json.dumps({'store_permissions_valid':False,'network_calls':0}));return 1
 print(json.dumps({'store_path':str(STORE),'directory_permissions':'0700','store_permissions_valid':True}))
 for provider in NAMES:
  try:key=load_key(provider)
  except Exception:
   print(json.dumps({'provider':provider,'credential_file_valid':False,'network_calls':0}));continue
  try:
   result=check_key(key) if provider=='jev' else check(provider,key)
   result.update(provider=provider,credential_file_valid=True,file_permissions='0600',endpoint='https://api.typesafe.ai/v1/models' if provider=='jev' else ENDPOINTS[provider])
   if result.get('authenticated') or result.get('authenticated_pod_read'):result['http_status']=200
   print(json.dumps(result))
  except Exception:print(json.dumps({'provider':provider,'credential_file_valid':True,'access_check_succeeded':False,'details':'Sanitized connection/check failure; no sensitive output.'}))
  finally:del key
 return 0
if __name__=='__main__':raise SystemExit(main())
