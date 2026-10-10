"""Read-only credit/route verification. Never runs inference or buys credits."""
import json,urllib.request,urllib.error
from decimal import Decimal
from secret_store import load_key
from jev_auth_check import NoRedirect
from openrouter_probe import read_json,quote,RESERVE,HERE
from jev_probe import save
CREDITS='https://openrouter.ai/api/v1/credits'
def credits(key,opener):
 try:
  data=read_json(urllib.request.Request(CREDITS,headers={'Authorization':'Bearer '+key},method='GET'),opener)['data']
  total=Decimal(str(data['total_credits']));used=Decimal(str(data['total_usage']))
  if not total.is_finite() or not used.is_finite() or total<0 or used<0:raise ValueError('Invalid credit status')
  remaining=total-used
  return {'http_status':200,'balance_verified':True,'remaining_credit_usd':str(remaining),'covers_reservation':remaining>=RESERVE}
 except urllib.error.HTTPError as e:
  code=e.code;e.close()
  return {'http_status':code,'balance_verified':False,'covers_reservation':False,'note':'The documented credits endpoint requires a management key. No permissions changed; no response body displayed.' if code==403 else 'Read-only balance request failed; no sensitive details displayed.'}
 except Exception:return {'balance_verified':False,'covers_reservation':False,'note':'Balance could not be verified; sanitized response/connection failure.'}
def main():
 opener=urllib.request.build_opener(NoRedirect).open
 try:pricing=quote(opener)
 except Exception:pricing={'route_verified':False,'note':'Price/route verification failed; no sensitive details displayed.'}
 key=load_key('openrouter')
 try:balance=credits(key,opener)
 finally:del key
 result={'pricing':pricing,'balance':balance,'inference_calls':0,'credit_purchases':0,'ledger_changed':False}
 save(HERE/'openrouter-preflight.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 try:main()
 except Exception:print('Preflight stopped without paid calls; no sensitive details displayed.')
