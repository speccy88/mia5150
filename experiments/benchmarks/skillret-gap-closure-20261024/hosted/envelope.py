"""Pure defensive parsing and validity gates. No network, keys or billing guesses."""
import decimal,hashlib,json,re,email.utils

def cooldown_headers(headers):
    """Retain only parsed cooldown guidance, never arbitrary header contents."""
    raw=headers.get('Retry-After') if headers else None
    result={'retry_after_header_present':raw is not None}
    if raw is None:return result
    try:
        seconds=int(raw)
        if seconds<0:raise ValueError('negative')
        result['retry_after_seconds']=seconds
    except (ValueError,TypeError):
        try:result['retry_after_utc']=email.utils.parsedate_to_datetime(raw).isoformat()
        except (ValueError,TypeError,OverflowError):result['retry_after_unparseable']=True
    return result

def decode_http(raw,status):
    try:value=json.loads(raw)
    except (ValueError,UnicodeError):return {'_malformed_json':True,'_raw_sha256':hashlib.sha256(raw).hexdigest(),'_raw_bytes':len(raw),'_http_status':status}
    if not isinstance(value,dict):return {'_non_object_json':type(value).__name__,'_http_status':status}
    return {**value,'_http_status':status}

def provider_matches(value,expected_provider='DeepInfra'):
    # Human-readable provider evidence does not prove FP8; the request pin is
    # recorded separately. Do not compare display names to endpoint slugs.
    expected=re.sub('[^a-z0-9]','',expected_provider.casefold())
    return isinstance(value,str) and re.sub('[^a-z0-9]','',value.casefold()) in {expected,expected+'fp8'}

def provenance_matches(model,provider,expected_model,expected_provider='DeepInfra'):return model==expected_model and provider_matches(provider,expected_provider)

def envelope_identity(value):
    if not isinstance(value,dict):value={'_non_object_json':type(value).__name__}
    usage=value.get('usage');choices=value.get('choices')
    return {'response_id':value.get('id') if isinstance(value.get('id'),str) else None,'model':value.get('model') if isinstance(value.get('model'),str) else None,'provider':value.get('provider') if isinstance(value.get('provider'),str) else None,'http_status':value.get('_http_status'),'response_keys':sorted(value),'usage_type':type(usage).__name__,'choices_type':type(choices).__name__,'choice_count':len(choices) if isinstance(choices,list) else None}

def parse_envelope(value):
    identity=envelope_identity(value)
    if not isinstance(value,dict):value={'_non_object_json':type(value).__name__}
    usage=value.get('usage') if isinstance(value.get('usage'),dict) else {}
    choices=value.get('choices') if isinstance(value.get('choices'),list) else []
    choice=choices[0] if choices and isinstance(choices[0],dict) else {}
    message=choice.get('message') if isinstance(choice.get('message'),dict) else {}
    content=message.get('content');finish=choice.get('finish_reason');details=usage.get('completion_tokens_details')
    status='returned'
    if value.get('_malformed_json'):status='malformed_json'
    elif value.get('_non_object_json'):status='non_object_response'
    elif 'error' in value and value['error'] is not None:status='provider_error'
    elif not choices:status='empty_choices'
    elif not isinstance(choices[0],dict):status='invalid_choice'
    elif 'error' in choice and choice['error'] is not None:status='choice_error'
    elif message.get('refusal'):status='refusal'
    elif not isinstance(choice.get('message'),dict):status='invalid_message'
    elif finish!='stop':status='abnormal_finish'
    elif not isinstance(content,str) or not content.strip():status='empty_content'
    return {**identity,'status':status,'input_tokens':usage.get('prompt_tokens'),'output_tokens':usage.get('completion_tokens'),'reasoning_tokens':details.get('reasoning_tokens') if isinstance(details,dict) else None,'cost_usd':usage.get('cost'),'finish_reason':finish,'content':content if isinstance(content,str) else None,'refusal_present':bool(message.get('refusal'))}

def valid_result(record,expected_model,expected_provider='DeepInfra'):
    try:cost=decimal.Decimal(str(record.get('cost_usd')));cost_known=not isinstance(record.get('cost_usd'),bool) and cost.is_finite() and cost>=0
    except decimal.InvalidOperation:cost_known=False
    return cost_known and record.get('status')=='returned' and record.get('finish_reason')=='stop' and not record.get('refusal_present') and isinstance(record.get('content'),str) and bool(record['content'].strip()) and provenance_matches(record.get('model'),record.get('provider'),expected_model,expected_provider) and all(isinstance(record.get(k),int) and not isinstance(record[k],bool) and record[k]>=0 for k in ['input_tokens','output_tokens'])

def canary_valid(record,expected_model):
    if not valid_result(record,expected_model):return False
    try:return json.loads(record['content'])==['s000000000000']
    except (ValueError,TypeError):return False
