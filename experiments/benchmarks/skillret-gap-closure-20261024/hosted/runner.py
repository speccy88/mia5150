"""Optional no-fallback OpenRouter stage. Read a credential only from closed stdin.

Never use shell arguments, environment variables or repository files for the key.
No generation retries: a reservation is a terminal attempt, including unknown use.
"""
import argparse,csv,datetime,decimal,gzip,hashlib,json,os,pathlib,re,sys,time,urllib.error,urllib.parse,urllib.request
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;STUDY=HERE.parent
sys.path.insert(0,str(STUDY))
from core import BM25,chat,digest,parse_selection,rank,render,retrieval_metrics,rrf,select_prompt,selection_metrics
from ledger import append_jsonl
from budget import Budget,BudgetStop
from envelope import decode_http,envelope_identity,parse_envelope,provenance_matches,valid_result,canary_valid,cooldown_headers
from request_log import begin as log_begin,finish as log_finish
API='https://openrouter.ai/api/v1/'
MODEL='deepseek/deepseek-v4-flash-0731'
PROVIDER={'only':['deepinfra/fp8'],'allow_fallbacks':False,'require_parameters':True}

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        # Never forward the authentication header to a redirect destination.
        return None

def read(p):return json.loads(pathlib.Path(p).read_text())
def rows(p):
    with open(p) as f:return [json.loads(x) for x in f if x.strip()]
def messages(prompt):
    start='<|im_start|>system\n';split='<|im_end|>\n<|im_start|>user\n';end='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    assert prompt.startswith(start) and prompt.endswith(end)
    system,user=prompt[len(start):-len(end)].split(split,1)
    return [{'role':'system','content':system},{'role':'user','content':user}]
def schema(kind,values):
    # Constant-size structural schema avoids hiding a catalog-size-dependent ID
    # enum inside provider prompts. Known-ID and uniqueness validation is strict
    # after return; this differs explicitly from the local enum grammar.
    body={'type':'string','enum':list(values)} if kind=='routing' else {'type':'array','items':{'type':'string','pattern':'^s[0-9a-f]{12}$'},'maxItems':5,'uniqueItems':True}
    return {'type':'json_schema','json_schema':{'name':'skill_route' if kind=='routing' else 'skill_selection','strict':True,'schema':body}}
def request_body(msg,kind,values):
    return {'model':MODEL,'provider':PROVIDER,'messages':msg,'response_format':schema(kind,values),'temperature':0,'seed':20261024,'max_tokens':128,'reasoning':{'effort':'none'},'stream':False}

class Client:
    def __init__(self,key,budget,tokenizer,out,allowances,read_only=False,profile=None):
        self.key=key;self.budget=budget;self.tokenizer=tokenizer;self.out=out;self.allowances=allowances;self.read_only=read_only
        self.profile=profile;self.model=profile['model'] if profile else MODEL;self.expected_provider=profile['provider_name'] if profile else 'DeepInfra'
    def http(self,suffix,body=None):
        if self.read_only and body is not None:raise BudgetStop('Read-only reconciliation forbids POST')
        assert suffix in ['key','models/'+self.model+'/endpoints','chat/completions'] or re.fullmatch(r'generation\?id=gen-[0-9A-Za-z-]{1,124}',suffix)
        request=urllib.request.Request(API+suffix,data=json.dumps(body).encode() if body is not None else None,headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        entry,started=log_begin(self,suffix,body)
        try:
            with urllib.request.build_opener(NoRedirect()).open(request,timeout=180) as response:payload=decode_http(response.read(),response.status);payload['_cooldown']=cooldown_headers(response.headers)
        except urllib.error.HTTPError as error:
            payload=decode_http(error.read(),error.code);payload['_cooldown']=cooldown_headers(error.headers)
        except Exception as error:
            log_finish(self,entry,started,error=error);raise
        log_finish(self,entry,started,payload=payload);return payload
    def safe_error(self,response):
        error=response.get('error') or {}
        if not isinstance(error,dict):error={'message':str(error)}
        def clean(value):
            text=str(value).replace(self.key,'[REDACTED]') if self.key else str(value)
            text=re.sub(r'\bgen-[0-9A-Za-z-]+\b','[PRIVATE_GENERATION_ID]',text)
            return re.sub(r'(?i)(?:Bearer\s+\S+|sk-(?:or-v1-)?[A-Za-z0-9_-]{12,})','[REDACTED]',text)[:2000]
        metadata=error.get('metadata') or {}
        if not isinstance(metadata,dict):metadata={}
        return {'code':error.get('code',response.get('_http_status')),'message':clean(error.get('message','')),'provider_name':clean(metadata.get('provider_name','')),'provider_detail':clean(metadata.get('raw','')),'limit_source':clean(metadata.get('limit_source',''))}
    def reconcile(self,key,response_id):
        assert re.fullmatch(r'gen-[0-9A-Za-z-]{1,124}',response_id)
        self.active_case=key
        response=self.http('generation?'+urllib.parse.urlencode({'id':response_id}));data=response.get('data') if isinstance(response,dict) else None
        if not isinstance(data,dict):data={}
        identity_ok=data.get('id')==response_id
        if not identity_ok:
            append_jsonl(self.out/'reconciliation.jsonl',{'case_key':key,'response_id':response_id,'status':'identity_mismatch' if data.get('id') is not None else 'identity_missing','id_matches':False,'data_present':'data' in response,'data_type':type(response.get('data')).__name__,'data_keys':sorted(data),'data_id_present':'id' in data,'top_level_id_present':'id' in response,'top_level_id_matches':response.get('id')==response_id,'returned_model':data.get('model'),'returned_provider':data.get('provider_name'),'reported_total_cost':data.get('total_cost'),'http_status':response.get('_http_status',200),'response_keys':sorted(response),'lookup_error':self.safe_error(response),'scope':'Hold retained: saved actual generation ID is absent or different; no new generation'})
            raise BudgetStop('Generation identity not verified; hold retained')
        if data.get('total_cost') is None:
            append_jsonl(self.out/'reconciliation.jsonl',{'case_key':key,'response_id':response_id,'status':'usage_still_unknown','error':self.safe_error(response)})
            raise BudgetStop('Generation usage remains unknown; no expansion')
        try:cost=decimal.Decimal(str(data['total_cost']))
        except decimal.InvalidOperation:cost=decimal.Decimal('NaN')
        if isinstance(data['total_cost'],bool) or not cost.is_finite() or cost<0:
            append_jsonl(self.out/'reconciliation.jsonl',{'case_key':key,'response_id':response_id,'status':'invalid_total_cost_hold_retained','cost_type':type(data['total_cost']).__name__})
            raise BudgetStop('Invalid authoritative total_cost; hold retained')
        receipt={'case_key':key,'response_id':response_id,'status':'provider_usage_reconciled','cost_usd':data['total_cost'],'input_tokens':data.get('native_tokens_prompt') if data.get('native_tokens_prompt') is not None else data.get('tokens_prompt'),'output_tokens':data.get('native_tokens_completion') if data.get('native_tokens_completion') is not None else data.get('tokens_completion'),'provider':data.get('provider_name'),'model':data.get('model'),'provenance_verified':provenance_matches(data.get('model'),data.get('provider_name'),self.model,self.expected_provider),'finish_reason':data.get('finish_reason'),'generation_error':data.get('error'),'scope':'Matched ID and authoritative total_cost settle billing; model/provider validity is a separate gate. No generation replay.'}
        append_jsonl(self.out/'reconciliation.jsonl',receipt);self.budget.settle(key,data['total_cost']);return receipt
    def verify(self,auth_only=False):
        auth=self.http('key');assert 'data' in auth,'Authentication response missing data'
        if auth_only:return {'authenticated':True,'mode':'read_only_reconciliation','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'generation_posts_allowed':False,'credential_storage':'anonymous stdin into process memory only'}
        data=self.http('models/deepseek/deepseek-v4-flash-0731/endpoints')
        endpoint=next(e for e in data['data']['endpoints'] if e['tag']=='deepinfra/fp8')
        assert endpoint['status']==0 and endpoint['context_length']>=1048576
        assert decimal.Decimal(endpoint['pricing']['prompt'])<=decimal.Decimal('.00000006') and decimal.Decimal(endpoint['pricing']['completion'])<=decimal.Decimal('.00000018'),'Price increased'
        assert {'structured_outputs','response_format','seed'}<=set(endpoint['supported_parameters'])
        return {'authenticated':True,'model':MODEL,'provider_pin':PROVIDER,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prices_within_approved_limits':True,'credential_storage':'anonymous stdin into process memory only'}
    def call(self,key,msg,kind,values,allowance):
        if self.read_only:raise BudgetStop('Read-only reconciliation forbids generation')
        self.active_case=key;self.active_kind=kind
        if self.profile:
            from diverse_contract import payload
            body=payload(self.profile,msg,kind,values)
        else:body=request_body(msg,kind,values)
        # Estimate model-visible content, not doubly escaped HTTP wire JSON.
        # The frozen allowance separately reserves 4096 wrapper/schema tokens.
        if self.profile:
            from diverse_contract import input_allowance
            tokens=input_allowance(body)-8192
        else:tokens=sum(len(self.tokenizer.encode(m['role']+'\n'+m['content']).ids) for m in msg)+len(self.tokenizer.encode(json.dumps(body['response_format'])).ids)
        if tokens+512>allowance:raise BudgetStop('Tokenizer/schema count exceeds frozen per-request allowance; no call')
        rate_in=self.profile['input_rate'] if self.profile else '.00000006';rate_out=self.profile['output_rate'] if self.profile else '.00000018'
        self.budget.reserve(key,decimal.Decimal(allowance)*decimal.Decimal(rate_in)+decimal.Decimal(128)*decimal.Decimal(rate_out),model=self.model,run_id='hosted-diverse-v2-20261024' if self.profile else 'hosted-followup-20261024')
        record={'case_key':key,'request_sha256':digest(body),'kind':kind,'reserved_input_tokens':allowance,'estimated_input_tokens':tokens,'output_token_limit':128,'physical_call':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_tokens':None,'output_tokens':None,'cost_usd':None}
        started=time.perf_counter()
        try:
            response=self.http('chat/completions',body)
            # Save ID/HTTP/shape before touching optional usage/choice/message.
            record.update(envelope_identity(response))
            append_jsonl(self.out/'envelopes.jsonl',{**{k:record[k] for k in ['case_key','request_sha256','utc']},**envelope_identity(response)})
            record.update(parse_envelope(response))
            if record.get('http_status')==429:self.rate_limit_halt=True
            record['http_request_id']=getattr(self,'last_http_request_id',None)
            if response.get('error'):record['provider_error']=self.safe_error(response)
            record['wall_seconds']=time.perf_counter()-started;append_jsonl(self.out/'calls.jsonl',record)
            if record['cost_usd'] is None and record.get('response_id'):
                recovered=self.reconcile(key,record['response_id']);record.update(cost_usd=recovered['cost_usd'],input_tokens=recovered['input_tokens'],output_tokens=recovered['output_tokens'],provider=recovered['provider'])
                if recovered['provenance_verified']:record['model']=recovered['model']
            else:self.budget.settle(key,record['cost_usd'])
            if getattr(self,'rate_limit_halt',False):raise BudgetStop('Rate limit response: halt for scope/cooldown review, no automatic model switch')
            if not provenance_matches(record.get('model'),record.get('provider'),self.model,self.expected_provider):raise BudgetStop('Unverified model/provider; expansion halted')
            if record['input_tokens'] is None or record['output_tokens'] is None:raise BudgetStop('Missing usage; expansion halted')
            if record['input_tokens']>allowance or record['output_tokens']>128:raise BudgetStop('Usage exceeded reserved token contract; expansion halted')
            return record
        except BudgetStop:raise
        except Exception as error:
            record.update(status='transport_or_provider_error',error_type=type(error).__name__,http_status=getattr(error,'code',None),wall_seconds=time.perf_counter()-started)
            append_jsonl(self.out/'calls.jsonl',record)
            self.budget.settle(key,None)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime-root',required=True);parser.add_argument('--reconcile-only',action='store_true');args=parser.parse_args();runtime=pathlib.Path(args.runtime_root)
    # The caller closes stdin immediately after an anonymous pipe transfer.
    key=sys.stdin.read().strip()
    if not key or '\n' in key:raise SystemExit('Credential not supplied through the private stdin flow')
    out=runtime/'hosted-execution';out.mkdir(exist_ok=True);budget=Budget(out/'budget-ledger.jsonl','2.00','3.00','1.5',600,model_caps={MODEL:'.60','google/gemma-3-27b-it':'.65','qwen/qwen3-30b-a3b-instruct-2507':'.70'},legacy_model=MODEL)
    prior,_=budget.state()
    from tokenizers import Tokenizer
    pin=read(runtime/'hosted-tokenizer/private-manifest.json');assert pin['repository']=='deepseek-ai/DeepSeek-V4-Flash-0731'
    tok=Tokenizer.from_file(pin['tokenizer.json']);tok.no_truncation();tok.no_padding()
    assert hashlib.sha256(pathlib.Path(pin['tokenizer.json']).read_bytes()).hexdigest()=='8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf'
    with (HERE/'cheaper-token-ledger.csv').open() as f:allowances={(r['query_id'],int(r['size']),r['method']):int(r['reserved_input_tokens']) for r in csv.DictReader(f)}
    client=Client(key,budget,tok,out,allowances,read_only=args.reconcile_only);receipt=client.verify(auth_only=args.reconcile_only);(HERE/'credential-status.json').write_text(json.dumps(receipt,indent=2));print('OpenRouter authentication verified; '+('read-only usage reconciliation' if args.reconcile_only else 'exact provider checked, bounded canaries starting'),flush=True)
    if args.reconcile_only:
        calls={r['case_key']:r for r in rows(out/'calls.jsonl')} if (out/'calls.jsonl').exists() else {}
        for case_id,state in prior.items():
            if state['kind'] in ['reserve','unknown']:
                response_id=calls.get(case_id,{}).get('response_id')
                if not response_id:raise BudgetStop('No generation ID for unresolved charge')
                reconciled=client.reconcile(case_id,response_id)
                print(json.dumps({k:v for k,v in reconciled.items() if k!='response_id'}),flush=True)
        print('Read-only reconciliation ended; no generation requested',flush=True);return
    if any(x['kind'] in ['reserve','unknown'] for x in prior.values()):raise BudgetStop('Unresolved earlier charge; no new request')
    raise BudgetStop('Legacy four-size matrix superseded by the authorized diverse-model protocol; generation remains disabled pending frozen amendment and gate review')
    # Resume only wholly unreserved cases; no semantic retry. Canaries from a
    # prior successful invocation are reused without new paid requests.
    calls={r['case_key']:r for r in rows(out/'calls.jsonl')} if (out/'calls.jsonl').exists() else {}
    for i in range(3):
        cid='nonbenchmark-canary-'+str(i+1)
        if cid not in calls:
            msg=[{'role':'system','content':'Return only the JSON array requested. No tools.'},{'role':'user','content':'Return ["s000000000000"] for this non-evaluation format check.'}]
            calls[cid]=client.call(cid,msg,'selection',[],8192)
        assert canary_valid(calls[cid],MODEL),'Canary response/provenance/finish/schema gate failed; no expansion'
    protocol=read(STUDY/'protocol.json');chosen=set(protocol['sensitivity_query_ids']);panel=[q for q in read(STUDY/'panel.json') if q['id'] in chosen]
    source=STUDY.parents[1]/'scaling/skillret-seed5150-scaling/scaling5150/provenance-audit/public-inputs';skills=rows(source/'skills-metadata.jsonl');docs={s['id']:render(s) for s in skills};major={s['id']:s['major'] for s in skills};domains=sorted(set(major.values()));bm=BM25(docs)
    manifest=read(STUDY/'embedding-manifest.json');dv=np.load(runtime/'docs.npy',mmap_mode='r');qv=np.load(runtime/'queries.npy',mmap_mode='r');di={d:i for i,d in enumerate(manifest['document_ids'])};qi={q:i for i,q in enumerate(manifest['query_ids'])}
    ids=read(STUDY/'catalog-document-ids.json');catalog={(c['query_id'],c['size']):[ids[i] for i in c['document_indices']] for c in json.loads(gzip.decompress((STUDY/'catalogs.json.gz').read_bytes())) if c['seed']==42 and c['query_id'] in chosen}
    labels=read(STUDY/'evaluator-labels.json');casefile=out/'cases.jsonl';done={r['case_id'] for r in rows(casefile)} if casefile.exists() else set()
    for query in panel:
        qid=query['id']
        for n in [10,100,1000,6006]:
            lib=catalog[(qid,n)];scores=dict(zip(lib,map(float,dv[[di[d] for d in lib]]@qv[qi[qid]])));br=bm.search(query['query'],lib);vr=rank(scores,lib)
            for method in ['listing','bm25','vector','domain_vector','hybrid']:
                cid=digest(['hosted',protocol['protocol_sha256'],qid,n,method])
                if cid in done:continue
                row={'case_id':cid,'query_id':qid,'size':n,'seed':42,'method':method,'model':MODEL,'provider_pin':PROVIDER,'panel':'balanced24_exposed','status':'pending','selected_ids':[],'gold_count':len(labels[qid]),'protocol_sha256':protocol['protocol_sha256']};started=time.perf_counter()
                if method=='domain_vector':
                    rkey=cid+'-routing';prompt=chat('Choose the single best supplied category. Return a JSON string.',json.dumps({'request':query['query'],'categories':domains},ensure_ascii=False));route=calls.get(rkey) or client.call(rkey,messages(prompt),'routing',domains,allowances[(qid,n,'routing')]);calls[rkey]=route
                    try:category=json.loads(route['content']);assert category in domains
                    except (ValueError,AssertionError,TypeError):row.update(status='invalid_route',routing=route);append_jsonl(casefile,row);done.add(cid);continue
                    allowed=[d for d in lib if major[d]==category];ranking=rank(scores,allowed);pool=ranking[:5];row['routing']=route;row['routing_gold_retention']=sum(major[d]==category for d in labels[qid])/len(labels[qid])
                else:ranking={'listing':lib,'bm25':br,'vector':vr,'hybrid':rrf(br,vr)}[method];pool=ranking if method=='listing' else ranking[:5]
                row['candidate_ids']=None if method=='listing' else pool;row['retrieval']=None if method=='listing' else retrieval_metrics(ranking,labels[qid]);row['candidate_recall']=len(set(pool)&set(labels[qid]))/len(labels[qid])
                prompt,mapping=select_prompt(query['query'],pool,docs,qid,42)
                answer=calls.get(cid) or client.call(cid,messages(prompt),'selection',mapping,allowances[(qid,n,method)]);calls[cid]=answer;row['selector']=answer
                try:row['selected_ids']=parse_selection(answer['content'],mapping);row['status']='ok'
                except (ValueError,TypeError):row['status']='invalid_selection';row['returned_cardinality_unknown']=True
                row['metrics']=selection_metrics(row['selected_ids'],labels[qid],row['status']);row['method_body_wall_seconds']=time.perf_counter()-started;row['end_to_end_seconds']=None;row['timing_scope']='Method-body wall includes API routing/selection and prompt handling; excludes precomputed query embeddings and BM25/vector scoring done outside this clock. Not end-to-end discovery latency; no direct local latency ranking.';append_jsonl(casefile,row);done.add(cid)
            print('hosted cases durable',len(done),'of480',flush=True)
    _,held=budget.state();(out/'execution.json').write_text(json.dumps({'cases':len(done),'usd_guarded_total':str(held),'usd_cap':'1.10','request_cap':600,'model':MODEL,'provider':PROVIDER,'status':'stage_complete'},indent=2));print('Hosted stage ended; inspect saved receipts',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        # No error payload, request header, credential or provider echo is printed.
        print('HOSTED_HALTED: '+type(error).__name__+'; inspect nonsecret budget/call receipts; no automatic generation retry',file=sys.stderr);raise SystemExit(1)
