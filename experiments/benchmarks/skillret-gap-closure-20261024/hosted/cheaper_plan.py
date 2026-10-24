"""Recalculate the complete same-panel stage using pinned DeepSeek tokenizer."""
import argparse,csv,datetime,gzip,hashlib,json,pathlib,sys,urllib.request
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;STUDY=HERE.parent
sys.path.insert(0,str(STUDY))
from core import BM25,render,rank,rrf,select_prompt,chat
def read(p):return json.loads(pathlib.Path(p).read_text())
def main():
    a=argparse.ArgumentParser();a.add_argument('--runtime-root',required=True);args=a.parse_args();runtime=pathlib.Path(args.runtime_root)
    from tokenizers import Tokenizer
    source=read(runtime/'hosted-tokenizer/private-manifest.json')
    assert source['repository']=='deepseek-ai/DeepSeek-V4-Flash-0731','Wrong tokenizer repository'
    tokenizer=Tokenizer.from_file(source['tokenizer.json']);tokenizer.no_truncation();tokenizer.no_padding()
    url='https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints'
    with urllib.request.urlopen(url,timeout=60) as response:endpoints=json.load(response)
    endpoint=next(e for e in endpoints['data']['endpoints'] if e['tag']=='deepinfra/fp8')
    assert endpoint['status']==0 and 'structured_outputs' in endpoint['supported_parameters']
    (HERE/'deepinfra-endpoint-snapshot.json').write_text(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':url,'endpoint':endpoint},indent=2))
    protocol=read(STUDY/'protocol.json');chosen=set(protocol['sensitivity_query_ids']);panel=[q for q in read(STUDY/'panel.json') if q['id'] in chosen]
    input_source=STUDY.parents[1]/'scaling/skillret-seed5150-scaling/scaling5150/provenance-audit/public-inputs'
    with (input_source/'skills-metadata.jsonl').open() as f:skills=[json.loads(x) for x in f]
    docs={s['id']:render(s) for s in skills};major={s['id']:s['major'] for s in skills};domains=sorted(set(major.values()));bm=BM25(docs)
    emb=read(STUDY/'embedding-manifest.json');dv=np.load(runtime/'docs.npy',mmap_mode='r');qv=np.load(runtime/'queries.npy',mmap_mode='r');di={d:i for i,d in enumerate(emb['document_ids'])};qi={q:i for i,q in enumerate(emb['query_ids'])}
    ids=read(STUDY/'catalog-document-ids.json');cases=json.loads(gzip.decompress((STUDY/'catalogs.json.gz').read_bytes()));catalog={(c['query_id'],c['size']):[ids[i] for i in c['document_indices']] for c in cases if c['seed']==42 and c['query_id'] in chosen}
    ledger=[];wrapper=4096
    for q in panel:
        for n in [10,100,1000,6006]:
            lib=catalog[(q['id'],n)];scores=dv[[di[d] for d in lib]]@qv[qi[q['id']]];scoremap=dict(zip(lib,map(float,scores)));br=bm.search(q['query'],lib);vr=rank(scoremap,lib)
            pools={'listing':[lib],'bm25':[br[:5]],'vector':[vr[:5]],'hybrid':[rrf(br,vr)[:5]],'domain_vector':[rank(scoremap,[d for d in lib if major[d]==domain])[:5] for domain in domains]}
            for method,possibilities in pools.items():
                counts=[];bytecounts=[]
                for pool in possibilities:
                    prompt,_=select_prompt(q['query'],pool,docs,q['id'],42);counts.append(len(tokenizer.encode(prompt).ids));bytecounts.append(len(prompt.encode()))
                total=max(counts)+wrapper
                ledger.append({'query_id':q['id'],'size':n,'method':method,'estimated_content_tokens':max(counts),'wrapper_schema_token_reserve':wrapper,'reserved_input_tokens':total,'output_token_allowance':128,'prompt_utf8_bytes_max':max(bytecounts),'context_preflight':'estimated_fit; exact provider wrapper pending' if total+128<=endpoint['context_length'] else 'pending_context; not silently omitted','domain_allowance':'maximum over all6 possible routed categories' if method=='domain_vector' else ''})
                if method=='domain_vector':
                    text=chat('Choose the single best supplied category. Return a JSON string.',json.dumps({'request':q['query'],'categories':domains},ensure_ascii=False))
                    count=len(tokenizer.encode(text).ids)
                    ledger.append({'query_id':q['id'],'size':n,'method':'routing','estimated_content_tokens':count,'wrapper_schema_token_reserve':wrapper,'reserved_input_tokens':count+wrapper,'output_token_allowance':128,'prompt_utf8_bytes_max':len(text.encode()),'context_preflight':'estimated_fit; exact provider wrapper pending','domain_allowance':''})
    for i in range(3):ledger.append({'query_id':'nonbenchmark-canary-'+str(i+1),'size':0,'method':'canary','estimated_content_tokens':None,'wrapper_schema_token_reserve':None,'reserved_input_tokens':8192,'output_token_allowance':128,'prompt_utf8_bytes_max':None,'context_preflight':'approved canary required before expansion','domain_allowance':''})
    assert len(ledger)==579
    tokens=sum(x['reserved_input_tokens'] for x in ledger);outputs=sum(x['output_token_allowance'] for x in ledger);content=sum(x['estimated_content_tokens'] or 0 for x in ledger)
    rates=endpoint['pricing'];cost=tokens*float(rates['prompt'])+outputs*float(rates['completion']);margin=cost*1.25;recommended=round(np.ceil(margin*100)/100,2)
    summary={'status':'pending_explicit_approval_and_safe_local_credential; no paid calls','model':'deepseek/deepseek-v4-flash-0731','provider':{'only':['deepinfra/fp8'],'allow_fallbacks':False,'require_parameters':True},'endpoint_context':endpoint['context_length'],'input_rate_usd_per_token':rates['prompt'],'output_rate_usd_per_token':rates['completion'],'cache_savings_assumed':False,'planned_selector_cases':480,'routing_cases':96,'canary_cases':3,'total_request_allowance':579,'request_cap':600,'retries':0,'input_token_allowance':tokens,'output_token_allowance':outputs,'estimated_content_tokens':content,'cost_at_pinned_rates_usd':cost,'planning_usd_with25pct_margin':margin,'recommended_hard_cap_usd':recommended,'cad_estimate_at1_5':recommended*1.5,'cad_per_usd':'planning assumption1.5, not exchange-rate quote','tokenizer':{'repository':source['repository'],'revision':source['revision'],'sha256':hashlib.sha256(pathlib.Path(source['tokenizer.json']).read_bytes()).hexdigest()},'context_pending_rows':sum(x['context_preflight'].startswith('pending_context') for x in ledger),'full_listing_cases_included':96,'largest_reserved_prompt_tokens':max(x['reserved_input_tokens'] for x in ledger),'limits':'HF tokenizer estimate plus4096-token wrapper/schema reserve is not provider-reported usage. All original-size listing cells remain planned. Exact effective wrapper/token/context and structured-JSON canary must pass before paid expansion. Preserve unknown-charge reserves. Do not silently swap this route to V4.1.','credential_requirement':'Locally configured OPENROUTER_API_KEY, never plaintext chat','paid_calls_executed':0}
    (HERE/'cheaper-budget-plan.json').write_text(json.dumps(summary,indent=2))
    with (HERE/'cheaper-token-ledger.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    print(json.dumps(summary))
if __name__=='__main__':main()
