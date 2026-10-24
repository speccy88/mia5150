"""Offline hosted admission/cost plan. No inference endpoint is called."""
import argparse,csv,gzip,json,pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent;STUDY=HERE.parent
sys.path.insert(0,str(STUDY))
from core import render,select_prompt,chat

def read(p):return json.loads(p.read_text())
def main():
    a=argparse.ArgumentParser();a.add_argument('--execute',action='store_true');args=a.parse_args()
    if args.execute:raise SystemExit('Hosted execution is pending explicit approval, local credentials, effective native/bare contract choice and a credentialed canary. This planner cannot send inference.')
    protocol=read(STUDY/'protocol.json');chosen=set(protocol['sensitivity_query_ids']);panel=[q for q in read(STUDY/'panel.json') if q['id'] in chosen]
    src=STUDY.parents[1]/'scaling/skillret-seed5150-scaling/scaling5150/provenance-audit/public-inputs'
    with (src/'skills-metadata.jsonl').open() as f:skills=[json.loads(x) for x in f]
    docs={s['id']:render(s) for s in skills};domains=sorted({s['major'] for s in skills})
    ids=read(STUDY/'catalog-document-ids.json');catalogs=json.loads(gzip.decompress((STUDY/'catalogs.json.gz').read_bytes()));lookup={(c['query_id'],c['size']):[ids[i] for i in c['document_indices']] for c in catalogs if c['seed']==42 and c['query_id'] in chosen}
    longest=sorted(docs,key=lambda d:len(docs[d].encode()),reverse=True)[:5]
    prepared=[]
    for q in panel:
        for n in [10,100,1000,6006]:
            for method in ['listing','bm25','vector','domain_vector','hybrid']:
                # Conservative shortlist bound, independent of any labels/routing result.
                pool=lookup[(q['id'],n)] if method=='listing' else longest
                prompt,_=select_prompt(q['query'],pool,docs,q['id'],42)
                prepared.append({'query_id':q['id'],'size':n,'method':method,'planning_input_token_upper':len(prompt.encode())+4096,'output_token_cap':128,'bound_type':'UTF8 byte count plus4096 wrapper reserve; anticipated interface estimate, executor must preflight actual effective messages before any paid call'})
                if method=='domain_vector':
                    prompt=chat('Choose one supplied category and return a JSON string.',json.dumps({'request':q['query'],'categories':domains}))
                    prepared.append({'query_id':q['id'],'size':n,'method':'routing','planning_input_token_upper':len(prompt.encode())+4096,'output_token_cap':128,'bound_type':'UTF8 byte count plus4096 wrapper reserve'})
    metadata=read(HERE/'endpoint-availability.json');models=read(HERE/'model-availability.json');estimates=[]
    for mid in ['openai/gpt-4o-mini','openai/gpt-4.1-mini','openai/gpt-4.1-nano','deepseek/deepseek-v4-flash-0731']:
        endpoints=metadata['endpoints'][mid].get('data',{}).get('endpoints',[])
        assert endpoints,'No verified public endpoint metadata'
        endpoint=next((e for e in endpoints if e['provider_name']=='OpenAI'),endpoints[0]);rates=endpoint['pricing'];admitted=[r for r in prepared if r['planning_input_token_upper']+128<=endpoint['context_length']]
        excluded=len(prepared)-len(admitted);input_tokens=sum(r['planning_input_token_upper'] for r in admitted)+3*8192;completion_allowance=(len(admitted)+3)*128
        estimate=(input_tokens*float(rates['prompt'])+completion_allowance*float(rates['completion']))*1.25
        estimates.append({'model':mid,'canonical_slug':models['models'][mid]['canonical_slug'],'proposed_provider':endpoint['provider_name'],'endpoint_tag':endpoint['tag'],'context_limit':endpoint['context_length'],'input_rate_usd_per_token':rates['prompt'],'output_rate_usd_per_token':rates['completion'],'planned_selector_cases':480,'fresh_router_cases':96,'canary_calls':3,'estimated_admitted_total_calls':len(admitted)+3,'conservative_context_exclusions':excluded,'planned_input_upper_tokens':input_tokens,'max_output_tokens':completion_allowance,'planning_usd_with25pct_margin':estimate,'planning_cad_at1_5':estimate*1.5,'cap_usd_proposed':5,'status':'pending_approval_and_credentials; no hosted inference attempted','admission':'Do not expand past cap. Full configuration may be infeasible under cap. Separate cache discounts/reasoning/output/failed reserves; no silent provider fallback.'})
    out={'metadata_checked_utc':metadata['checked_utc'],'scope':'24balanced exposed queries,seed42,allfive methods/four original sizes, four original hosted model levels; separate from200-query local primary','credential_requirement':'Local provider credential through private input; no chat disclosure','paid_calls_executed':0,'inference_availability':'Public metadata only; real capability and returned identity require approved canary','proposed_global_cap_usd':10,'proposed_global_cap_cad_at1_5':15,'caps_policy':'Global10USD and per-model5USD must both hold; do not assume all planned models/cases fit. No permission to spend granted by this file.','output_limit':128,'retries':0,'physical_route_policy':'Fresh route per domain case for comparable end-to-end latency','budget_enforcement_to_reuse':'Historical pilot Budget persistent reserve/lock pattern; do not reset old ledgers. Native runtime compatibility/effective-payload canary must be verified before admission.','estimates':estimates}
    (HERE/'budget-plan.json').write_text(json.dumps(out,indent=2))
    with (HERE/'budget-plan.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(estimates[0]));w.writeheader();w.writerows(estimates)
    print(json.dumps({'models':[{k:r[k] for k in ['model','proposed_provider','estimated_admitted_total_calls','planning_usd_with25pct_margin','conservative_context_exclusions']} for r in estimates],'proposed_global_cap_usd':10,'paid_calls':0}))
if __name__=='__main__':main()
