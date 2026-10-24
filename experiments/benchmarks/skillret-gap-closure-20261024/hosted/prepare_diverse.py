"""Freeze shared inputs, branch prompts and conservative cost plan before POST."""
import argparse,csv,datetime,decimal,gzip,json,pathlib,sys,urllib.request
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import numpy as np
from core import BM25,digest,render,rank,rrf
from runner import messages,NoRedirect
from diverse_contract import PROFILES,payload,input_allowance,selection_prompt,routing_prompt,canaries
D=decimal.Decimal

def read(p):return json.loads(p.read_text())
def rows(p):
    with p.open() as f:return [json.loads(x) for x in f if x.strip()]

def prepare(runtime):
    study=HERE.parent;out=runtime/'hosted-diverse-v2';out.mkdir(exist_ok=True)
    assert not (out/'cases.jsonl').exists(),'Cannot regenerate frozen inputs after model outcomes'
    chosen=set(read(study/'protocol.json')['sensitivity_query_ids']);panel=[q for q in read(study/'panel.json') if q['id'] in chosen];assert len(panel)==24
    source=study.parents[1]/'scaling/skillret-seed5150-scaling/scaling5150/provenance-audit/public-inputs'
    skills=rows(source/'skills-metadata.jsonl');docs={s['id']:render(s) for s in skills};major={s['id']:s['major'] for s in skills};domains=sorted(set(major.values()));bm=BM25(docs)
    manifest=read(study/'embedding-manifest.json');dv=np.load(runtime/'docs.npy',mmap_mode='r');qv=np.load(runtime/'queries.npy',mmap_mode='r');di={d:i for i,d in enumerate(manifest['document_ids'])};qi={q:i for i,q in enumerate(manifest['query_ids'])}
    ids=read(study/'catalog-document-ids.json');catalog={c['query_id']:[ids[i] for i in c['document_indices']] for c in json.loads(gzip.decompress((study/'catalogs.json.gz').read_bytes())) if c['seed']==42 and c['size']==100 and c['query_id'] in chosen}
    shared=[]
    for query in panel:
        qid=query['id'];lib=catalog[qid];scores=dict(zip(lib,map(float,dv[[di[d] for d in lib]]@qv[qi[qid]])));br=bm.search(query['query'],lib);vr=rank(scores,lib)
        for method in ['listing','bm25','vector','domain_vector','hybrid']:
            entry={'query_id':qid,'size':100,'seed':42,'method':method,'catalog_ids':lib,'branches':{}}
            if method=='domain_vector':entry['route_messages']=messages(routing_prompt(query['query'],domains));rankings={domain:rank(scores,[d for d in lib if major[d]==domain]) for domain in domains}
            else:rankings={'common':{'listing':lib,'bm25':br,'vector':vr,'hybrid':rrf(br,vr)}[method]}
            for branch,ranking in rankings.items():
                pool=ranking if method=='listing' else ranking[:5];prompt,mapping=selection_prompt(query['query'],pool,docs,qid)
                entry['branches'][branch]={'messages':messages(prompt),'mapping':mapping,'candidate_ids':None if method=='listing' else pool,'ranking':ranking,'pool_ids':pool}
            shared.append(entry)
    rows_plan=[];snapshots=[];profiles=[]
    for p in PROFILES:
        endpoint_url='https://openrouter.ai/api/v1/models/'+p['model']+'/endpoints'
        with urllib.request.build_opener(NoRedirect()).open(endpoint_url,timeout=45) as response:data=json.load(response)
        endpoint=next(e for e in data['data']['endpoints'] if e['tag']==p['endpoint']);assert endpoint['status']==0 and endpoint['context_length']>=p['context']
        assert D(endpoint['pricing']['prompt'])<=D(p['input_rate']) and D(endpoint['pricing']['completion'])<=D(p['output_rate'])
        assert {'max_tokens','structured_outputs','response_format'}<=set(endpoint['supported_parameters'])
        if p['seed']:assert 'seed' in endpoint['supported_parameters']
        snapshots.append({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':endpoint_url,'endpoint':endpoint});profiles.append(p)
        for i,c in enumerate(canaries()):
            body=payload(p,c['messages'],c['kind'],c['values']);a=input_allowance(body)
            rows_plan.append({'model':p['model'],'query_id':'canary-'+str(i+1),'method':'canary','stage':c['kind'],'input_allowance':a,'output_limit':128,'context_admitted':a+128<=p['context'],'max_cost_usd':str(D(a)*D(p['input_rate'])+D(128)*D(p['output_rate'])),'request_sha256':digest(body)})
        for e in shared:
            if e['method']=='domain_vector':
                body=payload(p,e['route_messages'],'routing',domains);a=input_allowance(body)
                rows_plan.append({'model':p['model'],'query_id':e['query_id'],'method':e['method'],'stage':'routing','input_allowance':a,'output_limit':128,'context_admitted':a+128<=p['context'],'max_cost_usd':str(D(a)*D(p['input_rate'])+D(128)*D(p['output_rate'])),'request_sha256':digest(body)})
            bodies=[payload(p,b['messages'],'selection',b['mapping']) for b in e['branches'].values()];a=max(map(input_allowance,bodies))
            rows_plan.append({'model':p['model'],'query_id':e['query_id'],'method':e['method'],'stage':'selection','input_allowance':a,'output_limit':128,'context_admitted':a+128<=p['context'],'max_cost_usd':str(D(a)*D(p['input_rate'])+D(128)*D(p['output_rate'])),'request_sha256':digest([digest(b) for b in bodies]) if len(bodies)>1 else digest(bodies[0])})
    plan={'version':'hosted-diverse-v2','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'query_ids':[q['id'] for q in panel],'panel':'balanced24 previously exposed; eight per gold-cardinality stratum','size':100,'seed':42,'methods':['listing','bm25','vector','domain_vector','hybrid'],'profiles':profiles,'shared_inputs_sha256':digest(shared),'request_plan_sha256':digest(rows_plan),'unknown_legacy_hold_usd':'.14','global_cap_usd':'2.00','global_generation_cap':600,'output_limit':128,'input_allowance_definition':'Twice UTF-8 message/schema bytes plus8192 wrapper/control-token allowance. Engineering reserve, not native tokenizer measurement or provider usage. Actual usage must fit.','timing_scope':'API wall and method-body wall; retrieval/query embedding preparation excluded, no comparison with local fresh end-to-end latency.'}
    plan['protocol_sha256']=digest(plan)
    totals=[]
    for p in profiles:
        subset=[r for r in rows_plan if r['model']==p['model']];cost=sum((D(r['max_cost_usd']) for r in subset if r['context_admitted']),D(0));legacy=D('.14') if p['slug']=='deepseek' else D(0)
        assert cost+legacy<=D(p['cap']),(p['model'],cost)
        totals.append({'model':p['model'],'planned_generation_attempts':len(subset),'admitted_attempts':sum(r['context_admitted'] for r in subset),'input_allowance_sum':sum(r['input_allowance'] for r in subset if r['context_admitted']),'output_allowance_sum':128*sum(r['context_admitted'] for r in subset),'conservative_planned_cost_usd':str(cost),'legacy_unknown_hold_usd':str(legacy),'model_cap_usd':p['cap']})
    total=sum((D(t['conservative_planned_cost_usd'])+D(t['legacy_unknown_hold_usd']) for t in totals),D(0));assert total<=2 and 1+len(rows_plan)<=600
    (out/'shared-inputs.json').write_text(json.dumps({'domains':domains,'major':major,'entries':shared},ensure_ascii=False));(out/'frozen-plan.json').write_text(json.dumps(plan,indent=2))
    (HERE/'diverse-protocol.json').write_text(json.dumps(plan,indent=2));(HERE/'diverse-endpoint-snapshots.json').write_text(json.dumps(snapshots,indent=2))
    with (HERE/'diverse-request-plan.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows_plan[0]));w.writeheader();w.writerows(rows_plan)
    result={'status':'frozen_before_new_model_outcomes','models':totals,'total_with_unknown_hold_usd':str(total),'planned_generation_attempts_including_legacy':1+len(rows_plan),'context_excluded_requests':sum(not r['context_admitted'] for r in rows_plan),'protocol_sha256':plan['protocol_sha256']};(HERE/'diverse-budget-plan.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',required=True);args=p.parse_args();prepare(pathlib.Path(args.runtime_root))
