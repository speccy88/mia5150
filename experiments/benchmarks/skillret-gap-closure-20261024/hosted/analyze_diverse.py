"""Regenerate balanced diagnostic tables, paired uncertainty and charts from receipts."""
import argparse,collections,csv,datetime,decimal,json,pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import numpy as np
from core import selection_metrics
from diverse_contract import PROFILES,parse
from envelope import valid_result
D=decimal.Decimal

def read(p):return json.loads(p.read_text())
def rows(p):
    if not p.exists():return []
    with p.open() as f:return [json.loads(x) for x in f if x.strip()]
def write_csv(p,items):
    if not items:return
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)

def analyze(runtime):
    out=HERE/'reports';cases=rows(runtime/'hosted-diverse-v2/cases.jsonl');calls=rows(runtime/'hosted-execution/calls.jsonl');lookup={r['case_key']:r for r in calls};pmap={p['model']:p for p in PROFILES};summary=[];pair=[];plan=read(HERE/'diverse-protocol.json')
    assert len(cases)==len({r['case_id'] for r in cases})
    # Rescore decisions and validate every successful response independently of
    # the stored metrics. Domain routing evidence is validated separately.
    shared=read(runtime/'hosted-diverse-v2/shared-inputs.json');inputs={(e['query_id'],e['method']):e for e in shared['entries']}
    for row in cases:
        assert row['metrics']==selection_metrics(row['selected_ids'],row['gold_ids'],row['status'])
        if row['status']=='ok':
            answer=lookup[row['selector_attempt_key']];p=pmap[row['model']];assert valid_result(answer,p['model'],p['provider_name']) and answer['http_status']==200
            e=inputs[(row['query_id'],row['method'])];branch=row.get('domain','common');assert parse(answer['content'],'selection',e['branches'][branch]['mapping'])==row['selected_ids']
            if row['method']=='domain_vector':
                route=lookup[row['route_attempt_key']];assert valid_result(route,p['model'],p['provider_name']) and route['http_status']==200;assert parse(route['content'],'routing',shared['domains'])==row['domain']
    bymodel=collections.defaultdict(list)
    for row in cases:bymodel[row['model']].append(row)
    for model,model_rows in bymodel.items():
        for method in plan['methods']:
            subset=[r for r in model_rows if r['method']==method];n=len(subset);valid=[r for r in subset if r['status']=='ok'];keys=[r[k] for r in subset for k in ['selector_attempt_key','route_attempt_key'] if k in r];physical=[lookup[k] for k in keys]
            summary.append({'model':model,'method':method,'catalog_size':100,'planned':24,'attempted':n,'valid':len(valid),'unattempted':24-n,'exact_count':sum(r['metrics']['exact'] for r in subset) if n else None,'balanced_exact_rate':sum(r['metrics']['exact'] for r in subset)/n if n else None,'balanced_macro_f1':sum(r['metrics']['f1'] for r in subset)/n if n else None,'mean_recall5_retrieval':float(np.mean([r['retrieval']['recall5'] for r in subset if r.get('retrieval')])) if any(r.get('retrieval') for r in subset) else None,'physical_generation_requests':len(physical),'input_tokens_sum':sum(r['input_tokens'] for r in physical if r.get('input_tokens') is not None),'input_usage_missing':sum(r.get('input_tokens') is None for r in physical),'output_tokens_sum':sum(r['output_tokens'] for r in physical if r.get('output_tokens') is not None),'output_usage_missing':sum(r.get('output_tokens') is None for r in physical),'known_billed_cost_usd':str(sum((D(str(r['cost_usd'])) for r in physical if r.get('cost_usd') is not None),D(0))),'unknown_cost_requests':sum(r.get('cost_usd') is None for r in physical),'mean_method_body_wall_seconds':float(np.mean([r['method_body_wall_seconds'] for r in subset])) if n else None,'timing_scope':'API/prompt method body; excludes offline retrieval/embedding; not local fresh end-to-end latency','status':'complete' if n==24 else 'partial' if n else 'not_run'})
        ref={r['query_id']:r for r in model_rows if r['method']=='bm25'}
        for method in ['listing','vector','domain_vector','hybrid']:
            alt={r['query_id']:r for r in model_rows if r['method']==method};ids=sorted(set(ref)&set(alt))
            if len(ids)!=24:continue
            strata=[[i for i,q in enumerate(ids) if len(ref[q]['gold_ids'])==n] for n in [1,2,3]];assert list(map(len,strata))==[8,8,8]
            rng=np.random.default_rng(20261024);indices=np.concatenate([rng.choice(s,(10000,8),replace=True) for s in strata],axis=1)
            for metric in ['exact','f1']:
                delta=np.array([alt[q]['metrics'][metric]-ref[q]['metrics'][metric] for q in ids]);dist=delta[indices].mean(1);lo,hi=np.quantile(dist,[.025,.975])
                pair.append({'model':model,'method':method,'reference':'bm25','metric':metric,'n_independent_exposed_queries':24,'balanced_mean_difference':float(delta.mean()),'paired_stratified_bootstrap95_low':float(lo),'paired_stratified_bootstrap95_high':float(hi),'bootstrap_samples':10000,'seed':20261024,'scope':'descriptive nominal intervals; no multiplicity-adjusted superiority claim'})
    write_csv(out/'hosted-selection.csv',summary);write_csv(out/'hosted-paired-bootstrap.csv',pair)
    state={}
    for event in rows(runtime/'hosted-execution/budget-ledger.jsonl'):
        if event['kind']=='reserve':state[event['case_key']]=dict(event)
        else:state[event['case_key']].update(event)
    known=sum((D(x['actual_usd']) for x in state.values() if x.get('actual_usd') is not None),D(0));holds=sum((D(x['reserve_usd']) for x in state.values() if x.get('actual_usd') is None),D(0));unknown=sum(x.get('actual_usd') is None for x in state.values())
    receipt={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed','canonical_cases_rescored':len(cases),'summary_cells':len(summary),'paired_intervals':len(pair),'generation_attempts_including_all_canaries_failures':len(state),'known_billed_sum_usd':str(known),'unknown_hold_usd':str(holds),'unknown_attempts':unknown,'actual_total_cost_usd':None if unknown else str(known),'guarded_total_usd':str(known+holds),'global_cap_usd':'2.00','global_request_cap':600,'scope':'Owner-run reproducibility check of saved hosted outputs, not independent review, human label validation or native SkillsMiddleware.'};assert known+holds<=2 and len(state)<=600;(out/'hosted-validation.json').write_text(json.dumps(receipt,indent=2))
    if summary:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4));names=[r['method'].replace('domain_vector','Domain vector').replace('bm25','BM25').capitalize() if r['method']!='bm25' else 'BM25' for r in summary];x=np.arange(len(names));axes[0].bar(x,[100*r['balanced_exact_rate'] if r['balanced_exact_rate'] is not None else np.nan for r in summary],color='#146c94');axes[0].set(ylabel='Exact-set selection (%)',title='Qwen / SiliconFlow: 24 exposed queries',ylim=(0,100));axes[0].set_xticks(x,names,rotation=25,ha='right');axes[1].bar(x,[float(r['known_billed_cost_usd']) for r in summary],color='#159895');axes[1].set(ylabel='Known billed USD',title='Benchmark routing + selection cost');axes[1].set_xticks(x,names,rotation=25,ha='right');fig.suptitle('Shared N=100 diagnostic; earlier UNKNOWN holds excluded from cost bars');fig.tight_layout();fig.savefig(out/'hosted-selection-cost.png',dpi=180);plt.close(fig)
    lines=['# Hosted diagnostic follow-up','',f"Recorded {len(cases)}/360 originally planned model/method cases. Qwen is the sole active benchmark model after the provider-specific continuation amendment; its target is120 cases. DeepInfra models remain unattempted beyond their preserved canaries.",'','The shared panel has24 previously exposed official queries: eight each with one, two and three positive qrels. Unweighted results estimate this balanced diagnostic, not official full-test prevalence. The object-wrapper v2 output contract differs from the local grammar. Retrieval inputs are reused; timing excludes retrieval and query embedding preparation.','',f"Physical generation attempts, including all canaries and failures: **{len(state)}**. Known billed generation costs: **US${known}**. Unknown liabilities: **US${holds}** across{unknown} attempts. Actual total spend remains **unknown**. Guarded total is **US${known+holds}**, below the shared US$2/600 ceiling. These totals include canaries and routing; read-only HTTP events are separate.",'','|Method|Attempted / planned|Valid|Exact sets|Balanced macro F1|Known benchmark USD|','|---|---:|---:|---:|---:|---:|']
    for r in summary:
        f1=f"{r['balanced_macro_f1']:.4f}" if r['balanced_macro_f1'] is not None else 'pending'
        lines.append(f"|{r['method']}|{r['attempted']} /24|{r['valid']}|{int(r['exact_count']) if r['exact_count'] is not None else 'pending'}|{f1}|{r['known_billed_cost_usd']}|")
    lines+=['','Paired stratified bootstrap intervals resample independent query IDs within the three gold-cardinality strata, keeping methods paired. They are descriptive nominal95% intervals over24 exposed requests; no broad superiority, generalization, causal model-size or original80-cell closure claim follows. Failed attempted decisions retain zero exact/F1 under the declared rule; unattempted cases remain unmeasured.','', 'See [selection table](hosted-selection.csv), [paired intervals](hosted-paired-bootstrap.csv), [owner validation](hosted-validation.json), [physical API ledger](api-call-ledger.jsonl) and [guard status](status.json).']
    text='\n'.join(lines)+'\n'
    for old,new in [('target is120','target is 120'),('has24','has 24'),('across2','across 2'),('over24','over 24'),('nominal95','nominal 95'),('original80','original 80'),(' /24',' / 24')]:text=text.replace(old,new)
    terminal_path=HERE/'mistral-reports/terminal-execution.json'
    terminal=read(terminal_path) if terminal_path.exists() else None
    if terminal:
        text+='\n## Later exploratory Mistral arm and terminal scope\n\nThe later arm was frozen after Qwen results on the same24 queries, N=100 catalogs, prompts, output contract and budget rules. Its Parasail BF16 deployment differs from Qwen/SiliconFlow FP8, so a successful comparison would still have provider/precision/timing confounding. Its first canary returned HTTP429 with typed source upstream_provider_shared_pool, no Retry-After header, no generation ID and no usage/cost. No Mistral selector or routing case was attempted; no retry or further model followed.\n\nThe original frozen selector plan remains360 cases:120 completed Qwen and240 unattempted DeepSeek/Gemma. The later added120 Mistral cases are separately unattempted. Combined:480 planned,120 completed,360 unattempted. No unattempted cell is assigned a performance zero. Only one model has successful benchmark evidence; a multi-model quality comparison remains incomplete.\n\nTerminal generation count is150:147 successful Qwen requests plus three failed provider canaries. Recorded read-only GET count is9 (earlier uninstrumented total remains unknown). Known charges stay US$0.02411946. Conservative unknown holds are US$0.14 DeepSeek, US$0.02 Gemma and US$0.03 Mistral; the final increase retains STOP and grants no continuation. Actual total cost is unknown; guarded total is US$0.21411946. See [terminal receipt](../mistral-reports/terminal-execution.json) and [unattempted Mistral cells](../mistral-reports/unattempted-mistral-cells.csv).\n'
    for a,b in [('same24','same 24'),('HTTP429','HTTP 429'),('remains360','remains 360'),(':120',': 120'),('and240','and 240'),('added120','added 120'),('Combined:480','Combined: 480'),(',120',', 120'),(',360',', 360'),('is150','is 150'),(':147',': 147'),('is9','is 9')]:text=text.replace(a,b)
    (out/'HOSTED_REPORT.md').write_text(text);print(json.dumps(receipt))
    counts='; '.join(f"{r['method'].replace('_',' ')} {int(r['exact_count'])}/{r['attempted']}" for r in summary if r['exact_count'] is not None)
    paragraph=f"The separate Qwen/SiliconFlow diagnostic uses 24 exposed balanced queries at 100 skills. Exact sets: {counts}. Known reported charges including successful canaries and routing total US${known}, all from Qwen in this completed stage. Other failed provider requests retain US${holds} UNKNOWN liability holds. Actual total cost remains unknown. There are {len(state)} cumulative generation attempts. These results are not official-prevalence estimates, a causal local-versus-hosted model comparison, native task execution, or completion of the original hosted matrix."
    if terminal:paragraph+=' The later Mistral/Parasail BF16 arm returned an upstream429 at its first canary, with no usage or known charge, and attempted no benchmark cases. Original planned selectors:360, completed120; later added selectors:120, completed0. Unattempted cells remain unmeasured. Only Qwen provides successful benchmark quality; multi-model comparison remains incomplete. No further generation is planned.'
    (out/'presentation-addendum.md').write_text('## Hosted follow-up diagnostic\n\n'+paragraph+'\n')
    tex=paragraph.replace('_',r'\_').replace('$',r'\$')
    (out/'paper-addendum.tex').write_text(r'\section{Separate hosted diagnostic follow-up}'+tex+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',required=True);analyze(pathlib.Path(p.parse_args().runtime_root))
