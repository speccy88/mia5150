"""Audit and summarize only this frozen local scaling experiment."""
import collections,csv,hashlib,json,statistics,sys,zipfile,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'scaling5150';sys.path.insert(0,str(ROOT))
from harness import evaluate,set_metrics
def read(p):return json.loads(p.read_text())
def main():
 m=read(HERE/'manifest.json');labels=read(HERE/'evaluator-qrels.json');cases={c['case_id']:c for c in m['cases']};out=HERE/'report';out.mkdir(exist_ok=True);tables=[];audits=[]
 retrieval=read(HERE/'retrieval-full/results.json');assert retrieval['case_count']==120
 for row in retrieval['rows']:
  lib=cases[row['case_id']]['skill_ids'];assert len(row['ranking'])==len(set(row['ranking'])) and set(row['ranking'])<=set(lib)
 for key,ev in retrieval['metrics'].items():
  method,size=key.split('/');assert ev['planned_queries']==ev['attempted_queries']==ev['valid_queries']==20
  rows=[r for r in retrieval['rows'] if r['method']==method and r['size']==int(size)];calc=evaluate({r['query_id']:labels[r['query_id']] for r in rows},{r['query_id']:r['ranking'] for r in rows})['aggregate'];assert calc==ev['aggregate'];tables.append({'kind':'retrieval','model':method,'method':method,'size':int(size),'planned':20,'attempted':20,'valid':20,**calc})
 audits.append({'retrieval_rows_verified':len(retrieval['rows']),'shared_library_membership':True,'gold_never_inserted_into_rankings_by_evaluator':True})
 select=[];rerank=[];statuses=[]
 for role,subdir,expected in [(r,f'selection-{r}-full',60) for r in ['small_qwen06','medium_qwen35']]+[(r,f'domain-constrained-{r}',12) for r in ['small_qwen06','medium_qwen35']]:
  p=HERE/subdir/'results.json'
  if not p.exists():statuses.append({'role':role,'status':'no_completed_full_result'});continue
  result=read(p);assert len(result['rows'])==expected
  for r in result['rows']:
   assert set(r['candidate_ids'])<=set(cases[r['case_id']]['skill_ids']);assert set(r['selected_skill_ids'])<=set(r['candidate_ids'])
   if r['method']=='full_listing':assert r['candidate_ids']==cases[r['case_id']]['skill_ids']
   if r['status']!='ok':assert r['selected_skill_ids']==[]
  for key,ev in result['metrics'].items():
   method,size=key.split('/');rr=[r for r in result['rows'] if r['method']==method and r['size']==int(size)];assert len(rr)==2
   for r in rr:assert set_metrics(r['selected_skill_ids'],labels[r['query_id']])==ev['set_per_query'][r['query_id']]
   tables.append({'kind':'final_selection','model':role,'method':method,'size':int(size),'planned':2,'attempted':len(rr),'valid':ev['valid'],**ev['aggregate'],**{'set_'+k:v for k,v in ev['set_aggregate'].items()}})
  select.append((role,result));statuses.append({'role':role,'artifact':subdir,'status_counts':dict(collections.Counter(r['status'] for r in result['rows']))})
 for role in ['small_skillret','medium_jevk5','fit_qwen8']:
  p=HERE/f'rerank-{role}/results.json'
  if not p.exists():statuses.append({'role':role,'status':'no_completed_result; inspect guard logs'});continue
  result=read(p);assert len(result['rows'])==6
  for r in result['rows']:
   assert set(r['ranking'])<=set(r['pool']);assert r['pool']==next(x['ranking'][:20] for x in retrieval['rows'] if x['case_id']==r['case_id'] and x['method']=='skillret_embed_06')
   if r['status']!='ok':assert r['ranking']==[]
  for size,ev in result['metrics'].items():tables.append({'kind':'rerank','model':role,'method':'common_dense20','size':int(size),'planned':2,'attempted':ev['attempted'],'valid':ev['valid'],**ev['aggregate']})
  rerank.append((role,result));statuses.append({'role':role,'status_counts':dict(collections.Counter(r['status'] for r in result['rows']))})
 performance=[]
 for role,result in select:
  for r in result['rows']:
   a=r.get('answer',{});route_in=r.get('logical_routing_input_tokens') or 0;route_out=r.get('logical_routing_output_tokens') or 0
   performance.append({'stage':'selection','model':role,'case_id':r['case_id'],'method':r['method'],'status':r['status'],'offered_input_tokens':a.get('input_tokens'),'logical_routing_input_tokens':route_in,'logical_routing_output_tokens':route_out,'logical_total_offered_tokens':(a.get('input_tokens') or 0)+(a.get('output_tokens') or 0)+route_in+route_out,'selection_output_tokens':a.get('output_tokens'),'cache_hit':a.get('cache_hit'),'measured_replay_or_fresh_seconds':r['selection_stage_wall_seconds'],'original_completion_seconds_if_cached':a.get('cached_original_physical_seconds'),'fresh_inference_seconds':None if a.get('cache_hit') else a.get('physical_seconds'),'note':'Offered tokens on context_overflow were tokenized only, not processed by completion. Selection stage excludes retrieval and route wall time; route costs appear in raw route records.'})
 for role,result in rerank:
  for r in result['rows']:performance.append({'stage':'rerank','model':role,'case_id':r['case_id'],'method':'common_dense20','status':r['status'],'offered_input_tokens':sum(a['input_tokens'] for a in r['answers']),'cache_hit':sum(a['cache_hit'] for a in r['answers']),'measured_replay_or_fresh_seconds':r['wall_seconds'],'fresh_inference_seconds':sum(a['seconds'] for a in r['answers'] if not a['cache_hit']),'note':'Cache_hit is number of candidate forwards reused. Raw answers retain original call seconds; wall time includes orchestration.'})
 if performance:
  keys=list(dict.fromkeys(k for row in performance for k in row))
  with (out/'request-costs.csv').open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(performance)
 fields=list(dict.fromkeys(k for row in tables for k in row))
 with (out/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(tables)
 resources=[]
 for run in sorted((ROOT/'runs').glob('scaling5150-*')):
  if not run.is_dir():continue
  tp=run/'telemetry.jsonl';ts=[json.loads(x) for x in tp.read_text().splitlines()] if tp.exists() else [];exit=read(run/'exit.json') if (run/'exit.json').exists() else None;stop=read(run/'guard-stop.json') if (run/'guard-stop.json').exists() else None
  valid=[t for t in ts if t.get('host') and t.get('guest_available') is not None]
  resources.append({'run':run.name,'exit':exit,'guard_stop':stop,'samples':len(ts),'min_host_free_gib':min((t['host']['available_bytes']/2**30 for t in valid),default=None),'min_guest_free_gib':min((t['guest_available']/2**30 for t in valid),default=None),'min_gpu_free_mib':min((t['gpu_free_mib'] for t in valid),default=None),'max_gpu_temp_c':max((t['gpu_temp'] for t in valid),default=None),'max_worker_rss_gib':max((t.get('worker_rss_bytes',0)/2**30 for t in valid),default=None),'max_worker_cpu_seconds':max((t.get('worker_cpu_seconds',0) for t in valid),default=None)})
 (out/'resource-summary.json').write_text(json.dumps(resources,indent=2));(out/'audit.json').write_text(json.dumps({'audits':audits,'statuses':statuses,'independent_library_audit':read(HERE/'independent-audit.json'),'new_paid_calls':0,'new_downloads':0},indent=2))
 lines=['# MIA5150: seed-5150 local library scaling','', 'This is a new, free local experiment. It is separate from every prior TypeSafe, routing, LoRA and six-query screen. No paid calls, model downloads, uploads or Git push were used.','', 'Twenty fixed public development queries share the same nested gold-conditioned libraries across methods. The first two form the separate paired generation and reranking diagnostic. Libraries contain all official positives plus seeded negatives; gold is not inserted into retrieved shortlists or disclosed in model prompts. All 4,392 public queries have prior exposure; these results are not an untouched final test. Known labels may not exhaust all semantically valid skills.','', 'Sizes: 10, 50, 100, 250, 1,000 and 6,006. The union retains the supplied PDF grid and the newer repository grid supplied by the parent reviewer. The repository project_proposal.tex/reference_metrics.py were not found locally; direct inspection here used the supplied PDF and research-plan contract. Source hashes and the exact seed/library algorithm are in protocol.json, manifest.json and independent-audit.json.','', '## Retrieval: 20 queries per cell','', '| Method | Library | Recall@5 | Recall@10 | NDCG@10 | MRR@5 | Completeness@5 |','|---|---:|---:|---:|---:|---:|---:|']
 primary=['bm25','qwen_embed_06','skillret_embed_06','eg2_text270m_fp32','jina_nano239m_fp32','hybrid_rrf60','query_clause_bm25']
 for size in m['sizes']:
  for method in primary:
   ev=retrieval['metrics'].get(f'{method}/{size}')
   if ev: a=ev['aggregate'];lines.append(f"| {method} | {size} | {a['Recall@5']:.4f} | {a['Recall@10']:.4f} | {a['NDCG@10']:.4f} | {a['MRR@5']:.4f} | {a['Completeness@5']:.4f} |")
 lines+=['','All additional specialist, field, centroid, MMR/DSR and matched candidate-budget arms are in metrics.csv and raw retrieval results. Candidate recall/oracles and fixed-top5 set diagnostics are stored separately; a ranked top5 is not a model-generated final set. Candidate-budget comparisons must use dense20 versus roundrobin20 and dense30 versus dense20+10 novel BM25.','', '## Final selection: two-query paired diagnostic','', '| Model | Method | Library | Valid / planned | Set precision | Set recall | Set F1 | Exact set |','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in tables:
  if r['kind']=='final_selection':lines.append(f"| {r['model']} | {r['method']} | {r['size']} | {r['valid']}/{r['planned']} | {r['set_precision']:.4f} | {r['set_recall']:.4f} | {r['set_f1']:.4f} | {r['set_exact']:.4f} |")
 lines+=['','The common selector sees BM25, vector, LM-domain-vector or hybrid top5, or every actual library item for full listing. LM routing predicts one of six categories from query text alone; no fallback is used. Centroid routing is a separate retrieval ablation. Aliases and display order are independent of labels. Invalid output, empty domain and full-listing context overflow retain zero scores and the planned denominator. Context is 8,192 tokens with a 256-token output allowance; overflowing libraries are not secretly shortened.','', 'Original unconstrained domain-route responses all violated the required JSON-string schema (objects/fences instead); all original zeros remain. domain_vector_lm_constrained_route is a separately frozen infrastructure ablation that adds only a six-category output grammar to routing. It uses no gold category and leaves the selection prompt/parser unchanged. Exact old selection responses, including failures, are reused rather than retried. Compare this separate arm explicitly, not as a silent correction to original predictions.','', '## Common-pool reranking: two queries, top20, three sizes','', '| Model | Library | Valid / planned | Recall@5 | Recall@10 | NDCG@10 |','|---|---:|---:|---:|---:|---:|']
 for r in tables:
  if r['kind']=='rerank':lines.append(f"| {r['model']} | {r['size']} | {r['valid']}/{r['planned']} | {r['Recall@5']:.4f} | {r['Recall@10']:.4f} | {r['NDCG@10']:.4f} |")
 lines+=['','Native model prompts/readouts differ and are pinned; this is a deployment comparison, not a parameter-count-only experiment. Missing native option logits invalidate the whole case. No gold items are added to these pools.','', 'The 8B Q4 file passed local GGUF extent validation and loaded under the unchanged guard at context2,048. Warmup and one candidate returned; the next candidate raised ValueError in native scoring, so the whole first case failed and five remained unattempted. The failed response was not persisted before the exception, so its exact cause cannot be conclusively recovered from the saved raw data. All six planned cases remain zero for strict accounting; these are operational failures, not evidence of poor intrinsic ranking quality. No repeated failed request or widened logprob tuning was performed.','', '## Cost, fit and timing','', 'All new work costs USD0. Small retrieval tiers are 239M/270M FP32 and 0.6B FP16. The small selector is Qwen3-0.6B Q8; the medium selector is Qwen3.5-4B Q4. JevK5-4B Q5 and cached Qwen3-Reranker-8B Q4 are separate reranker candidates. An 8B file-size estimate is not a fit result. The guard traces include runtime memory, context/KV startup, host/guest headroom, temperature and swap; failed fits are preserved. No global-best claim follows from this small panel.','', 'WSL remains 6GiB RAM, two processors and 2GiB swap. GPU workloads run sequentially. The guard requires host/guest free RAM >=2GiB, free VRAM >=1GiB, temperature <=80C, fresh host telemetry and no active swap.','', 'Index creation, retrieval CPU/search, query forwards, startup, reranking and selection are recorded separately. Identical old embedding vectors and exact repeated generation requests can be reused. A cache hit is not zero-cost serving inference: cached query times and original physical completion times retain their provenance. Null historical timings remain unknown. Do not divide replay wall time by full-listing latency or call it measured end-to-end speedup. Logical route tokens are charged to every domain-vector case, while physical reuse is recorded separately. Full fresh end-to-end latency is not measured for every cell.','', 'Resource details: resource-summary.json. Guard worker CPU/RSS excludes child llama-server. Additional evidence/*-server-resources.json captures child CPU and peak RSS for the later medium full-selector, rerankers and constrained-domain runs; early selector pilot/small-full child accounting is unavailable. System-wide RAM/GPU samples are available for every guard run. Per-request logical tokens, raw model responses, cache flags and timings are in the raw selection/reranking results. New embedding call logs include input token counts/truncation, CPU and wall time.','', '## Scope and limits','', 'Implemented and tested: deterministic common libraries; gold isolation; exact vector/BM25 retrieval; field/clause/fusion/structural extensions; independent metric checks; bounded local selector and common-pool reranker pipelines; raw/CSV/report export. Execution status below is authoritative—implemented is not equivalent to successfully run.','', 'Not run in this frozen experiment: hosted APIs/RunPod, new downloads, BGE-small and CLM without verified local artifacts, fresh LoRA/training, graph/dependency methods lacking annotations, hard-overlap distractor expansion, larger generative panels, and Plumb retry after its earlier resource failure. Historical experiments are not counted as new scaling results.','', 'Status details:','', '```json',json.dumps(statuses,indent=2),'```','', '## Reproduction','', 'Use the existing isolated .venv in /home/user/code/mia5150-local. Preserve existing output directories; scripts intentionally refuse overwriting experimental runs. Set MIA_HOST_TELEMETRY to the current Windows monitor JSONL and run every compute stage through guard.py. See commands.txt. Local cached model paths/revisions/hashes are pinned; weights and credentials are not in the portable evidence package.']
 import numpy as np
 rng=np.random.default_rng(5150);draws=rng.integers(0,20,size=(2000,20));paired=[];qs=[q['id'] for q in m['queries']];baseline=retrieval['metrics']['bm25/6006']
 for method in primary[1:]:
  ev=retrieval['metrics'].get(method+'/6006')
  if not ev:continue
  for metric in ['Recall@5','NDCG@10']:
   delta=np.array([ev['per_query'][q][metric]-baseline['per_query'][q][metric] for q in qs]);lo,hi=np.quantile(delta[draws].mean(1),[.025,.975]);paired.append({'method':method,'metric':metric,'mean_delta':float(delta.mean()),'paired_bootstrap95':[float(lo),float(hi)],'wins':int((delta>1e-12).sum()),'ties':int((abs(delta)<=1e-12).sum()),'losses':int((delta< -1e-12).sum()),'n':20,'resamples':2000,'seed':5150})
 (out/'paired-bootstrap.json').write_text(json.dumps({'rows':paired,'caveat':'Descriptive paired intervals on exposed20-query panel, not multiplicity-adjusted; no promotion/tuning from these results.'},indent=2))
 headline=['At 6,006 skills on the same 20 queries:']
 for method in primary:
  ev=retrieval['metrics'].get(method+'/6006')
  if ev:headline.append(f"- {method}: Recall@5 {ev['aggregate']['Recall@5']:.4f}; NDCG@10 {ev['aggregate']['NDCG@10']:.4f}.")
 lines[2:2]=headline+['','Paired descriptive intervals are in paired-bootstrap.json. This small exposed panel does not establish a general model ranking.','']
 fitlines=['','## Observed practical fit','', '| Guarded run | Min host free GiB | Min guest free GiB | Min GPU free MiB | Max GPU C |','|---|---:|---:|---:|---:|']
 for r in resources:
  if any(t in r['run'] for t in ['selection-medium_qwen35-full','domain-constrained-small','rerank-']):
   if r['min_host_free_gib'] is not None:fitlines.append(f"| {r['run']} | {r['min_host_free_gib']:.2f} | {r['min_guest_free_gib']:.2f} | {r['min_gpu_free_mib']} | {r['max_gpu_temp_c']} |")
 fitlines+=['','For the tested final-selector deployment, Qwen3.5-4B Q4 is usable at context8,192; the two-query hybrid final-set F1 is0.4500 versus0.3429 for Qwen3-0.6B Q8. Neither achieved an exact set at6,006. This is a narrow diagnostic, not proof that4B is globally optimal. The8B reranker fits context2,048 but its current scoring adapter is not usable, so it cannot yet displace a working smaller reranker.','', 'On the two shared full-library reranking cases, the SkillRet0.6B reranker improves Recall@5 from the dense pool order0.6667 to0.8333 and NDCG@10 from0.7346 to0.8254. JevK5-4B lowers Recall@5 to0.1667 and NDCG@10 to0.4691 on the same pool. Larger is not consistently better here. Small-scope reranker results must not be pooled with the20-query retrieval table.']
 lines+=fitlines
 # Standalone scientific SVG: common axes, no external JS/assets.
 colors=['#444444','#cc6677','#228833','#4477aa','#aa3377','#ee7733','#009988'];svg=['<svg xmlns="http://www.w3.org/2000/svg" width="980" height="500" viewBox="0 0 980 500">','<rect width="980" height="500" fill="white"/>','<g font-family="sans-serif" font-size="13" fill="#222">','<text x="65" y="28" font-size="18">Recall@5 vs gold-conditioned library size — 20 fixed public queries</text>']
 def x(n):return 70+650*(math.log10(n)-1)/(math.log10(6006)-1)
 def y(v):return 400-330*v
 for tick in [0,.25,.5,.75,1]:svg.extend([f'<path d="M70 {y(tick)} H720" stroke="#ddd"/>',f'<text x="25" y="{y(tick)+4}">{tick:g}</text>'])
 for n in m['sizes']:svg.append(f'<text x="{x(n)}" y="425" text-anchor="middle">{n}</text>')
 svg.append('<text x="300" y="455">Library size (log scale)</text>')
 for i,method in enumerate(primary):
  pts=[(x(n),y(retrieval['metrics'][f'{method}/{n}']['aggregate']['Recall@5'])) for n in m['sizes'] if f'{method}/{n}' in retrieval['metrics']]
  if not pts:continue
  svg.append('<polyline fill="none" stroke="'+colors[i]+'" stroke-width="2" points="'+' '.join(f'{a:.2f},{b:.2f}' for a,b in pts)+'"/>')
  for a,b in pts:svg.append(f'<circle cx="{a}" cy="{b}" r="3" fill="{colors[i]}"/>')
  svg.append(f'<text x="745" y="{65+i*27}" fill="{colors[i]}">{method}</text>')
 svg+=['<text x="65" y="483">Gold is in each library, never forced into shortlists. No untouched-test or full-discovery claim.</text>','</g></svg>'];(out/'retrieval-curve.svg').write_text('\n'.join(svg))
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'metric_cells':len(tables),'statuses':statuses,'report':str(out/'REPORT.md'),'full_catalog':headline}))
if __name__=='__main__':main()
