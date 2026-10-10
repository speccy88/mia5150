"""Offline verification/report of the completed frozen screen; no credentials/network."""
import hashlib,json,statistics,sys
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2';OUT=HERE/'varied-screen-v3'
from probe_contract import digest
from varied_prompts import VARIANTS
from varied_screen import analyze,promote,PIN
sys.path.insert(0,str(ROOT))
from harness import qrels

def main():
 p=json.loads((HERE/'varied-screen-plan.json').read_text());packets=json.loads((HERE/'prepared-varied-screen.json').read_text());r=json.loads((OUT/'results.json').read_text());dev=json.loads((OUT/'development.json').read_text());ledger=json.loads((HERE/'budget-ledger.json').read_text())
 assert p['protocol_sha256']==r['protocol_sha256']==PIN and digest(packets)==p['requests_sha256']
 assert len(r['rows'])==24 and r['attempts']==24 and not r['selected']
 for row,item in zip(r['rows'],packets['dev']):
  assert row['request_sha256']==digest(item['request']) and row['pool']==item['pool'] and row['status']=='ok'
  assert row['id']==item['id'] and row['variants']==item['variants']
  for value in row['outputs'].values():
   pool=row['pool'];scores=value['scores'];order=[pool[i] for i in sorted(range(len(pool)),key=lambda i:(-scores[i],i))]
   assert order==value['ranking'] and set(order)==set(value['fusion'])==set(pool)
   fs={d:1/(61+pool.index(d))+1/(61+order.index(d)) for d in pool}
   assert sorted(pool,key=lambda d:(-fs[d],pool.index(d)))==value['fusion']
 labels=qrels('test');ids=[q['id'] for q in p['dev_queries']]
 assert analyze(r['rows'],ids,p['pools'],labels,VARIANTS,True)==dev
 assert promote(dev,r['rows'])==[] and not (OUT/'confirmation.json').exists()
 assert sum(x['input_tokens'] for x in r['rows'])==r['known_input_tokens']
 assert Decimal(r['known_usage_usd'])==Decimal(r['known_input_tokens'])*Decimal('.042')/1000000
 times=[x['api_seconds'] for x in r['rows']]
 telemetry=[json.loads(x) for x in (ROOT/'runs/varied-screen-v3/telemetry.jsonl').read_text().splitlines()]
 resources={'host_min_available_gib':min(x['host']['available_bytes'] for x in telemetry)/2**30,'guest_min_available_gib':min(x['guest_available'] for x in telemetry)/2**30,'gpu_min_free_mib':min(x['gpu_free_mib'] for x in telemetry),'gpu_max_temperature_c':max(x['gpu_temp'] for x in telemetry),'active_swap_observed':any(x['swap']!=telemetry[0]['swap'] for x in telemetry),'sampled_worker_peak_rss_mib':max(x.get('worker_rss_bytes',0) for x in telemetry)/2**20}
 batch={tag:{'calls':len(xs),'tokens':sum(x['input_tokens'] for x in xs),'median_api_seconds':statistics.median(x['api_seconds'] for x in xs)} for tag in ['original_action','anchored_role','constraint','depth10'] if (xs:=[x for x in r['rows'] if x['tag']==tag])}
 summary={'status':'completed','calls':24,'successful_calls':24,'confirmation_calls':0,'confirmation_labels_used':False,'known_input_tokens':r['known_input_tokens'],'known_usage_usd':r['known_usage_usd'],'stage_reservation_usd':'0.10','shared_committed_plus_external_usd':str(Decimal(ledger['committed_usd'])+Decimal(ledger['external_spend_usd'])),'shared_cap_usd':ledger['cap_usd'],'cash_charge_usd':None,'prior_known_usage_usd':'0.006159237','cumulative_known_usage_usd':str(Decimal('.006159237')+Decimal(r['known_usage_usd'])),'known_cost_excludes_earlier_uncertain_OpenRouter_call':True,'median_api_seconds':statistics.median(times),'total_api_seconds':sum(times),'batch_timings':batch,'resources':resources,'unit_tests_passed':65,'selected':[],'protocol_sha256':PIN,'results_sha256':hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest(),'result_verification':'All request hashes, skill-ID score ordering, fixed RRF60 output, token accounting and development metrics recomputed offline.'}
 (OUT/'verified-summary.json').write_text(json.dumps(summary,indent=2))
 lines=['# Jev/BM25 five-rubric screen — completed','','The approved screen completed with **24/24 successful calls**. No new rubric beat the original on the prespecified primary metric, true macro Recall@5. The original reranker improved BM25 Recall@5 from **0.527778 to 0.583333**, reaching the top20 candidate-pool ceiling on these six development cases. This is a small public development result, not a confirmed full-benchmark gain.','',
 '## Direct reranking results','','| Arm | Recall@5 | Recall@10 | NDCG@10 | Completeness@5 |','|---|---:|---:|---:|---:|']
 names=['BM25','original_direct','action_direct','anchored_direct','role_choice_direct','constraint_direct','original_depth10_direct','Pool_oracle']
 for name in names:
  a=dev['metrics'][name]['aggregate'];lines.append('| '+name+' | '+' | '.join(f'{a[k]:.6f}' for k in ['Recall@5','Recall@10','NDCG@10','Completeness@5'])+' |')
 lines+=['','The action-fit Noul and anchored Score each lost 0.055556 Recall@5 versus original (zero wins, five ties, one loss). Role Choice and constraint Score tied original on all six queries. The sole recall regression was q-01972. Constraint Score improved secondary NDCG@10 over original by 0.033376, but it did not pass the primary promotion rule. No metric was switched after observing results.','',
 'Original top10 reranking cannot recover positives outside its smaller pool: Recall@5/10 stayed 0.527778. Top20 increased reachable recall to 0.583333. True recall uses every positive qrel as denominator; completeness is separate. The pool oracle is a label-based diagnostic, not a prediction.','',
 '## Fixed RRF60 fusion, reported separately','','| Rubric | Recall@5 | Recall@10 | NDCG@10 |','|---|---:|---:|---:|']
 for v in VARIANTS:
  a=dev['metrics'][v+'_fusion']['aggregate'];lines.append('| '+v+' | '+' | '.join(f'{a[k]:.6f}' for k in ['Recall@5','Recall@10','NDCG@10'])+' |')
 lines+=['','Action-fit fusion had the highest observed NDCG@10 (0.586629), while its Recall@5 remained below original. This secondary, development-only observation is not a promoted or confirmed winner. Rubric/type effects are confounded within the action and Choice arms; this is a practical factorized screen, not a pure wording-only causal comparison.','',
 '## Stopping rule and evidence','','All five top20 arms used identical BM25 candidates, with a separate original top10 depth arm. The exact questions, candidate IDs, model, cohort, mapping, metric definitions and selection rule were frozen before inference. Four batches per query gave 24 calls. Promotion required >=0.05 mean Recall@5 gain over original direct Jev, at least two wins, more wins than losses, and all development calls valid. **No arm qualified**, so all six conditional confirmation calls were skipped. Confirmation labels remain unused. No repeats, prompt changes, additional cohorts or retrospective selection were performed.','',
 'Original already reached the candidate-pool recall ceiling, so no reranker could surpass it on this cohort. This is evidence that the screen is coverage-limited, not that the alternative rubrics are globally ineffective. The six public development cases were previously exposed. The earlier eight-query dense-retrieval confirmation, routing pilot and this BM25 screen remain distinct experiments. Hosted TypeSafe Jev here is not local JevK5.','',
 'The 65 offline tests pass. After execution, all 24 request hashes, rankings, fixed RRF60 outputs, token totals and development metrics were verified from saved evidence. No failed or omitted development cases were removed from the denominator.','',
 '## Cost, latency and resources','',
 f"Input tokens: **{r['known_input_tokens']:,}**; metered usage at the pinned USD0.042/million rate: **USD{r['known_usage_usd']}**. The full USD0.10 reservation remains committed; no automatic refund. Shared commitments are **USD{summary['shared_committed_plus_external_usd']} / USD4** (USD0.87 reservations plus USD1 external). Earlier known usage plus this run is USD{summary['cumulative_known_usage_usd']}; the earlier uncertain OpenRouter call remains excluded from known usage. Fred reports USD5 signup credit, without balance verification. Metered usage is not a verified cash charge. No top-up occurred.", '',
 f"Median API batch latency: {summary['median_api_seconds']:.3f}s; sum: {summary['total_api_seconds']:.3f}s. Batch sizes differ (10,20,40 questions), so these are operational batch timings, not fair per-rubric speed comparisons. Cached BM25 retrieval and network setup are not an end-to-end production benchmark.", '',
 f"Resource guard completed normally: host minimum available {resources['host_min_available_gib']:.2f}GiB, guest {resources['guest_min_available_gib']:.2f}GiB, GPU minimum free {resources['gpu_min_free_mib']}MiB, maximum GPU temperature {resources['gpu_max_temperature_c']}C, no active swap observed. No local GPU inference was launched for this API screen. WSL remains 6GiB RAM / 2 processors / 2GiB swap.", '',
 '## Status and reproduction','','The former approval blocker was resolved by direct user authorization in this task. Earlier blocker reports and ZIPs are preserved as historical snapshots and are superseded by this completed report. Other paid work remains paused. The one-shot runner now rejects duplicate execution. Local model experiments remain separate, pending sequential resource-guarded execution.','',
 'Offline verification: `.venv/bin/python round2/report_varied_screen.py` from `/home/user/code/mia5150-local`. Tests: `.venv/bin/python -m unittest discover -s round2 -p "test_*.py"`. These commands do not access keys or make inference calls. Do not rerun the paid worker.', '',
 f'Protocol SHA256: `{PIN}`. Results SHA256: `{summary["results_sha256"]}`.', '',
 'API contract: https://docs.typesafe.ai/api . Pricing/model reference: https://docs.typesafe.ai/models .']
 (HERE/'VARIED_SCREEN_RESULTS.md').write_text('\n'.join(lines)+'\n')
 (HERE/'VARIED_SCREEN_BUDGET_STATUS.json').write_text(json.dumps({k:summary[k] for k in ['stage_reservation_usd','shared_committed_plus_external_usd','shared_cap_usd','known_usage_usd','cumulative_known_usage_usd','cash_charge_usd']},indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
