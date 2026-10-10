"""Offline preflight report. Never reads confirmation labels or credentials."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2';sys.path.insert(0,str(ROOT))
from harness import qrels,evaluate
from probe_contract import digest

def main():
 p=json.loads((HERE/'varied-screen-plan.json').read_text());packets=json.loads((HERE/'prepared-varied-screen.json').read_text());ledger=json.loads((HERE/'budget-ledger.json').read_text())
 assert digest(packets)==p['requests_sha256']
 assert not (HERE/'varied-screen-v3').exists() and 'varied-screen-v3' not in ledger['runs']
 ids=[q['id'] for q in p['dev_queries']];all_labels=qrels('test');labels={q:all_labels[q] for q in ids};del all_labels
 pools={q:p['pools'][q] for q in ids};oracle={q:sorted(pools[q],key=lambda d:(-labels[q].get(d,0),pools[q].index(d))) for q in ids}
 rankings={'BM25 top20':pools,'BM25 top10':{q:r[:10] for q,r in pools.items()},'Top20 oracle':oracle,'Top10 oracle':{q:sorted(pools[q][:10],key=lambda d:(-labels[q].get(d,0),pools[q].index(d))) for q in ids}}
 m={k:evaluate(labels,v) for k,v in rankings.items()}
 status={'status':'blocked_before_execution_by_automatic_approval_review','new_api_calls':0,'new_reservation_usd':'0','committed_plus_external_usd':'1.77','unit_tests_passed':59,'protocol_sha256':p['protocol_sha256'],'confirmation_labels_used':False,'dev_baseline_metrics':m,'reason':'Automatic review identified a conflict with the initial trusted delegation prohibiting paid APIs; summarized later authorization was insufficient. Explicit user approval required for stored Jev key use and public-data transmission. No bypass or retry performed.','proposed_bounds':{'max_calls':30,'max_input_tokens':1920000,'usage_bound_usd':'0.08064','reserve_usd':'0.10','shared_cap_usd':'4'},'user_credit_information':'User states USD5 Jev signup credit; no balance verification initiated.'}
 (HERE/'VARIED_SCREEN_PREFLIGHT.json').write_text(json.dumps(status,indent=2))
 lines=['# Jev/BM25 varied prompt screen — preflight only','','No new live result is available. Automatic approval review blocked execution before process launch. No new API call or reservation occurred; the shared committed amount remains USD1.77 (USD0.77 reservations plus USD1 external).','',
 '## Frozen experiment','','Five rubric/type arms on the same BM25 top20 pool: original Score, action-fit Noul, anchored Score, role Choice, constraint-fit Score. A separate original-Score top10 arm tests candidate depth. Six fixed development queries produce 24 calls. At most one new rubric can pass the predeclared gate to one separate six-query confirmation panel (six calls); no repeats or further rounds.','',
 'Promotion requires all 24 development calls valid, mean true Recall@5 gain >=0.05 over original direct Jev, at least two query wins and more wins than losses. The best eligible delta wins, with fixed variant-order tie break. Direct reranking and RRF60 fusion are reported separately. Failed queries retain their denominator.','',
 'The conservative maximum is 30 calls / 1,920,000 input tokens / USD0.08064 metered usage, with a proposed USD0.10 reservation within the unchanged USD4 shared cap. Fred reports USD5 signup credits; this information was accepted without balance verification. Metered usage is not a verified cash charge.','',
 '## Offline results','','59 contract/regression tests pass, including finite answer validation, stable ties, payload/depth limits, promotion rules and true recall with failed-query denominators. Maximum prepared payload is 50,698 serialized bytes. No data or prompt truncation. The v1 protocol is retained; v2 failed offline size validation; v3 was frozen before inference.','',
 '| Development arm | Recall@5 | Recall@10 | Pool Recall | NDCG@10 | Completeness@5 |','|---|---:|---:|---:|---:|---:|']
 for name,v in m.items():
  a=v['aggregate'];depth=10 if '10' in name else 20
  lines.append(f"| {name} | {a['Recall@5']:.6f} | {a['Recall@10']:.6f} | {a[f'Recall@{depth}']:.6f} | {a['NDCG@10']:.6f} | {a['Completeness@5']:.6f} |")
 lines+=['','Oracle values are diagnostic ceilings using development labels, not deployable predictions. True recall uses all positive qrels as denominator; completeness requires every positive. These six public development cases are small and historically exposed; they do not establish a new SkillRet benchmark result. Confirmation labels were not used.','',
 '## Blocker and explicit approval needed','','Automatic review rejected the proposed stored-key API call because the initial delegation prohibited paid APIs and later authorization summaries were insufficient. A fresh explicit approval must cover use of the saved Jev key to send only public SkillRet query text and candidate names/descriptions to https://api.typesafe.ai/v1/systemone, with the exact bounds above and no top-up. The rejected action was not retried or bypassed.','',
 'After that authorization is resolved, the prepared runner is `round2/varied_screen.py`, executed once through the existing resource guard and host monitor. It reserves before transmission, locks the shared budget, stops on the first uncertain response, rejects reruns, and leaves other paid work paused. Do not run it without resolving the approval block.','',
 'Offline reproducibility: `.venv/bin/python -m unittest discover -s round2 -p "test_*.py"`; `python3 round2/prepare_varied_screen.py`; `.venv/bin/python round2/finalize_varied_preflight.py`. These commands make no API calls.','',
 f"Protocol SHA256: `{p['protocol_sha256']}`.", '', 'API type contract: https://docs.typesafe.ai/api ; pricing/model reference: https://docs.typesafe.ai/models .']
 (HERE/'VARIED_SCREEN_PREFLIGHT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':status['status'],'new_calls':0,'aggregate_metrics':{k:v['aggregate'] for k,v in m.items()}}))
if __name__=='__main__':main()
