# Jev/BM25 varied prompt screen — preflight only

No new live result is available. Automatic approval review blocked execution before process launch. No new API call or reservation occurred; the shared committed amount remains USD1.77 (USD0.77 reservations plus USD1 external).

## Frozen experiment

Five rubric/type arms on the same BM25 top20 pool: original Score, action-fit Noul, anchored Score, role Choice, constraint-fit Score. A separate original-Score top10 arm tests candidate depth. Six fixed development queries produce 24 calls. At most one new rubric can pass the predeclared gate to one separate six-query confirmation panel (six calls); no repeats or further rounds.

Promotion requires all 24 development calls valid, mean true Recall@5 gain >=0.05 over original direct Jev, at least two query wins and more wins than losses. The best eligible delta wins, with fixed variant-order tie break. Direct reranking and RRF60 fusion are reported separately. Failed queries retain their denominator.

The conservative maximum is 30 calls / 1,920,000 input tokens / USD0.08064 metered usage, with a proposed USD0.10 reservation within the unchanged USD4 shared cap. Fred reports USD5 signup credits; this information was accepted without balance verification. Metered usage is not a verified cash charge.

## Offline results

59 contract/regression tests pass, including finite answer validation, stable ties, payload/depth limits, promotion rules and true recall with failed-query denominators. Maximum prepared payload is 50,698 serialized bytes. No data or prompt truncation. The v1 protocol is retained; v2 failed offline size validation; v3 was frozen before inference.

| Development arm | Recall@5 | Recall@10 | Pool Recall | NDCG@10 | Completeness@5 |
|---|---:|---:|---:|---:|---:|
| BM25 top20 | 0.527778 | 0.527778 | 0.583333 | 0.457957 | 0.333333 |
| BM25 top10 | 0.527778 | 0.527778 | 0.527778 | 0.457957 | 0.333333 |
| Top20 oracle | 0.583333 | 0.583333 | 0.583333 | 0.641298 | 0.333333 |
| Top10 oracle | 0.527778 | 0.527778 | 0.527778 | 0.563085 | 0.333333 |

Oracle values are diagnostic ceilings using development labels, not deployable predictions. True recall uses all positive qrels as denominator; completeness requires every positive. These six public development cases are small and historically exposed; they do not establish a new SkillRet benchmark result. Confirmation labels were not used.

## Blocker and explicit approval needed

Automatic review rejected the proposed stored-key API call because the initial delegation prohibited paid APIs and later authorization summaries were insufficient. A fresh explicit approval must cover use of the saved Jev key to send only public SkillRet query text and candidate names/descriptions to https://api.typesafe.ai/v1/systemone, with the exact bounds above and no top-up. The rejected action was not retried or bypassed.

After that authorization is resolved, the prepared runner is `round2/varied_screen.py`, executed once through the existing resource guard and host monitor. It reserves before transmission, locks the shared budget, stops on the first uncertain response, rejects reruns, and leaves other paid work paused. Do not run it without resolving the approval block.

Offline reproducibility: `.venv/bin/python -m unittest discover -s round2 -p "test_*.py"`; `python3 round2/prepare_varied_screen.py`; `.venv/bin/python round2/finalize_varied_preflight.py`. These commands make no API calls.

Protocol SHA256: `d832c146580a51e466d8e887c6a9901979bbefce844c553f18205189822c407e`.

API type contract: https://docs.typesafe.ai/api ; pricing/model reference: https://docs.typesafe.ai/models .
