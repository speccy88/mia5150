# Jev/BM25 five-rubric screen — completed

The approved screen completed with **24/24 successful calls**. No new rubric beat the original on the prespecified primary metric, true macro Recall@5. The original reranker improved BM25 Recall@5 from **0.527778 to 0.583333**, reaching the top20 candidate-pool ceiling on these six development cases. This is a small public development result, not a confirmed full-benchmark gain.

## Direct reranking results

| Arm | Recall@5 | Recall@10 | NDCG@10 | Completeness@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.527778 | 0.527778 | 0.457957 | 0.333333 |
| original_direct | 0.583333 | 0.583333 | 0.507304 | 0.333333 |
| action_direct | 0.527778 | 0.583333 | 0.525100 | 0.333333 |
| anchored_direct | 0.527778 | 0.583333 | 0.507611 | 0.333333 |
| role_choice_direct | 0.583333 | 0.583333 | 0.502964 | 0.333333 |
| constraint_direct | 0.583333 | 0.583333 | 0.540679 | 0.333333 |
| original_depth10_direct | 0.527778 | 0.527778 | 0.501573 | 0.333333 |
| Pool_oracle | 0.583333 | 0.583333 | 0.641298 | 0.333333 |

The action-fit Noul and anchored Score each lost 0.055556 Recall@5 versus original (zero wins, five ties, one loss). Role Choice and constraint Score tied original on all six queries. The sole recall regression was q-01972. Constraint Score improved secondary NDCG@10 over original by 0.033376, but it did not pass the primary promotion rule. No metric was switched after observing results.

Original top10 reranking cannot recover positives outside its smaller pool: Recall@5/10 stayed 0.527778. Top20 increased reachable recall to 0.583333. True recall uses every positive qrel as denominator; completeness is separate. The pool oracle is a label-based diagnostic, not a prediction.

## Fixed RRF60 fusion, reported separately

| Rubric | Recall@5 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| original | 0.583333 | 0.583333 | 0.501882 |
| action | 0.527778 | 0.583333 | 0.586629 |
| anchored | 0.527778 | 0.583333 | 0.507611 |
| role_choice | 0.583333 | 0.583333 | 0.535257 |
| constraint | 0.583333 | 0.583333 | 0.540679 |

Action-fit fusion had the highest observed NDCG@10 (0.586629), while its Recall@5 remained below original. This secondary, development-only observation is not a promoted or confirmed winner. Rubric/type effects are confounded within the action and Choice arms; this is a practical factorized screen, not a pure wording-only causal comparison.

## Stopping rule and evidence

All five top20 arms used identical BM25 candidates, with a separate original top10 depth arm. The exact questions, candidate IDs, model, cohort, mapping, metric definitions and selection rule were frozen before inference. Four batches per query gave 24 calls. Promotion required >=0.05 mean Recall@5 gain over original direct Jev, at least two wins, more wins than losses, and all development calls valid. **No arm qualified**, so all six conditional confirmation calls were skipped. Confirmation labels remain unused. No repeats, prompt changes, additional cohorts or retrospective selection were performed.

Original already reached the candidate-pool recall ceiling, so no reranker could surpass it on this cohort. This is evidence that the screen is coverage-limited, not that the alternative rubrics are globally ineffective. The six public development cases were previously exposed. The earlier eight-query dense-retrieval confirmation, routing pilot and this BM25 screen remain distinct experiments. Hosted TypeSafe Jev here is not local JevK5.

The 65 offline tests pass. After execution, all 24 request hashes, rankings, fixed RRF60 outputs, token totals and development metrics were verified from saved evidence. No failed or omitted development cases were removed from the denominator.

## Cost, latency and resources

Input tokens: **168,117**; metered usage at the pinned USD0.042/million rate: **USD0.007060914**. The full USD0.10 reservation remains committed; no automatic refund. Shared commitments are **USD1.87 / USD4** (USD0.87 reservations plus USD1 external). Earlier known usage plus this run is USD0.013220151; the earlier uncertain OpenRouter call remains excluded from known usage. Fred reports USD5 signup credit, without balance verification. Metered usage is not a verified cash charge. No top-up occurred.

Median API batch latency: 0.270s; sum: 6.982s. Batch sizes differ (10,20,40 questions), so these are operational batch timings, not fair per-rubric speed comparisons. Cached BM25 retrieval and network setup are not an end-to-end production benchmark.

Resource guard completed normally: host minimum available 5.30GiB, guest 4.97GiB, GPU minimum free 8006MiB, maximum GPU temperature 43C, no active swap observed. No local GPU inference was launched for this API screen. WSL remains 6GiB RAM / 2 processors / 2GiB swap.

## Status and reproduction

The former approval blocker was resolved by direct user authorization in this task. Earlier blocker reports and ZIPs are preserved as historical snapshots and are superseded by this completed report. Other paid work remains paused. The one-shot runner now rejects duplicate execution. Local model experiments remain separate, pending sequential resource-guarded execution.

Offline verification: `.venv/bin/python round2/report_varied_screen.py` from `/home/user/code/mia5150-local`. Tests: `.venv/bin/python -m unittest discover -s round2 -p "test_*.py"`. These commands do not access keys or make inference calls. Do not rerun the paid worker.

Protocol SHA256: `d832c146580a51e466d8e887c6a9901979bbefce844c553f18205189822c407e`. Results SHA256: `01fad98bc7bf5907a761b16a0634da3d8bcef1b7253762e2d65eedd99310b1fd`.

API contract: https://docs.typesafe.ai/api . Pricing/model reference: https://docs.typesafe.ai/models .
