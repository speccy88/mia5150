# Frozen lexical view: negative result, clearer coverage evidence

## Eight-query experiment

The original-query BM25 top100 ranking reproduced **exactly for all8 queries**. Seven queries had a changed request-clause view. We applied the already-frozen extraction rule and original/view equal-weight RRF60 once; no rule tuning or reruns followed evaluation.

| Method | Macro Recall@20 (primary) | NDCG@10 | Pool oracle NDCG@10 |
|---|---:|---:|---:|
| Original BM25 | 0.750000 | 0.579248 | 0.778287 |
| Original + request-clause view | 0.687500 | 0.548357 | 0.701643 |
| Cached SkillRet reference | 0.812500 | 0.764143 | Not needed for this contrast |

**The hypothesis was not supported.** Recall fell0.0625 and NDCG fell0.030892. Seven queries tied on Recall@20; q-02693 regressed from0.5 to0.0. Ranking changed on three cases: q-01417 improved, q-02237 worsened, and q-02693 lost its previously retrieved relevant candidate. Completeness@20 stayed0.625. All8 queries remain in the denominator.

This separates a candidate-coverage regression (lower recall/oracle) from ordering effects (actual NDCG). Keeping the original query as a branch did not guarantee that its useful candidates remained in the fused top20. Do not adopt this extraction rule.

The protocol was frozen before this cohort's predictions and evaluation. As with the other panels, the underlying public benchmark is historically exposed and is not a blind final test.

## Coverage denominators and overlap

The cached full benchmark has **4392 queries** and **7187 positive query–skill pairs**. Dense top20 misses **1264 pairs**, affecting **1120 queries**. One query can miss more than one relevant skill.

Of those1264 missing pairs:

| Recovery source | Missing pairs recovered |
|---|---:|
| Dense top100 | 644 |
| BM25 top10 | 206 |
| Both sources (intersection) | 158 |
| Dense100 only | 486 |
| BM25 top10 only | 48 |
| Either source (union) | **692** |
| Neither source | 572 |

Therefore644+206-158=692. These are **pairs, not query counts**. Dense100 recovers at least one missing pair for604 queries; BM25 top10 for201; either source for642 queries. These query counts also overlap and must not be added.

The earlier20-query panels had zero BM25 recoveries; the full cached data shows that those small panels did not represent the broader coverage problem.

## Diagnostic domain breakdown

Domains come from positively labeled skills, **not inferred query categories**. They cannot be used as an inference-time gate. A query may contribute to several domains; domain query counts are not additive. Micro recall divides recovered relevant pairs by all relevant pairs in that domain.

| Gold-skill domain | Queries contributing | Relevant pairs | Dense R@20 | Dense R@100 | Dense20 + BM25top10 R |
|---|---:|---:|---:|---:|---:|
| ai_agents | 1255 | 1358 | 0.6973 | 0.8314 | 0.7467 |
| developer_tools | 1199 | 1320 | 0.7356 | 0.8621 | 0.7659 |
| testing_qa | 332 | 338 | 0.7929 | 0.9142 | 0.8402 |
| security | 278 | 285 | 0.8456 | 0.9298 | 0.8702 |
| web_frontend | 619 | 645 | 0.8729 | 0.9535 | 0.8977 |
| backend_api | 711 | 745 | 0.8832 | 0.9490 | 0.8993 |

The union column has a variable candidate count20–30; it is a diagnostic coverage ceiling, not a fair fixed-budget quality improvement claim. Dense100 also uses a larger pool. Neither guarantees useful ordering.

## One distinct next hypothesis

Test **fixed-budget candidate complementarity**: compare dense top30 with dense top20 plus the10 highest-ranked *new* BM25 candidates, each with exactly30 candidates. Freeze a fresh small cohort and this allocation before scoring. Measure Recall@30, completeness and oracle NDCG; report all failures and coverage losses. This separates lexical complementarity from simply allowing more candidates and needs no new model/API calls because both top100 caches exist. A reranker test would be separate later work, not implied by oracle gains.

## Validation, resources and status

The CPU-only run used the existing fail-closed guard and one thread. Recorded exit status0; three telemetry samples observed minimum host availableRAM2.78GiB, guest4.84GiB, sampled workerRSS maximum61.44MiB. Swap counters stayed unchanged; resource settings were not relaxed. The first launch was blocked before worker execution because its monitor produced no telemetry. That failed setup is preserved; the unchanged experiment ran only after a fresh direct monitor was working. The owned monitor has stopped.

No paid calls, GPU computation or downloads. Commitments remain **$1.75/$4**. The47-test offline suite and reusable adapter are preserved. Browser/CUA tools are not exposed in this execution context; the prior unsuccessful Edge rendering was not repeated, so browser runtime/layout verification remains unconfirmed. The CLI cached demo is verified.

Evidence: `round2/clause-8-v1/baseline-reproduction.json`, `predictions.json`, `evaluation.json`; `round2/domain-coverage.json`; `runs/clause-eight-v2/worker.log`, `telemetry.jsonl`, `exit.json`. Protocol SHA256: `3e125ee205be9fd1ee267c3b9e6799ccef96a70005cfc36938831b98c6e97201`.
