# Exactly30 candidates: a small gain with real regressions

Construction was frozen before scoring. Control=dense top30. Treatment=dense top20 followed by the10 highest-ranked novel BM25 IDs, scanning the cached lexical top100. Both contain exactly30 unique IDs for every one of the4392 public queries. Stored order and ties were preserved; no label-driven construction or parameter search.

| Measure | Dense30 | Dense20 + lexical10 |
|---|---:|---:|
| Macro query Recall@30 | 0.873330 | 0.874962 |
| Positive-pair recall | 0.850703 | 0.858077 |
| Positive pairs covered /7187 | 6114 | 6167 |
| Completeness@30 | 0.779372 | 0.787341 |
| Pool-oracle NDCG@10 | 0.892345 | 0.892718 |

Macro recall gains **0.001632**, about0.16percentage points. Treatment gains183 pairs but loses130, net53. Query-level outcomes:175 wins,4094 ties,123 losses. The oracle gain is only0.000373. These are coverage ceilings using relevance labels, **not actual reranker quality**; no reranker ran.

## Domain tradeoffs

These domains come from gold skill metadata, only for diagnosis. Query membership overlaps across domains; do not sum query counts or use them as inference-time routing labels.

| Domain | Relevant pairs | Dense pair recall | Treatment pair recall | Gained/lost pairs |
|---|---:|---:|---:|---:|
| ai_agents | 1358 | 0.7349 | 0.7577 | 61/30 |
| testing_qa | 338 | 0.8225 | 0.8402 | 13/7 |
| devops_infra | 898 | 0.9120 | 0.9243 | 22/11 |
| web_frontend | 645 | 0.8915 | 0.9039 | 15/7 |
| developer_tools | 1320 | 0.7735 | 0.7727 | 39/40 |
| backend_api | 745 | 0.9060 | 0.9020 | 9/12 |
| business_ops | 553 | 0.9168 | 0.9096 | 5/9 |

Worst regressions lose the sole relevant skill entirely (query recall1→0), including q-00285 and q-00923 in data_ml, q-00925 in web_frontend, q-00938 in developer_tools, and q-01324 in ai_agents. The full per-query record retains IDs and gained/lost skills. Do not hide these cases behind the positive aggregate.

## Interpretation and next decision

The lexical branch is complementary in some cases, but a fixed20/10 allocation is not a clear universal replacement for dense30. Keep the tested dense20+Jev fixed-RRF60 path as the current small-panel lead; do not imply that this candidate-pool experiment proves its quality at30 candidates.

The largest uncertainty remains generalization beyond exposed public queries. Repeating more convenient panels will not produce a genuinely blind test; that needs independently authored/labeled queries.

One cheap, distinct operational test could check **candidate-order invariance** of typed Jev scoring: freeze two cases and reverse the20 candidate-question order, map answers back to stable skill IDs, compare to cached scores and final ranks. This probes sensitivity to request ordering without a sweep or another broad benchmark. It would not resolve blind generalization and is only proposed, not executed or reserved here. There is no reason to spend budget merely to fill the night.

## Evidence

`round2/candidate-budget30-v1/freeze.json` records protocol and source-cache hashes before labels loaded; `pools.json` stores both constructed pools; `evaluation.json` and `per-query.json` retain complete outcomes. Guarded run `runs/candidate-budget30-v1` exited0. No downloads, new models, API calls or GPU work. Shared commitments unchanged at **$1.75/$4**.

Protocol SHA256: `c601df43beee7295cd6518c285f93ca4a41f1041f26292d823758799c9ae87b8`. This is a full historically exposed public-data diagnostic, not blind confirmation. Browser runtime QA remains explicitly unverified.
