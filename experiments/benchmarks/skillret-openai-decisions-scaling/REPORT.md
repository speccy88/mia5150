# SkillRet Decisions API scaling results

## Execution status

Retrieval completed for all 12,600 cases across the three methods, seven catalog sizes and three seeds. The 20-candidate pools produced 12,598 unique Decisions payloads. Two logical cases have payloads identical to another case.

The cloud proxy rejected the connection to the official OpenAI API before authentication. The synthetic compatibility check and the ten live smoke cases are pending. The run ledger records zero API requests, zero provider tokens and USD 0.00 in API charges. All 12,600 reranking cases remain unattempted. The reranking plots indicate the missing measurements and the corresponding CSV values are blank.

## Recall@5 across catalog sizes

The values below are percentages, with equal weighting across 200 queries and three seeds per size.

| Catalog size | BM25 | Vector | Hybrid | BM25 + Decisions | Vector + Decisions | Hybrid + Decisions |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 50 | 79.69% | 81.08% | 86.50% | Unmeasured | Unmeasured | Unmeasured |
| 100 | 74.83% | 75.36% | 83.39% | Unmeasured | Unmeasured | Unmeasured |
| 250 | 67.47% | 67.14% | 77.94% | Unmeasured | Unmeasured | Unmeasured |
| 500 | 63.50% | 63.67% | 73.58% | Unmeasured | Unmeasured | Unmeasured |
| 1,000 | 59.25% | 60.00% | 68.22% | Unmeasured | Unmeasured | Unmeasured |
| 2,000 | 55.14% | 55.86% | 63.28% | Unmeasured | Unmeasured | Unmeasured |
| 5,000 | 48.56% | 51.58% | 57.25% | Unmeasured | Unmeasured | Unmeasured |

From 50 to 5,000 skills, Recall@5 declined by 31.14 percentage points for BM25, 29.50 for vector retrieval, and 29.25 for hybrid retrieval. Hybrid had the highest Recall@5 at every size. At 5,000 skills, hybrid led vector retrieval by 5.67 percentage points and BM25 by 8.69 points. The three-seed sample standard deviations at that size were 0.10, 0.43 and 0.52 percentage points for BM25, vector and hybrid respectively.

## Candidate coverage and latency

At 5,000 skills, candidate Recall@20 was 60.47% for BM25, 60.58% for vector retrieval and 68.42% for hybrid retrieval. This leaves 11.92, 9.00 and 11.17 percentage points of headroom above their respective Recall@5 baselines. The candidate pool limits the gain any second-stage method can achieve.

Median local retrieval latency at 5,000 skills was 3.78 ms for BM25, 133.77 ms for vector retrieval and 137.72 ms for hybrid retrieval. Dense timings include a separately measured batch-one CPU query embedding, reused across that query's measured cells. Catalog construction, document embedding and API time are outside these local measurements.

## Historical comparison and validation

The archived Jev results were independently recomputed from the raw rankings and binary qrels for the same 200 queries. At 6,006 skills, historical BM25 Recall@5 was 60.58%, candidate Recall@20 was 72.75%, and Jev Recall@5 was 70.92%, a gain of 10.33 percentage points. The Jev arm used a different metadata renderer and BM25 preprocessing. Its point stays at 6,006 in the presentation graph. The graph also identifies the nine-model OpenRouter prompt benchmark with 20 single-label queries and the October matched retrieval study with 4,392 queries.

For the October archive's overlapping sizes of 100 and 1,000, all 3,600 method, query, size and seed cases matched catalog membership and ordered top-five skill IDs. The archive stores only five IDs per case. The independent validator checks the full new pools, rescored metrics, figures, request hashes, ledger and historical source hashes. The paired bootstrap and measured API latency and cost analyses require live responses. [October agreement](evidence/october_agreement.json) · [Independent validation](results/validation.json)

## Interpretation

On this exposed development panel with all known positives included in each catalog, additional distractors reduced Recall@5 for every first-stage method. Hybrid had the highest Recall@5 and candidate Recall@20 at 5,000 skills. Whether Decisions reranking reduces the decline with catalog growth remains unresolved.

[Experimental protocol](PROTOCOL.md) · [Recall@5 figure](results/figures/recall-at-5.pdf) · [Presentation figure](results/figures/presentation-comparison.pdf) · [Presentation data](results/presentation_plot_data.csv)
