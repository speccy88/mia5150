# SkillRet Decisions API scaling results

## Execution status

All 12,600 logical retrieval and reranking cases were accounted for across seven catalog sizes and three seeds. The ledger covers 12,598 distinct benchmark payloads and one separate synthetic compatibility request. 12,594 logical cases received valid scores; 6 used retrieval fallback, and none were unattempted. HTTP attempt failures (including retries): 4 × `predicate_refusal`. No failed probability was filled in.

The saved rankings, raw probabilities, binary judgments, usage, and figures passed independent validation. Full-precision values are in the [analysis CSV](results/plot_data.csv); all reported percentages are rounded only for display.

## Recall@5 across catalog sizes

The three Decisions columns are operational scores, including retrieval fallback where applicable. The baseline and reranked results each use equal query and seed weights within a catalog size.

| Catalog size | BM25 | Vector | Hybrid | BM25 + Decisions | Vector + Decisions | Hybrid + Decisions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 79.69% | 81.08% | 86.50% | 96.50% | 91.00% | 96.08% |
| 100 | 74.83% | 75.36% | 83.39% | 91.53% | 86.81% | 92.53% |
| 250 | 67.47% | 67.14% | 77.94% | 81.44% | 80.06% | 87.64% |
| 500 | 63.50% | 63.67% | 73.58% | 76.83% | 73.31% | 82.72% |
| 1,000 | 59.25% | 60.00% | 68.22% | 71.28% | 67.03% | 77.92% |
| 2,000 | 55.14% | 55.86% | 63.28% | 65.17% | 62.47% | 73.47% |
| 5,000 | 48.56% | 51.58% | 57.25% | 59.92% | 58.17% | 65.25% |

### Valid-only Decisions Recall@5

| Catalog size | BM25 | Vector | Hybrid |
| ---: | ---: | ---: | ---: |
| 50 | 96.50% | 91.00% | 96.08% |
| 100 | 91.51% | 86.93% | 92.53% |
| 250 | 81.44% | 80.06% | 87.64% |
| 500 | 76.83% | 73.31% | 82.72% |
| 1,000 | 71.28% | 67.14% | 77.92% |
| 2,000 | 65.17% | 62.47% | 73.55% |
| 5,000 | 59.92% | 58.17% | 65.25% |

## Candidate Recall@20 ceilings

| Catalog size | BM25 | Vector | Hybrid |
| ---: | ---: | ---: | ---: |
| 50 | 97.19% | 91.33% | 97.08% |
| 100 | 92.47% | 87.11% | 93.64% |
| 250 | 81.86% | 80.81% | 88.58% |
| 500 | 78.00% | 74.33% | 83.97% |
| 1,000 | 72.22% | 68.56% | 79.92% |
| 2,000 | 66.14% | 64.11% | 75.97% |
| 5,000 | 60.47% | 60.58% | 68.42% |

A reranker cannot recover a relevant skill that never enters its 20-candidate pool. None of the measured reranked scores exceeds the ceiling.

## Reranking gain and uncertainty

Values are the paired absolute change over the corresponding retrieval baseline, relative change in parentheses, then the 95% interval from 10,000 query-level bootstrap resamples (seed 42). The intervals are conditional on this development sample.

| Catalog size | BM25 | Vector | Hybrid |
| ---: | --- | --- | --- |
| 50 | +16.81 pp (+21.09% relative); [+12.81, +20.92] pp | +9.92 pp (+12.23% relative); [+7.19, +12.97] pp | +9.58 pp (+11.08% relative); [+6.64, +12.70] pp |
| 100 | +16.69 pp (+22.31% relative); [+12.69, +20.89] pp | +11.44 pp (+15.19% relative); [+8.44, +14.67] pp | +9.14 pp (+10.96% relative); [+6.22, +12.22] pp |
| 250 | +13.97 pp (+20.71% relative); [+10.50, +17.67] pp | +12.92 pp (+19.24% relative); [+9.14, +16.92] pp | +9.69 pp (+12.44% relative); [+6.53, +13.08] pp |
| 500 | +13.33 pp (+21.00% relative); [+9.64, +17.33] pp | +9.64 pp (+15.14% relative); [+6.61, +12.97] pp | +9.14 pp (+12.42% relative); [+6.22, +12.25] pp |
| 1,000 | +12.03 pp (+20.30% relative); [+8.44, +15.94] pp | +7.03 pp (+11.71% relative); [+4.28, +10.06] pp | +9.69 pp (+14.21% relative); [+6.53, +13.06] pp |
| 2,000 | +10.03 pp (+18.19% relative); [+6.97, +13.25] pp | +6.61 pp (+11.83% relative); [+3.81, +9.64] pp | +10.19 pp (+16.11% relative); [+7.00, +13.64] pp |
| 5,000 | +11.36 pp (+23.40% relative); [+7.86, +15.14] pp | +6.58 pp (+12.76% relative); [+3.31, +10.08] pp | +8.00 pp (+13.97% relative); [+4.61, +11.61] pp |

Across-seed sample standard deviation of operational Recall@5:

| Catalog size | BM25 | Vector | Hybrid |
| ---: | ---: | ---: | ---: |
| 50 | 0.66 pp | 0.65 pp | 0.68 pp |
| 100 | 0.43 pp | 0.47 pp | 0.38 pp |
| 250 | 0.77 pp | 0.97 pp | 0.55 pp |
| 500 | 0.36 pp | 1.87 pp | 0.97 pp |
| 1,000 | 0.91 pp | 0.64 pp | 0.60 pp |
| 2,000 | 1.45 pp | 1.09 pp | 0.76 pp |
| 5,000 | 0.43 pp | 0.65 pp | 0.72 pp |

## Latency, usage and cost

- Successful fresh benchmark API calls: median 225.7 ms, p95 512.9 ms (n=12,592; latency excludes cached local reuse). For recorded logical cases, end-to-end median 337.8 ms, p95 676.0 ms (n=12,598; includes retrieval and any fallback).
- Returned model values: `gpt-6-luna` (12,597).
- Known estimated API cost including the synthetic request: **USD 9.2316** at published rates. Conservative budget accounting: USD 10.1547 plus USD 0.5775 held for uncertain outcomes, against the USD 15 cap. No separate provider invoice reconciliation was performed.
- 12,597 HTTP responses, 12,597 settled attempt records, 2 unknown dispatches; 2 logical local-cache reuses. Smoke cases were reused in the full run.
- 85,716,615 input tokens including 0 cache-read and 26,395,576 cache-write tokens, and 0 output tokens. Observed smoke projection USD 9.9720; guarded projection USD 10.9115.
- Median / p95 latency by method at 5,000 skills: BM25: 225.3 ms / 517.1 ms API; 229.6 ms / 520.8 ms end-to-end; Vector: 229.0 ms / 506.8 ms API; 381.2 ms / 703.0 ms end-to-end; Hybrid: 225.1 ms / 475.1 ms API; 374.1 ms / 663.1 ms end-to-end. [Per-case timings](results/latency.csv) and [per-attempt usage](results/api_usage.csv) retain the underlying observations.

## Does reranking reduce the decline with scale?

- **BM25:** 31.14 pp without reranking vs. 36.58 pp with reranking (5.44 pp more decline).
- **Vector:** 29.50 pp without reranking vs. 32.83 pp with reranking (3.33 pp more decline).
- **Hybrid:** 29.25 pp without reranking vs. 30.83 pp with reranking (1.58 pp more decline).

The endpoint comparison is descriptive; per-size intervals quantify matched retrieval-versus-reranking changes, not uncertainty in this endpoint-to-endpoint trend. The 200 queries are a previously exposed development panel, and every sampled catalog contains all known positives. These results measure sensitivity to added distractors, not generalization to a new population.

## Historical comparison and validation

The archived Jev results were independently recomputed from the raw rankings and binary qrels for the same 200 queries. At 6,006 skills, historical BM25 Recall@5 was 60.58%, candidate Recall@20 was 72.75%, and Jev Recall@5 was 70.92%, a gain of 10.33 percentage points. The Jev arm used a different metadata renderer and BM25 preprocessing. Its point stays at 6,006 in the presentation graph. The graph also identifies the nine-model OpenRouter prompt benchmark with 20 single-label queries and the October matched retrieval study with 4,392 queries.

At the overlapping October sizes of 100 and 1,000, all 3,600 historical comparisons agreed on catalog membership and ordered top-five skill IDs. The archive exposes only its top five. For context only, the historical Jev point of 70.9167% at 6,006 skills compares descriptively with 59.92% / 58.17% / 65.25% for BM25, vector and hybrid plus Decisions at 5,000 skills. Do not treat this differently preprocessed result or the other two panels as matched Decisions conditions.

[Experimental protocol](PROTOCOL.md) · [Independent validation](results/validation.json) · [Recall@5 figure](results/figures/recall-at-5.png) · [Presentation figure](results/figures/presentation-comparison.png) · [Presentation data](results/presentation_plot_data.csv) · [Bootstrap intervals](results/bootstrap_intervals.csv)
