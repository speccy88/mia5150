# OpenRouter Benchmark Report: audited update

This replaces the stale seven-model narrative with the nine-model audit of
commit `3fb7e66a1a219bc43d7c2057dca3d1c70a3c6c45`.
Read the [full audited report](../../experiments/benchmarks/openrouter-prompt-audit-20261108/results/REPORT.md),
[reproduction instructions and figures](../../experiments/benchmarks/openrouter-prompt-audit-20261108/README.md),
and [conditional retrieval comparison](../../experiments/benchmarks/openrouter-matched-retrieval-20261108/README.md).
The [historical report](../../experiments/benchmarks/openrouter-prompt-audit-20261108/source/historical_REPORT.md)
is preserved separately. All original raw result files remain unchanged.

## Independent count

- Planned: 1,260 model × size × question cells, nine models, seven sizes, 20 repeated questions.
- Retained CSV: 1,224 unique rows, no duplicate evaluation keys.
- Completed: 1,184 error-free response records, including six empty parsed lists.
- Not attempted due to estimated context: 40 placeholder rows, not measured quality zeros.
- Missing: 36 Claude cases (7 at 1,000; 14 at 2,000; 15 at 5,000).
- Invalid positions unavailable: 72 responses. No raw response bodies or original parser module were saved in this snapshot.
- Recorded list-price estimates: **US$35.140929**. Actual billed spend: **unverified**.

The saved per-query rank metrics reproduce arithmetically. The preserved
metrics.json aggregates do not: all 63 cells claim zero valid responses and
zero efficiency, with empty retrieval summaries; exact_set_accuracy is
actually top-1. Corrected aggregates are provided separately in the audit. Original response-order
metrics cannot be recovered by moving the separately stored invalid entries
back into guessed positions. The audit separates the 1,112 no-reported-invalid
responses from the 1,184-response parser-filtered sensitivity and conditional
invalid-insertion bounds. Empty completed outputs are retrieval failures;
unattempted and context-excluded cases remain NA.

## Nine-model summary

Metrics below use the retained-ranking sensitivity for comparability with saved output. The identifiable subset size and its top-1 value are shown separately. Aggregate sizes differ, so this is **not a leaderboard**.

| Model | Completed/planned | No-invalid n | Top-1, retained | Top-1, no-invalid | Recall@5 | NDCG@5 | Median / p95 s | Est. US$ | Actual US$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek V4 Flash | 120/140 | 105 | 0.792 | 0.829 | 0.900 | 0.849 | 1.196 / 3.678 | 0.791199 | NA |
| Kimi K2.5 | 120/140 | 115 | 0.717 | 0.722 | 0.833 | 0.780 | 31.244 / 144.774 | 2.391340 | NA |
| Qwen 3.8 Flash | 140/140 | 126 | 0.764 | 0.770 | 0.864 | 0.813 | 1.985 / 7.370 | 1.394393 | NA |
| GPT-6 Luna | 140/140 | 127 | 0.721 | 0.740 | 0.843 | 0.784 | 1.539 / 3.109 | 0.880663 | NA |
| GPT-6.1 Terra | 140/140 | 138 | 0.664 | 0.667 | 0.814 | 0.740 | 1.981 / 3.402 | 8.796055 | NA |
| Gemini 2.5 Flash | 140/140 | 131 | 0.743 | 0.748 | 0.843 | 0.794 | 0.964 / 2.971 | 2.840658 | NA |
| Claude Sonnet 5.5 | 104/140 | 99 | 0.817 | 0.818 | 0.913 | 0.867 | 1.387 / 4.898 | 11.554796 | NA |
| Kimi K3.1 Pro | 140/140 | 140 | 0.721 | 0.721 | 0.821 | 0.775 | 14.652 / 37.334 | 5.138800 | NA |
| DeepSeek V4.1 Flash | 140/140 | 131 | 0.757 | 0.756 | 0.850 | 0.806 | 0.819 / 5.918 | 1.353025 | NA |

## Scientific interpretation

One gold label per question makes Recall, Hit and Completeness identical at
each K; MAP equals MRR. Precision uses a fixed K denominator. Top-1 accuracy
and exact-set match are different. Full K=1/3/5/10 metrics and exploratory
95% paired query-cluster intervals are in the full report and CSVs.
The same 20 questions recur across sizes: treating all rows as independent
would overstate certainty. Compare matched cells, not unequal raw averages.

The old US$17 actual-cost and 83.9% caching assertions lack billing/cache
records in the repository and are not used for value claims. The runner
slices descriptions to 1,024 characters, rather than preserving arbitrary
full text. Exact prompt/library reproduction is blocked by missing original
inputs. No paid calls or new hosted-model inference were performed. The separate
retrieval study uses newly measured local embeddings and explicitly
conditional reconstructed inputs.
