# Audited OpenRouter prompt-based benchmark

Source: `3fb7e66a1a219bc43d7c2057dca3d1c70a3c6c45`. Offline audit only; no new inference, API requests, or spending.

## Evidence and denominators

The planned grid has 1,260 cells: nine models × seven catalog sizes × 20 repeated questions. The CSV has 1,224 rows: 1,184 error-free responses and 40 preflight context exclusions (not API failures). Claude has 36 absent cells. There are no duplicated evaluation keys and no recorded attempted API failures in this retained CSV. Retry/deleted-attempt history is not recoverable from it.

The preserved metrics.json is also defective: all 63 model-size entries report zero valid responses, empty retrieval metrics and zero token/latency/cost efficiency, despite positive CSV measurements. Its exact_set_accuracy reproduces top-1 rather than true returned-set equality. The missing aggregate helper prevents a definitive root-cause replay; this audit replaces those aggregates in separate files, using explicit error-string handling and scored denominators.

The historical REPORT described seven models/944 rows and US$28.65; it omitted the two newly added models. Its original bytes are preserved in source/historical_REPORT.md. The source details.csv, metrics.json, configuration, query list and historical figures are unchanged.

**Original response-order metrics cannot be completely recovered.** The CSV stores parsed valid names separately from invalid names, not raw responses or their positions. Seventy-two completed rows contain reported invalid names; six completed rows have an empty parsed ranking. One stored ranking was truncated from 11 valid names to 10. The imported benchmark_models module, exact runtime version and exact prompt/library manifests are absent from the source commit. Even the zero-reported-invalid subset cannot rule out unrecorded parser normalization/deduplication.

Three labeled views are therefore provided, without inventing ranks:
- `no_reported_invalid`: identifiable stored-order subset (1,112 completed rows). This excludes all 72 rows with unrecoverable invalid positions, but includes completed empty outputs as retrieval failures. It is selection-biased and not a full-grid accuracy estimate.
- `retained_ranking`: all 1,184 completed rows scored exactly as stored; filtered-rank sensitivity, not raw-response accuracy.
- `invalid_insertion_lower`: all recorded invalid entries inserted before the stored list. A conditional lower bound for insertion of recorded invalids only, not a bound on unknown normalization, omitted duplicates, malformed prose or unsaved attempts. The retained view is the corresponding upper bound under that assumption.

Context exclusions and absent cases have NA quality, latency and actual cost. Availability remains visible, never a false measured zero. Empty completed outputs receive zero retrieval scores. Unverified actual cost is NA even when an estimated cost exists.

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

## Metric definitions and interpretation

For a nonempty binary gold set G and the first K output slots: Recall@K is unique gold hits / |G|; Precision@K is unique gold hits / K (short lists are not rewarded); Hit@K is any hit; MRR@K is reciprocal first relevant rank; NDCG@K uses log2(rank+1) discount and ideal DCG up to min(K, |G|); AP@K sums precision at relevant ranks / min(K, |G|), and MAP is its macro mean. Completeness@K means all golds found. F1@K is the harmonic mean of per-query precision and recall, then macro-averaged. All appear at K=1,3,5,10.

The generic scorer leaves invalid and duplicate slots in place; a repeated relevant ID gets credit once. Top-1 is membership of the first slot in G. Exact-set match requires the whole saved returned set to equal G with no invalid slot; repeated valid IDs do not change set identity, and duplicate rate is reported separately. Exact-set here describes the saved list, not unavailable pre-truncation output.

**Single-gold limitation:** every query has one labeled gold, so Recall = Hit = Completeness and MAP = MRR at each K. Precision = Recall/K, and F1 = 2×Recall/(K+1). These are mathematical identities, not eight independent findings. Plausible alternative skills are unjudged, not proven irrelevant. A valid ten-item list containing the gold can have top-1=1 but exact-set=0. Candidate/list completeness measures gold presence, not whether ten answers were returned.

Slot validity in summary.csv is only the recorded parsed-valid share of saved names plus reported invalid names. The original slot denominator, JSON/syntax validity and ID-level catalog membership are unavailable. Name-level outputs cannot disambiguate duplicate-name catalog records. Do not call this end-to-end raw-output validity.

## Paired uncertainty and fair comparison

Intervals are exploratory 95% percentile query-cluster bootstrap intervals, 5,000 draws, seed 5150. Sampling one question resamples all its observed library sizes together. Paired contrasts use intersecting (query ID, size) cells and resample question clusters, never the 1,184 rows as independent observations. CSV output contains the actual paired-cell and independent-query denominators. Only 20 distinct questions exist. Partial cases may reduce this further.

results/size_differences.csv contains within-model larger-minus-smaller library-size contrasts on the same queries. results/paired_differences.csv contains descriptive model differences for the identical successful cells of each pair; results/common_panel_summary.csv uses the common cells across all nine models within each view. No multiplicity correction is applied. Missing outcomes need not be missing at random. Neither a raw average nor overlapping point estimates establishes an advantage. All questions are exposed, golds are forcibly included in each sampled library, and source/library equivalence to other experiments is not established.

## Costs, latency and token provenance

Recorded/recomputed list-price estimate: **US$35.140929** for retained completed rows. Actual billed spend: **NA**. Prices are the frozen historical runner registry, not verified current rates. Tokens are recorded runner fields, potentially containing an estimated-input fallback; no raw usage payload distinguishes those cases. The recomputation reproduces every stored per-row estimate. These estimates exclude unknown deleted retries and are not an invoice or a total-spend assertion.

The old US$17 actual-cost total and 83.9% caching claim have no invoice, usage export, cache-token evidence or per-request billed-cost record in this package. They remain historical unsupported assertions and are not used for savings or value claims. Reasoning-token and cache savings cannot be derived from response latency.

Latency is the saved llm_latency/latency_s sum of request-attempt durations for completed rows, excluding explicit retry backoff sleeps. It is not TTFT. Separate e2e_latency summaries include the full per-query wall elapsed time, including backoff. The report uses mean, median and interpolated p95. Tokens and latency exclude preflight placeholders, whose input_tokens are estimates, not consumed tokens. Cross-model runs occurred at different times/providers; latency comparisons are descriptive, with no control for load, caching or reasoning settings.

## Reproduction and limits

Run `python experiments/benchmarks/openrouter-prompt-audit-20261108/audit.py` then `python -m unittest discover -s experiments/benchmarks/openrouter-prompt-audit-20261108/tests -v`. Only NumPy is required for the audit. Use the separate visualization reproduction instructions for charts. All input hashes are recorded in results/audit.json and rechecked before writing outputs. Original paid runners are not invoked.

Outputs: metrics.json and summary.csv (corrected aggregates), per_query.jsonl (full planned grid with status/NA), summary.csv (all models/sizes/cohorts/metrics), common_panel_summary.csv, paired_differences.csv, audit.json (counts/checks/hashes) and this report. Fixed seed and sorted rows make numeric artifacts reproducible. Existing raw results are never overwritten.

To obtain true original-position, strict-ID or independently billed metrics would require the missing original responses, exact parser and prompts/library manifests, request/attempt IDs, provider usage exports and invoices. This audit does not synthesize them or rerun paid calls. A new controlled experiment would also need held-out queries, multi-gold adjudication, explicit failure accounting, matched settings and a budget.
