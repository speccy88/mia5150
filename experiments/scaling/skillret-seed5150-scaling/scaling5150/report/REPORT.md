# MIA5150: seed-5150 local library scaling — audited preliminary

**Audited preliminary:** stored array integrity, reuse equality and bounded ranking replay are verified. EG2/Jina historical producer provenance remains incomplete: their per-run records do not bind every source vector file to the exact loaded model revision, tokenizer bytes and immutable worker source. No re-encoding was performed.

At 6,006 skills on the same 20 queries:
- bm25: Recall@5 0.6250; NDCG@10 0.6469.
- qwen_embed_06: Recall@5 0.6000; NDCG@10 0.6296.
- skillret_embed_06: Recall@5 0.7000; NDCG@10 0.7220.
- eg2_text270m_fp32: Recall@5 0.6500; NDCG@10 0.6198.
- jina_nano239m_fp32: Recall@5 0.5333; NDCG@10 0.5510.
- hybrid_rrf60: Recall@5 0.8000; NDCG@10 0.7553.
- query_clause_bm25: Recall@5 0.3833; NDCG@10 0.4372.

Paired descriptive intervals are in paired-bootstrap.json. This small exposed panel does not establish a general model ranking.

This is a new, free local experiment. It is separate from every prior TypeSafe, routing, LoRA and six-query screen. No paid calls, model downloads, uploads or Git push were used.

Twenty fixed public development queries share the same nested gold-conditioned libraries across methods. The first two form the separate paired generation and reranking diagnostic. Libraries contain all official positives plus seeded negatives; gold is not inserted into retrieved shortlists or disclosed in model prompts. All 4,392 public queries have prior exposure; these results are not an untouched final test. Known labels may not exhaust all semantically valid skills.

The requested four-size subset is 50, 250, 1,000 and 6,006: 20 queries × 4 sizes = 80 libraries. Two additional frozen sizes, 10 and 100, preserve the supplied PDF grid, giving 20 × 6 = 120 libraries total. The repository proposal at commit041f2ce specifies50/250/1000; the user requested seed5150 and the full6006 endpoint. The same20 query IDs and exact per-query library are used across all17 retrieval methods. The expensive selector and reranker panels use the same first2 query IDs, not20.

At execution time, direct source inspection used the supplied PDF and research-plan contract, plus the parent reviewer’s repository findings. After execution, project_proposal.tex and skillret-pilot/reference_metrics.py were fetched through the connected GitHub app and read at041f2ce. The comparison below is a local adaptation, not exact reproduction of the repository proposal. The post-run audit documents every material difference without altering any completed run.

## Retrieval: 20 queries per cell

| Method | Library | Recall@5 | Recall@10 | NDCG@10 | MRR@5 | Completeness@5 |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 10 | 0.9833 | 1.0000 | 0.9953 | 1.0000 | 0.9500 |
| qwen_embed_06 | 10 | 0.9167 | 1.0000 | 0.9250 | 0.9250 | 0.8500 |
| skillret_embed_06 | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| eg2_text270m_fp32 | 10 | 1.0000 | 1.0000 | 0.9604 | 0.9500 | 1.0000 |
| jina_nano239m_fp32 | 10 | 0.9500 | 1.0000 | 0.9650 | 0.9500 | 0.9500 |
| hybrid_rrf60 | 10 | 1.0000 | 1.0000 | 0.9984 | 1.0000 | 1.0000 |
| query_clause_bm25 | 10 | 0.9167 | 1.0000 | 0.9093 | 0.9000 | 0.8500 |
| bm25 | 50 | 0.9833 | 0.9833 | 0.9852 | 1.0000 | 0.9500 |
| qwen_embed_06 | 50 | 0.9000 | 0.9000 | 0.8656 | 0.9167 | 0.8500 |
| skillret_embed_06 | 50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| eg2_text270m_fp32 | 50 | 0.8833 | 0.9833 | 0.9158 | 0.9000 | 0.8500 |
| jina_nano239m_fp32 | 50 | 0.9167 | 0.9500 | 0.9199 | 0.9500 | 0.8500 |
| hybrid_rrf60 | 50 | 0.9833 | 0.9833 | 0.9883 | 1.0000 | 0.9500 |
| query_clause_bm25 | 50 | 0.7167 | 0.8667 | 0.7052 | 0.7500 | 0.5000 |
| bm25 | 100 | 0.9833 | 0.9833 | 0.9611 | 0.9750 | 0.9500 |
| qwen_embed_06 | 100 | 0.7750 | 0.9000 | 0.8459 | 0.9000 | 0.6500 |
| skillret_embed_06 | 100 | 0.9833 | 1.0000 | 0.9961 | 1.0000 | 0.9500 |
| eg2_text270m_fp32 | 100 | 0.8833 | 0.8833 | 0.8785 | 0.9000 | 0.8500 |
| jina_nano239m_fp32 | 100 | 0.8417 | 0.9167 | 0.8894 | 0.9500 | 0.7000 |
| hybrid_rrf60 | 100 | 0.9833 | 0.9833 | 0.9856 | 1.0000 | 0.9500 |
| query_clause_bm25 | 100 | 0.6167 | 0.7917 | 0.6462 | 0.7042 | 0.4000 |
| bm25 | 250 | 0.9167 | 0.9833 | 0.9072 | 0.9000 | 0.8500 |
| qwen_embed_06 | 250 | 0.7750 | 0.7750 | 0.7882 | 0.9000 | 0.6500 |
| skillret_embed_06 | 250 | 0.9833 | 0.9833 | 0.9698 | 0.9750 | 0.9500 |
| eg2_text270m_fp32 | 250 | 0.8667 | 0.8833 | 0.8290 | 0.8500 | 0.8000 |
| jina_nano239m_fp32 | 250 | 0.8000 | 0.8667 | 0.8217 | 0.9000 | 0.6500 |
| hybrid_rrf60 | 250 | 0.9667 | 0.9833 | 0.9651 | 0.9750 | 0.9000 |
| query_clause_bm25 | 250 | 0.5667 | 0.6667 | 0.5709 | 0.6667 | 0.3500 |
| bm25 | 1000 | 0.8250 | 0.9167 | 0.8048 | 0.8250 | 0.7500 |
| qwen_embed_06 | 1000 | 0.7250 | 0.7750 | 0.7423 | 0.8600 | 0.5500 |
| skillret_embed_06 | 1000 | 0.9333 | 0.9833 | 0.8995 | 0.8917 | 0.9000 |
| eg2_text270m_fp32 | 1000 | 0.7500 | 0.8000 | 0.7300 | 0.7750 | 0.6500 |
| jina_nano239m_fp32 | 1000 | 0.7333 | 0.7500 | 0.7103 | 0.8167 | 0.6000 |
| hybrid_rrf60 | 1000 | 0.9667 | 0.9667 | 0.8979 | 0.9125 | 0.9000 |
| query_clause_bm25 | 1000 | 0.5750 | 0.6417 | 0.5345 | 0.6417 | 0.3500 |
| bm25 | 6006 | 0.6250 | 0.8000 | 0.6469 | 0.6875 | 0.5000 |
| qwen_embed_06 | 6006 | 0.6000 | 0.6500 | 0.6296 | 0.7500 | 0.4000 |
| skillret_embed_06 | 6006 | 0.7000 | 0.8000 | 0.7220 | 0.7500 | 0.6000 |
| eg2_text270m_fp32 | 6006 | 0.6500 | 0.7000 | 0.6198 | 0.6917 | 0.5000 |
| jina_nano239m_fp32 | 6006 | 0.5333 | 0.6333 | 0.5510 | 0.6333 | 0.3500 |
| hybrid_rrf60 | 6006 | 0.8000 | 0.8250 | 0.7553 | 0.8167 | 0.7000 |
| query_clause_bm25 | 6006 | 0.3833 | 0.5000 | 0.4372 | 0.5500 | 0.1500 |

All additional specialist, field, centroid, MMR/DSR and matched candidate-budget arms are in metrics.csv and raw retrieval results. Candidate recall/oracles and fixed-top5 set diagnostics are stored separately; a ranked top5 is not a model-generated final set. Candidate-budget comparisons must use dense20 versus roundrobin20 and dense30 versus dense20+10 novel BM25.

## Final selection: two-query paired diagnostic

| Model | Method | Library | Valid / planned | Set precision | Set recall | Set F1 | Exact set |
|---|---|---:|---:|---:|---:|---:|---:|
| small_qwen06 | bm25 | 10 | 2/2 | 0.6250 | 0.8333 | 0.6000 | 0.0000 |
| small_qwen06 | bm25 | 50 | 2/2 | 0.3750 | 0.6667 | 0.4000 | 0.0000 |
| small_qwen06 | bm25 | 100 | 2/2 | 0.4167 | 0.6667 | 0.4500 | 0.0000 |
| small_qwen06 | bm25 | 250 | 2/2 | 0.2917 | 0.6667 | 0.3667 | 0.0000 |
| small_qwen06 | bm25 | 1000 | 2/2 | 0.1000 | 0.5000 | 0.1667 | 0.0000 |
| small_qwen06 | bm25 | 6006 | 2/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| small_qwen06 | vector | 10 | 2/2 | 0.6250 | 0.8333 | 0.6000 | 0.0000 |
| small_qwen06 | vector | 50 | 2/2 | 0.7500 | 0.8333 | 0.7333 | 0.0000 |
| small_qwen06 | vector | 100 | 2/2 | 0.5833 | 0.8333 | 0.6667 | 0.0000 |
| small_qwen06 | vector | 250 | 2/2 | 0.6667 | 0.8333 | 0.6500 | 0.0000 |
| small_qwen06 | vector | 1000 | 2/2 | 0.4583 | 0.8333 | 0.5333 | 0.0000 |
| small_qwen06 | vector | 6006 | 1/2 | 0.1667 | 0.5000 | 0.2500 | 0.0000 |
| small_qwen06 | domain_vector_lm | 10 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| small_qwen06 | domain_vector_lm | 50 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| small_qwen06 | domain_vector_lm | 100 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| small_qwen06 | domain_vector_lm | 250 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| small_qwen06 | domain_vector_lm | 1000 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| small_qwen06 | domain_vector_lm | 6006 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| small_qwen06 | hybrid | 10 | 2/2 | 0.6250 | 1.0000 | 0.7619 | 0.0000 |
| small_qwen06 | hybrid | 50 | 2/2 | 0.3750 | 0.8333 | 0.4857 | 0.0000 |
| small_qwen06 | hybrid | 100 | 2/2 | 0.3500 | 0.6667 | 0.3667 | 0.0000 |
| small_qwen06 | hybrid | 250 | 2/2 | 0.4583 | 0.8333 | 0.5333 | 0.0000 |
| small_qwen06 | hybrid | 1000 | 2/2 | 0.3500 | 0.6667 | 0.3667 | 0.0000 |
| small_qwen06 | hybrid | 6006 | 2/2 | 0.2500 | 0.6667 | 0.3429 | 0.0000 |
| small_qwen06 | full_listing | 10 | 2/2 | 0.4000 | 1.0000 | 0.5417 | 0.0000 |
| small_qwen06 | full_listing | 50 | 2/2 | 0.1000 | 0.5000 | 0.1667 | 0.0000 |
| small_qwen06 | full_listing | 100 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| small_qwen06 | full_listing | 250 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| small_qwen06 | full_listing | 1000 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| small_qwen06 | full_listing | 6006 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| medium_qwen35 | bm25 | 10 | 2/2 | 1.0000 | 0.8333 | 0.9000 | 0.5000 |
| medium_qwen35 | bm25 | 50 | 2/2 | 1.0000 | 0.8333 | 0.9000 | 0.5000 |
| medium_qwen35 | bm25 | 100 | 1/2 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| medium_qwen35 | bm25 | 250 | 2/2 | 0.5000 | 0.8333 | 0.5833 | 0.0000 |
| medium_qwen35 | bm25 | 1000 | 2/2 | 0.7500 | 0.6667 | 0.7000 | 0.5000 |
| medium_qwen35 | bm25 | 6006 | 2/2 | 0.1667 | 0.1667 | 0.1667 | 0.0000 |
| medium_qwen35 | vector | 10 | 2/2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| medium_qwen35 | vector | 50 | 2/2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| medium_qwen35 | vector | 100 | 2/2 | 0.7500 | 1.0000 | 0.8333 | 0.5000 |
| medium_qwen35 | vector | 250 | 2/2 | 0.7500 | 1.0000 | 0.8333 | 0.5000 |
| medium_qwen35 | vector | 1000 | 2/2 | 0.5417 | 1.0000 | 0.6786 | 0.0000 |
| medium_qwen35 | vector | 6006 | 2/2 | 0.3500 | 0.6667 | 0.3667 | 0.0000 |
| medium_qwen35 | domain_vector_lm | 10 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| medium_qwen35 | domain_vector_lm | 50 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| medium_qwen35 | domain_vector_lm | 100 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| medium_qwen35 | domain_vector_lm | 250 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| medium_qwen35 | domain_vector_lm | 1000 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| medium_qwen35 | domain_vector_lm | 6006 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| medium_qwen35 | hybrid | 10 | 2/2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| medium_qwen35 | hybrid | 50 | 1/2 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| medium_qwen35 | hybrid | 100 | 1/2 | 0.2500 | 0.5000 | 0.3333 | 0.0000 |
| medium_qwen35 | hybrid | 250 | 2/2 | 0.5833 | 0.8333 | 0.6667 | 0.0000 |
| medium_qwen35 | hybrid | 1000 | 2/2 | 0.5000 | 0.8333 | 0.5833 | 0.0000 |
| medium_qwen35 | hybrid | 6006 | 2/2 | 0.4167 | 0.6667 | 0.4500 | 0.0000 |
| medium_qwen35 | full_listing | 10 | 2/2 | 0.8750 | 1.0000 | 0.9286 | 0.5000 |
| medium_qwen35 | full_listing | 50 | 2/2 | 0.6000 | 1.0000 | 0.6667 | 0.5000 |
| medium_qwen35 | full_listing | 100 | 2/2 | 0.4333 | 0.8333 | 0.5000 | 0.0000 |
| medium_qwen35 | full_listing | 250 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| medium_qwen35 | full_listing | 1000 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| medium_qwen35 | full_listing | 6006 | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| small_qwen06 | domain_vector_lm_constrained_route | 10 | 2/2 | 0.4167 | 0.6667 | 0.4500 | 0.0000 |
| small_qwen06 | domain_vector_lm_constrained_route | 50 | 2/2 | 0.2917 | 0.6667 | 0.3667 | 0.0000 |
| small_qwen06 | domain_vector_lm_constrained_route | 100 | 2/2 | 0.2500 | 0.6667 | 0.3429 | 0.0000 |
| small_qwen06 | domain_vector_lm_constrained_route | 250 | 2/2 | 0.2917 | 0.6667 | 0.3929 | 0.0000 |
| small_qwen06 | domain_vector_lm_constrained_route | 1000 | 1/2 | 0.1250 | 0.5000 | 0.2000 | 0.0000 |
| small_qwen06 | domain_vector_lm_constrained_route | 6006 | 2/2 | 0.2667 | 0.6667 | 0.3750 | 0.0000 |
| medium_qwen35 | domain_vector_lm_constrained_route | 10 | 2/2 | 1.0000 | 0.6667 | 0.7500 | 0.5000 |
| medium_qwen35 | domain_vector_lm_constrained_route | 50 | 2/2 | 0.7500 | 0.6667 | 0.7000 | 0.5000 |
| medium_qwen35 | domain_vector_lm_constrained_route | 100 | 2/2 | 0.4167 | 0.6667 | 0.4500 | 0.0000 |
| medium_qwen35 | domain_vector_lm_constrained_route | 250 | 2/2 | 0.5000 | 0.6667 | 0.5333 | 0.0000 |
| medium_qwen35 | domain_vector_lm_constrained_route | 1000 | 2/2 | 0.4167 | 0.6667 | 0.4500 | 0.0000 |
| medium_qwen35 | domain_vector_lm_constrained_route | 6006 | 2/2 | 0.1000 | 0.5000 | 0.1667 | 0.0000 |

The common selector sees BM25, vector, LM-domain-vector or hybrid top5, or every actual library item for full listing. LM routing predicts one of six categories from query text alone; no fallback is used. Centroid routing is a separate retrieval ablation. Aliases and display order are independent of labels. Invalid output, empty domain and full-listing context overflow retain zero scores and the planned denominator. Context is 8,192 tokens with a 256-token output allowance; overflowing libraries are not secretly shortened.

Original unconstrained domain-route responses all violated the required JSON-string schema (objects/fences instead); all original zeros remain. domain_vector_lm_constrained_route is a separately frozen infrastructure ablation that adds only a six-category output grammar to routing. It uses no gold category and leaves the selection prompt/parser unchanged. Exact old selection responses, including failures, are reused rather than retried. Compare this separate arm explicitly, not as a silent correction to original predictions.

## Common-pool reranking: two queries, top20, three sizes

| Model | Library | Valid / planned | Recall@5 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| small_skillret | 50 | 2/2 | 0.6667 | 0.8333 | 0.8182 |
| small_skillret | 1000 | 2/2 | 1.0000 | 1.0000 | 0.9735 |
| small_skillret | 6006 | 2/2 | 0.8333 | 0.8333 | 0.8254 |
| medium_jevk5 | 50 | 2/2 | 1.0000 | 1.0000 | 0.9837 |
| medium_jevk5 | 1000 | 2/2 | 0.5000 | 1.0000 | 0.6401 |
| medium_jevk5 | 6006 | 2/2 | 0.1667 | 0.8333 | 0.4691 |
| fit_qwen8 | 50 | 0/2 | 0.0000 | 0.0000 | 0.0000 |
| fit_qwen8 | 1000 | 0/2 | 0.0000 | 0.0000 | 0.0000 |
| fit_qwen8 | 6006 | 0/2 | 0.0000 | 0.0000 | 0.0000 |

Native model prompts/readouts differ and are pinned; this is a deployment comparison, not a parameter-count-only experiment. Missing native option logits invalidate the whole case. No gold items are added to these pools.

The 8B Q4 file passed local GGUF extent validation and loaded under the unchanged guard at context2,048. Warmup and one candidate returned; the next candidate raised ValueError in native scoring, so the whole first case failed and five remained unattempted. The failed response was not persisted before the exception, so its exact cause cannot be conclusively recovered from the saved raw data. All six planned cases remain zero for strict accounting; these are operational failures, not evidence of poor intrinsic ranking quality. No repeated failed request or widened logprob tuning was performed.

## Cost, fit and timing

All new work costs USD0. Small retrieval tiers are 239M/270M FP32 and 0.6B FP16. The small selector is Qwen3-0.6B Q8; the medium selector is Qwen3.5-4B Q4. JevK5-4B Q5 and cached Qwen3-Reranker-8B Q4 are separate reranker candidates. An 8B file-size estimate is not a fit result. The guard traces include runtime memory, context/KV startup, host/guest headroom, temperature and swap; failed fits are preserved. No global-best claim follows from this small panel.

WSL remains 6GiB RAM, two processors and 2GiB swap. GPU workloads run sequentially. The guard requires host/guest free RAM >=2GiB, free VRAM >=1GiB, temperature <=80C, fresh host telemetry and no active swap.

Index creation, retrieval CPU/search, query forwards, startup, reranking and selection are recorded separately. Identical old embedding vectors and exact repeated generation requests can be reused. A cache hit is not zero-cost serving inference: cached query times and original physical completion times retain their provenance. Null historical timings remain unknown. Do not divide replay wall time by full-listing latency or call it measured end-to-end speedup. Logical route tokens are charged to every domain-vector case, while physical reuse is recorded separately. Full fresh end-to-end latency is not measured for every cell.

Resource details: resource-summary.json. Guard worker CPU/RSS excludes child llama-server. Additional evidence/*-server-resources.json captures child CPU and peak RSS for the later medium full-selector, rerankers and constrained-domain runs; early selector pilot/small-full child accounting is unavailable. System-wide RAM/GPU samples are available for every guard run. Per-request logical tokens, selector HTTP responses, cache flags and timings are saved. JevK5 retains processed scores/probabilities, not original completion responses or pre-normalization raw logits/top-logprobs. Missing responses were not reconstructed. New embedding call logs include input token counts/truncation, CPU and wall time.

## Scope and limits

Implemented and tested: deterministic common libraries; gold isolation; exact vector/BM25 retrieval; field/clause/fusion/structural extensions; independent metric checks; bounded local selector and common-pool reranker pipelines; raw/CSV/report export. Execution status below is authoritative—implemented is not equivalent to successfully run.

Not run in this frozen experiment: hosted APIs/RunPod, new downloads, BGE-small and CLM without verified local artifacts, fresh LoRA/training, graph/dependency methods lacking annotations, hard-overlap distractor expansion, larger generative panels, and Plumb retry after its earlier resource failure. Historical experiments are not counted as new scaling results.

Status details:

```json
[
  {
    "role": "small_qwen06",
    "artifact": "selection-small_qwen06-full",
    "status_counts": {
      "ok": 39,
      "empty_or_invalid_domain": 12,
      "invalid_selection": 3,
      "context_overflow": 6
    }
  },
  {
    "role": "medium_qwen35",
    "artifact": "selection-medium_qwen35-full",
    "status_counts": {
      "ok": 39,
      "empty_or_invalid_domain": 12,
      "context_overflow": 6,
      "invalid_selection": 3
    }
  },
  {
    "role": "small_qwen06",
    "artifact": "domain-constrained-small_qwen06",
    "status_counts": {
      "ok": 11,
      "invalid_selection": 1
    }
  },
  {
    "role": "medium_qwen35",
    "artifact": "domain-constrained-medium_qwen35",
    "status_counts": {
      "ok": 12
    }
  },
  {
    "role": "small_skillret",
    "status_counts": {
      "ok": 6
    }
  },
  {
    "role": "medium_jevk5",
    "status_counts": {
      "ok": 6
    }
  },
  {
    "role": "fit_qwen8",
    "status_counts": {
      "failed": 1,
      "not_attempted": 5
    }
  }
]
```

## Reproduction

Use the existing isolated .venv in /home/user/code/mia5150-local. Preserve existing output directories; scripts intentionally refuse overwriting experimental runs. Set MIA_HOST_TELEMETRY to the current Windows monitor JSONL and run every compute stage through guard.py. See commands.txt. Local cached model paths/revisions/hashes are pinned; weights and credentials are not in the portable evidence package.

## Observed practical fit

| Guarded run | Min host free GiB | Min guest free GiB | Min GPU free MiB | Max GPU C |
|---|---:|---:|---:|---:|
| scaling5150-domain-constrained-small-v1 | 5.34 | 4.61 | 6380 | 57 |
| scaling5150-rerank-fit_qwen8-v1 | 2.86 | 4.41 | 3120 | 59 |
| scaling5150-rerank-medium_jevk5-v1 | 2.47 | 4.17 | 4824 | 65 |
| scaling5150-rerank-small_skillret-v1 | 2.68 | 4.12 | 6684 | 57 |
| scaling5150-selection-medium_qwen35-full-v1 | 4.37 | 4.17 | 4948 | 65 |

For the tested final-selector deployment, Qwen3.5-4B Q4 is usable at context8,192; the two-query hybrid final-set F1 is0.4500 versus0.3429 for Qwen3-0.6B Q8. Neither achieved an exact set at6,006. This is a narrow diagnostic, not proof that4B is globally optimal. The8B reranker fits context2,048 but its current scoring adapter is not usable, so it cannot yet displace a working smaller reranker.

On the two shared full-library reranking cases, the SkillRet0.6B reranker improves Recall@5 from the dense pool order0.6667 to0.8333 and NDCG@10 from0.7346 to0.8254. JevK5-4B lowers Recall@5 to0.1667 and NDCG@10 to0.4691 on the same pool. Larger is not consistently better here. Small-scope reranker results must not be pooled with the20-query retrieval table.

## Post-run source and reproducibility audit

Sources read after execution: [project_proposal.tex](https://github.com/speccy88/mia5150/blob/041f2ce/project_proposal.tex) and [reference_metrics.py](https://github.com/speccy88/mia5150/blob/041f2ce/skillret-pilot/reference_metrics.py). The reviewed pure-standard-library helper was used only for an offline audit, never model inference. All120 memberships exactly match its sampler at seed5150. All18,360 checked retrieval scalar values match its definitions for Recall/Precision/NDCG/Completeness at5/10 and MRR at5.

Display order differs in all120 cases: the helper hashes seed NUL query_id NUL display NUL skill_id; the frozen local protocol hashes seed NUL display NUL query_id NUL skill_id. Both are label-independent and shared across methods, but are not byte-identical to the source helper. Completed predictions and frozen libraries were not rewritten.

Other material deviations from the original proposal:

- The exact BGE-small-en-v1.5 vector arm is untested here. The exact BGE-backed native domain-vector arm is also untested. Tested local domain variants use the same local selector to predict six major categories and restrict SkillRet-0.6B vectors; no global fallback. The centroid arm is a separate ablation and is not equivalent to an LM router.
- Common local metadata is original name + separator + description, capped once at1,024 characters. The proposal’s additional128-BGE-token cap is not applied. Neural encoder caps are listed below. Thus these results cannot be compared numerically to the historical BGE metadata baseline as a method-only change.
- Direct localhost llama.cpp generation and opaque aliases replace native LangChain Deep Agents SkillsMiddleware/StateBackend and select_skills calls returning original IDs. That middleware integration and the separate native activation-read check are not reproduced in this experiment. No skill bodies are executed.
- Local Qwen selectors replace the proposal’s hosted DeepSeek/provider configuration. Paid services are outside this experiment. Seed5150 replaces42. The20 queries are selected deterministically from an existing exposed128-query cache panel; the proposal’s previously unused20-query/4192-ID-final-test claim does not apply because all4392 public queries have prior exposure.
- Only2 queries receive generation/reranking. This is an explicit resource-bounded diagnostic, not a20-query selection comparison.

### Complete list of17 retrieval identities

| Identity | Definition |
|---|---|
| bm25 | Common capped metadata; library-local IDF; top100 retained |
| qwen_embed_06 | Qwen3-Embedding-0.6B native query/document embeddings |
| skillret_embed_06 | SKILLRET-Embedding-0.6B native embeddings |
| skillrouter_embed_06 | SkillRouter-Embedding-0.6B native embeddings |
| skillreason_embed_06 | SkillReason-embedding-0.6b native embeddings |
| eg2_text270m_fp32 | EmbeddingGemma2 text-only271M parameters, native SearchQuery/Document prompts |
| jina_nano239m_fp32 | Jina v5 text nano239M, native retrieval adapter and query/document prompts |
| hybrid_rrf60 | Equal RRF60 of BM25 and SkillRet top100 |
| domain_centroid | Mean SkillRet similarity per major category; choose one, then restrict dense order; no LM |
| field_tfidf | Equal name/description TF-IDF cosine, library-local IDF |
| mmr5 | Fixed lambda0.8, SkillRet top20, five-item budget and frozen relevance gate |
| dsr5 | Fixed0.9 relevance/diversity logdet blend on SkillRet top20; five-item budget |
| dense20 | SkillRet top20 candidate control |
| dense30 | SkillRet top30 candidate control |
| roundrobin20 | Deduplicated alternating BM25/SkillRet candidates, budget20 |
| dense20_novel10 | Preserve SkillRet top20, append up to10 novel BM25 candidates; budget30 |
| query_clause_bm25 | Whole query plus first3 qualifying deterministic clauses, equal RRF60; no LM rewriting |

Exact ties use original skill-ID ordering for retrieval/fusion; pointwise reranker ties use frozen pool order. Candidate-budget controls are paired at equal budgets. All known gold skills are included only by library construction, not by ranking, prompting or final selection.

### Query panel and configuration

Query IDs: q-03666, q-00674, q-03533, q-04226, q-01436, q-03558, q-04106, q-00172, q-02209, q-03656, q-01962, q-01329, q-03650, q-01330, q-04599, q-01672, q-04856, q-01955, q-01186, q-04985.

Gold multiplicities:10 single-positive,8 double-positive,2 triple-positive queries. Both expensive panels use q-03666 (one positive) and q-00674 (three positives). Generative sizes are all6; reranker sizes are50/1000/6006 with the same SkillRet top20 pools across models.

| Role | Deployment | Input/context control |
|---|---|---|
| Four0.6B encoders | FP16, normalized1024-dimensional vectors | Native prompts; max1024 tokens; batch1 for refreshed headers |
| EG2 text encoder | FP32, eager attention,271,002,624 loaded text parameters;768-dimensional vectors | Native SearchQuery/Document; max512 tokens; batch1 |
| Jina nano encoder | FP32, eager attention,238,897,920 parameters;768-dimensional vectors | Native retrieval adapter; max1024 tokens; batch1 |
| Small selector/router | Qwen3-0.6B Q8_0 |8192 context;256 output tokens; temperature0; seed5150; thinking disabled |
| Medium selector/router | Qwen3.5-4B Q4_K_M |Same8192/256/temperature0/seed5150 controls |
| Small reranker | SKILLRET-Reranker-0.6B FP16 local conversion |2048 context; native yes-minus-no logits; batch1; no truncation |
| Medium reranker | JevK5-4B-v0.3 Q5_K_M |2048 context; native four-option Score; temperature1.22; top256 logprobs; missing options invalidate |
|8B fit/rerank challenger | Qwen3-Reranker-8B Q4_K_M |2048 context; one generated token; temperature0; native yes/no top256 readout; stopped after failure |

llama.cpp is cached build b11425/CUDA12.8, offline, localhost-only, all GPU layers, t=1/tb=1, batch128/microbatch128, dio loading, cache RAM0 and context checkpoints0. Embedding compute threads are1; auxiliary runtime threads may be present. One GPU workload runs at a time.

### Pinned artifacts

Frozen protocol SHA-256: `f3e9ccc168475a350296373999427c21740c4ce92c6dc234edf69b7ff7b5ac60`.

Library SHA-256: `90251d1bdb077e6649b5d80fc1293714c02dd8db3ab15b18978da6829ebb5bc3`.

Rendered text SHA-256: `bbfeb6430462b4e0edafd6a54b17fa362181738f6319285b51db4ec145a5d8b4`.

Dataset revision: `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`. Official scorer source revision: `0cc508e6bfcfff1fe6352d90ff5957ba38a64bba`.

| Dataset file | SHA-256 |
|---|---|
| skills/test.jsonl | `1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81` |
| queries/test.jsonl | `4d07a35f40e597f8468c6dab480c95a1f9b40ecbde80274081ba2e7c643d0f6d` |
| qrels/test.jsonl | `edbfae0b8109de460d54e27e6235b1b47316f34956744db0571ad1b4ddb860af` |

| Encoder repository | Revision | Weight SHA-256 |
|---|---|---|
| google/embeddinggemma-2 | `914f7f89142e33e77833254d9c9b90c3cef7303b` | `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79` |
| jinaai/jina-embeddings-v5-text-nano | `8a7f00aac812071b69403df470f1038ec85f8925` | `f07de152e254b5d03b1273e0bca0a11dfade686508503500d460813428539651` |
| Qwen/Qwen3-Embedding-0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |
| donghongjiang/SkillReason-embedding-0.6b | `1ef1a664d60c1fb0951a13cceace57a07c877c17` | `d639543800f3587fc62f2bc3c55de09a5f3e404de8548e90e9ab098b98559585` |
| ThakiCloud/SKILLRET-Embedding-0.6B | `67f1594c46c40e13a5eb2f4092cc85c98c9ed32d` | `f73118cac018ffa7ebb5a1ffbdf82034490dfb7f2559558f1e79277f1e8de172` |
| pipizhao/SkillRouter-Embedding-0.6B | `c03c9bcee9fce92ab0262bb6dcf54d174a8ba558` | `cbab45b8a3c786b8c23aedb24fb22aff74d0e9b62a52369a3090c37f26b00360` |

Other deployment pins:

- Qwen3-0.6B-Q8_0.gguf: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`; cached repository revision `23749fefcc72300e3a2ad315e1317431b06b590a`.
- Qwen3.5-4B-Q4_K_M.gguf: `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`; cached unsloth revision `e87f176479d0855a907a41277aca2f8ee7a09523`.
- JevK5-4B-v0.3-Q5_K_M.gguf: `b1df9869dd14c0f5b243fa956dd985f0141c3d23ec4847da4c503c73c1050601`; alibiserikbay/JevK5-GGUF revision `ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a`.
- Qwen3-Reranker-8B.Q4_K_M.gguf: `b683a127768786c0ccd87859b21a2c061ea70f30a7d14efd4476683854eea197`; mradermacher/Qwen3-Reranker-8B-GGUF revision `179cfd14a6fa8ac85a8fc9520f982dbef57ef188`. Internal tensor extents match the complete5,026,265,760-byte file; no independently verified publisher LFS checksum is available.
- Small reranker source: ThakiCloud/SKILLRET-Reranker-0.6B revision `ce09c883af78affc72a85014e040f3d28a5b1711`. Exact tensor-at-a-time FP16 cast; original unchanged. Local sharded conversion manifest SHA-256: `0ebb8d138969d9ad6ac6eac234753b5307e531cb76f72db9cb8f2957f1fcaaca`. Per-shard hashes are in that manifest.

### Actual new embedding work

| Model/stage | New documents | New queries | Document-forward seconds | Startup seconds |
|---|---:|---:|---:|---:|
| eg2_text270m_fp32 / pilot | 30 | 1 | 2.763 | 14.211 |
| eg2_text270m_fp32 / full | 5707 | 16 | 526.829 | 14.514 |
| jina_nano239m_fp32 / pilot | 30 | 1 | 1.835 | 11.435 |
| jina_nano239m_fp32 / full | 5707 | 16 | 346.044 | 8.628 |
| qwen_embed_06 / full_cached_refresh | 13 | 0 | 1.083 | 6.762 |
| skillreason_embed_06 / full_cached_refresh | 13 | 0 | 1.003 | 7.455 |
| skillret_embed_06 / full_cached_refresh | 13 | 0 | 1.304 | 10.244 |
| skillrouter_embed_06 / full_cached_refresh | 13 | 0 | 1.008 | 7.126 |

Total: **11,526 new document forwards + 34 new query forwards +8 warmups =11,568 embedding forwards**. New document-forward time is881.870 seconds; known new query-forward time is2.734 seconds. These totals exclude historical cache creation, startup, warmup and I/O.

Each full EG2/Jina index reused299 document vectors (including30 created in its new canary) and4 query vectors. Each0.6B index reused5993 document and20 query vectors, freshly re-encoding13 headers changed by the common character cap. All new embedding inputs reported zero truncation. Historical cached timings are retained separately, never treated as fresh or free.

Full-index sampled telemetry: EG2 minimum host/guest free4.24/3.52GiB, GPU free6750MiB, maximum59C; Jina minimum host/guest free3.63/3.84GiB, GPU free6880MiB, maximum59C. Neither index triggered the guard. Resource traces and model manifests, not weight size alone, support fit statements.

### 8B failure, preservation and recovery boundary

The8B scorer made3 forwards: synthetic warmup, one successful benchmark candidate, then a failure on the next candidate in q-03666/random/50. The first complete ranking is invalid; all5 later cases are unattempted. The guard did not fire and no OOM was recorded. The exact failure message/raw response was not saved before the ValueError, which is an instrumentation gap.

A missing native yes/no token in the returned top256 is the likely cause, inferred from the adapter’s failure branches, the forward count and the matching small-model prompt length of331 tokens for that candidate. This is not conclusively verified against the omitted8B raw response. It is not evidence that8B cannot fit: measured free GPU memory stayed at least3120MiB and host free RAM at least2.86GiB.

Recovery would require a separately versioned adapter that persists each raw response before validation and checks a reliable targeted yes/no-logit path on non-benchmark canaries before another frozen comparison. No probability-floor substitution, silent scoring change or retry of the failed request occurred in this run. All original failures remain preserved.

### Validation, uncertainty and release

Eleven unit tests passed. The independent audit verified all120 libraries and official hand-metric cases. The additional source audit matched120 reference memberships and18,360 metric scalars. Two notebook code cells executed successfully over183 metric cells. The SVG passed XML/data-structure checks; no browser visual review is claimed.

At6006, hybrid-minus-BM25 Recall@5 is+0.175 with a descriptive paired-bootstrap95% interval[0.050,0.350] (2000 resamples, seed5150;4 wins/16 ties/0 losses). Its NDCG@10 delta is+0.10845, interval[0.03761,0.19566]. These are not multiplicity-adjusted or independent-final-test confidence claims.

No new paid calls, model downloads, training, external publication or Git push. Model processes and the task telemetry monitor were stopped. Final observed GPU state:0MiB used,8006MiB free,47C. SearchJev training remains queued and was not started during this audit.

## Read-only evidence audit

A fresh low-priority audit checked the existing local sources and arrays. It made no GPU, inference, training or model-network calls and installed nothing. It used 1 MiB streamed hashes and 64-row vector chunks, one BLAS thread, a 1 GiB address-space cap, and live 2 GiB host/guest RAM reserves with no swap-counter increases during the audit. Peak RSS including gold verification was 121.09 MiB. No source arrays, frozen protocols, predictions, metrics or failure cases were changed.

- **28 arrays:** SHA-256, shape, dtype, finite checks and norm ranges are recorded. All are finite. Ordered document-ID, uncapped/capped-text, historical 128-query and selected 20-query hashes are explicit.
- **Four cached 0.6B encoders:** each retains 5,993 document rows and all 20 query rows bit-for-bit. All 13 changed header identities were independently derived from the 1,024-character cap; their source/output vector-row and old/new text hashes are recorded. Source array hashes match existing cache pins.
- **EG2 and Jina:** each has 299 reused document rows and 4 query rows matching its existing source arrays bit-for-bit. Pilot/full role, text view, model-manifest hash, dtype, attention, batch and maximum-token settings agree. Current tokenizer/config hashes match pinned downloads. Prefix and L2 normalization behavior are checked against available native source/config; EG2's historical runtime also records matching prefixes. The pilot-array mapping names the immediate source; that pilot itself contains 2 reused document rows and 1 reused query row. Available compatibility evidence cannot establish the exact historical loaded model/tokenizer/worker bytes. Jina has no prior historical array-hash receipt; its source arrays are hashed now.
- **Ranking replay:** 24/24 dense array-to-ranking checks match (six encoders × two fixed queries × sizes 50 and 6006). All 120 equal-weight RRF60 hybrid rankings also match when independently rebuilt from saved BM25 and SkillRet rank channels. These are bounded consistency checks, not fresh model benchmarks or a full independent BM25 reindex.
- **Pinned gold:** all 20 evaluator gold dictionaries match the local upstream qrels; all 120 seeded, nested libraries and their display order reconstruct exactly; all 2,040 retrieval rows have unique in-library IDs. Recomputed Recall@5 at 6006 is BM25 0.625, SkillRet 0.700 and hybrid 0.800. Completeness is relative to pinned official positives, not exhaustive semantic relevance. Upstream network origin was not revalidated.
- **Public inputs:** exact test queries (6,053,507 bytes), qrels (661,204 bytes), the 128-query panel (157,802 bytes), and the 6,006-row corpus metadata projection are included. The original 117,606,037-byte corpus was stream-hashed and matched its pin. Full skill bodies and vector arrays are omitted to keep the archive modest; the projection cannot reconstruct the original corpus-file hash, and a fresh vector replay requires the existing local arrays.
- **JevK5:** existing reviewed adapter files match their earlier review pins and are included with the Apache 2.0 license. The saved field named `raw` contains processed adapter scores/probabilities, not original completion responses or pre-normalization logits.

The exact two-sided paired sign-flip test for hybrid versus BM25 Recall@5 at 6006 gives **p = 0.125** (4 positive differences, 16 ties). The paired bootstrap 95% interval [0.05, 0.35] remains descriptive; this exposed 20-query panel does not establish a general winner or independent confirmation.

Receipt: `scaling5150/provenance-audit/receipt.json`, SHA-256 `ce46cf064cd71d4b258253b534b69f938e39b4beaf528659db35228b63def751`. `independently_complete_provenance=false` remains explicit. The source audit is read-only; this package adds evidence and corrects report provenance language only.
