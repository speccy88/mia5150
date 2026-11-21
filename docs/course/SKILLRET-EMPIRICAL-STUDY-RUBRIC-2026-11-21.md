# SkillRet: controlled empirical benchmark and grading evidence

**Status (21 November 2026):** Independently reanalyzed existing retrieval records; no model inference, API calls, or new training. This note applies the course's **Benchmark and Empirical Study** rubric to a deliberately narrow primary study. It does not pool the incompatible prompt, embedding-document-view, or reranker panels.

## Research question and claim

**Question:** At a fixed candidate budget of five, does equal-weight Reciprocal Rank Fusion (RRF60) of BM25 and BGE-small improve skill-retrieval coverage over either component as catalog size grows?

**Primary claim:** On the 4,392-query corrected SkillRet test panel with a fixed 6,006-skill metadata-only catalog, Hybrid improves macro Recall@5 by **10.97 percentage points** over the matched BM25-only ablation, with a descriptive **95% paired query-bootstrap interval of [9.95, 12.02] points**.

**Primary metric:** macro Recall@5, an accepted metric in the original SkillRet paper. **Secondary metric:** MRR@5. This is retrieval, not final agent skill selection or task-execution success.

## Published benchmark baseline (distinct from our internal control)

The published **SkillRet v3**, Table 3, reports **BM25 Recall@5 = 53.03%** and **NDCG@10 = 51.69%** on 4,392 queries and 6,006 skills: [Kang et al., 2026](https://arxiv.org/html/2605.05726v3).

Our **internal matched BM25 control** scores **47.73% Recall@5**. This is *not* a like-for-like reproduction of the paper's 53.03%: the paper evaluates full skill-document text and its own BM25 implementation, whereas our controlled retrieval experiment uses capped name+description metadata, a pinned sparse BM25 implementation, and its own indexing/search conventions. **Do not subtract the published 53.03% from our 58.70% to claim a paper-to-paper performance improvement.** To claim direct superiority to the published baseline, the official full-document BM25 protocol must be reproduced or a new fully matched comparison run.

## Primary controlled design and component ablation

| Factor | Fixed condition |
| --- | --- |
| Benchmark and ground truth | Corrected, pinned SkillRet test revision; 4,392 queries; 7,187 positive qrels |
| Primary catalog | Same 6,006 unique skill IDs for every method, same order and relevance labels |
| Document view | Same pinned name+description metadata view for all three methods |
| Returned candidates | Exactly five skill IDs per query |
| Varied factor | **Retrieval strategy only:** BM25 lexical; BGE-small cosine vector; RRF60 fusion of the two |
| Encoder | Same BGE-small-en-v1.5 vectors, pinned revision, FP32 normalized CLS embeddings |
| Other conditions | Same query text, preprocessing, pooling, label definitions, and evaluated query IDs |
| Secondary scaling | Gold-conditioned catalog sizes N=10,100,1,000 with seeded distractor sets (not an independent open-world test) |

**The ablation is BM25 only / Vector only / Hybrid**, where removing each Hybrid component shows how much the combined retriever benefits from complementary rankings. Catalog size is analyzed separately, not silently changed inside the primary method contrast.

### Results: exact same official queries and catalog-size conditions

Macro Recall@5, mean across catalog seeds 42,43,44 (percent); parenthesized value is the *sample standard deviation across catalog seeds*, in percentage points:

| Skills | BM25 | Vector | Hybrid | Hybrid − BM25 |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 94.93 (0.09) | 93.05 (0.21) | 95.94 (0.07) | +1.01 pp |
| 100 | 74.42 (0.24) | 76.50 (0.09) | 82.83 (0.24) | +8.41 pp |
| 1,000 | 60.00 (0.16) | 62.85 (0.03) | 69.71 (0.22) | +9.71 pp |
| 6,006 | 47.73 (0.00) | 51.73 (0.00) | 58.70 (0.00) | +10.97 pp |

At the full 6,006-skill catalog, seeds 42/43/44 produce **the identical library and identical rankings**. Their zero standard deviation is a consequence of catalog construction, **not evidence of low statistical uncertainty**. Instead, paired uncertainty is estimated by resampling independent query IDs, keeping each query's catalog-seed observations together.

### Primary paired query-bootstrap result

From the original 15,054,930-byte gzip archive (SHA-256 `a55bfecfa99508cebcc0d6c0238575c60bfbd919d46c2e69654f66dcd0b53a35`), 5,000 paired query-cluster bootstrap resamples with seed 20261121:

| Contrast at N=6,006 | Recall@5 improvement | Nominal 95% paired interval |
| --- | ---: | ---: |
| Hybrid − BM25 | **+10.97 pp** | **[+9.95, +12.02] pp** |
| Hybrid − Vector | **+6.97 pp** | **[+6.11, +7.84] pp** |
| Vector − BM25 | +4.00 pp | [+2.77, +5.21] pp |

Intervals are conditional on exposed, synthetic queries with possibly incomplete positive relevance judgments. They do not demonstrate generalization to unseen human requests. Secondary contrasts are exploratory and not multiplicity-adjusted.

### Why the fusion helps: query-level ablation

At N=6,006, seed 42, compared with BM25:

- Hybrid retrieves **more** relevant top-five skills on **924 / 4,392** queries; **fewer** on **247**; **the same number** on **3,221**.
- BM25 retrieves **zero relevant** top-five skills on **1,627** queries. Hybrid recovers **at least one** on **688** of these; Vector alone does so on **642**.
- Hybrid outperforms both single-method components on **342** queries. It is not always the best per-query method.

This is mechanism evidence for lexical/semantic complementarity, not a claim that fusion invariably wins.

## Seeds, variation, and compute accounting

**Catalog-seed design:** 42/43/44 vary deterministic distractor membership for N < 6,006, while each query and its positive qrels remain fixed. These are *catalog sampling replicates*, not model-training replicas. At full catalog size all seeds are identical. The proper independent unit for the paired bootstrap is the **query** (n=4,392), never the 52,704 repeated query/seed/size cases.

Search-only recorded mean milliseconds per query at N=6,006:

| Configuration | Search-only mean | Excluded from this timing |
| --- | ---: | --- |
| BM25 | 7.40 ms | Index build and data preparation |
| Vector | 6.85 ms | Offline batch query embedding, model startup, corpus indexing |
| Hybrid | 14.35 ms | Offline batch query embedding, model startup, corpus indexing |

The above are **not end-to-end retrieval latency nor total computing cost**. They are original recorded timings, not newly remeasured samples. We should not describe the Vector or Hybrid query inference as free. For cost-complete grading, profile the same three pipelines with *fresh* query encoding, initial index/embedding construction, peak memory, CPU/GPU model, repeated warm/cold runs, p50/p95 latency, and an explicit costing basis. No missing charge or GPU time is invented.

**Supporting cost evidence, kept in separate panels:** The BGE-M3 document-view experiment records metadata indexing 7.13 minutes versus matched full-text indexing 16.75 minutes and median fresh-query latency 82.71 versus 83.36 ms on the stated local machine. The Jev/CLM 200-query development study reports cost estimates, not actual invoices. The nine-model OpenRouter audit has list-price estimates but unknown actual billing. None of these provides matched end-to-end resource accounting for the three primary retrievers.

## Rubric decision and outstanding work

| Requirement | Evidence | Status |
| --- | --- | --- |
| Defined established task / accepted metric | SkillRet retrieval; macro Recall@5 and MRR@5; verified 4,392-query records | **Met** |
| At least one published baseline | Published SkillRet v3 BM25 Table 3, plus our internally matched BM25 | **Partially met**: published baseline not reproduced under matching document-view/protocol |
| One factor varied, other inputs fixed | BM25, Vector, Hybrid on same queries, libraries, top-5 and metadata | **Met** for primary comparison |
| Ablation explaining effect | BM25-only, Vector-only, Hybrid; 688 BM25 misses recovered by Hybrid | **Met** |
| Seeds and between-run variation | Three catalog seeds for N=10,100,1000, query-cluster CI; full-corpus invariance stated | **Met with qualification**: seeds are catalog draws, not independent model runs |
| Compute cost of each configuration | Three measured search-only latencies; BGE-M3 separate measured indexing | **Partial**: fresh encoding/indexing and complete latency/cost are not matched in the primary panel |

**Prioritized remaining measurements before claiming full rubric compliance:**

1. **Published baseline reproduction** — run full-document BM25 from the official paper's code/data and report both it and our metadata-only BM25 under a clearly declared controlled protocol; do not conflate the official paper's numbers with our metadata setup.
2. **Matched full-path resource profile** — benchmark BM25 / Vector / Hybrid on one fixed CPU/GPU, including index creation, query embedding where applicable, retrieval scoring, peak memory, p50/p95 end-to-end latency, and total time. Use repeated executions and retain raw timings.
3. **Report controls and provenance** — include software version pins, dataset SHA, catalog seeds, raw artifact SHA, query bootstrap seed, hardware, and independent audit. Limit conclusions to retrieval on previously exposed synthetic requests.

## Reproduction and source references

- Main frozen study: [October matched retrieval](../../experiments/benchmarks/skillret-gap-closure-20261024/report/RESEARCH_DRAFT.md#completed-full-query-retrieval).
- Read-only verification: [Independent full-query retrieval audit](../../experiments/benchmarks/skillret-gap-closure-20261024/audit/INDEPENDENT_FULL_RETRIEVAL_README.md).
- Raw archive: `experiments/benchmarks/skillret-gap-closure-20261024/results/retrieval.jsonl.gz` (byte-verified above).
- Original summary: [full-retrieval.csv](../../experiments/benchmarks/skillret-gap-closure-20261024/report/full-retrieval.csv).
- Complementary representation study: [BGE-M3 full-text report](../../experiments/benchmarks/skillret-bge-m3-fulltext-20261108/REPORT.md).
- Complementary rerank development study: [Jev versus CLM](../../experiments/benchmarks/skillret-reranker-study/README.md).
- Complementary prompt study: [OpenRouter nine-model audit](../../experiments/benchmarks/openrouter-prompt-audit-20261108/results/REPORT.md).
- Exact stored analysis: `scripts/skillret_empirical_analysis.py` regenerates the paired bootstrap from the compressed immutable archive (NumPy dependency, no model/API calls).

## Paper-ready conclusion

> We evaluate three fixed retrieval architectures on the corrected SkillRet benchmark using the same 4,392 queries, 6,006 skill descriptions and top-five candidate budget. Hybrid BM25+BGE-small retrieval improves Recall@5 over BM25 by 10.97 percentage points (95% paired query-bootstrap interval: 9.95–12.02 points). Component ablations show that the gain reflects complementary lexical and semantic retrieval: Hybrid recovers a relevant skill on 688 queries missed entirely by BM25@5, but reduces relevant coverage on 247 other queries. The measured search-only overhead is 14.35 ms per query for Hybrid versus 7.40 ms for BM25; full pipeline costs remain to be measured. These findings are restricted to the exposed synthetic SkillRet questions and do not establish downstream agent task success.
