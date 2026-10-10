# Benchmark contract

All requirements below are proposed implementation acceptance criteria. They are not evidence that a harness has already passed them.

## Dataset and representation

Official current counts to validate: test skills 6,006; test queries 4,392; test qrels 7,187; train skills 10,123; train queries 63,259; train qrels 127,190. Source: https://huggingface.co/datasets/ThakiCloud/SKILLRET and https://github.com/ThakiCloud/SKILLRET (accessed 2026-10-05). Freeze a full Hub commit, not only the moving `main` alias. The README's example short revision `a050ad2` is a lead to resolve, not a revision already downloaded or validated here.

Official document view is `name | description | skill_md`. In Track B use a separately versioned name+description view to represent the proposal's header-discovery setting. Never imply that full-body and header-only systems saw identical evidence. Save exact rendered inputs or reproducible hashes plus the rendering code; record true token counts and truncation separately per model. Index-side generated views are new representations, not edits to source skills.

Only opaque query ID and query text may enter scoring. Exclude query-side skill_ids, skill_names, required-skill count k, original source IDs and label-derived features. Qrels live behind the evaluator boundary. Original IDs can remain in a private evaluator manifest for joins, never prompts.

## Ranked retrieval metrics

Let G_q be the labelled relevant skill-ID set for query q, and R_q@k the unique retrieved IDs in the first k positions. Compute macro averages over the fixed query manifest, with per-query results retained.

- Recall@k = |G_q ∩ R_q@k| / |G_q|.
- Completeness@k = 1 only when every member of G_q is retrieved in the first k. It does not penalize extra irrelevant results; final-set precision does.
- MRR@5 = 1/r for the first relevant rank r≤5, otherwise 0. A gold at rank 6 does NOT contribute 1/6.
- NDCG@k uses binary relevance, logarithmic rank discount and ideal DCG for the query's available labels; verify exact settings against the official scorer.
- MAP@k must match the pinned official `pytrec_eval` convention. Do not substitute a convenient alternative AP denominator.

SkillRet reports NDCG, Recall, Completeness and MAP at 5/10/15. MRR@5 and final selection are proposal additions. Source: https://github.com/ThakiCloud/SKILLRET#metrics.

Record Hit@k separately if used. It means at least one gold is present and must never be relabelled Recall@k on multi-positive queries. Report subgroup results for gold-set sizes 1, 2 and 3 (or the actual validated distribution), but do not expose that number to a model.

## Candidate-pool and oracle diagnostics

Save candidate Recall@20/50/100 and Completeness at those cutoffs. Also compute the best attainable top-k metrics if the true relevant members of a fixed pool were perfectly sorted. This is a diagnostic, not a model result.

Gold-insertion oracle: force missing gold into a controlled pool and run the same reranker. Label this oracle prominently and exclude it from production comparisons. The published 88.89 oracle NDCG@10 is not a theoretical maximum. The ordinary pipeline must not use qrels to build candidate lists.

## Final selection and activation

Require `selected_skill_ids` as a validated set. Report exact-set match, set precision/recall/F1, invalid-ID rate, empty-selection rate, and selected-set size. On a single-gold task exact-set accuracy means exactly that gold and no extras. A ranking's Hit@1 is not automatically final selection accuracy.

Use a common downstream LLM and prompt when isolating discovery method. Then vary final LLM as a separate factor. No gold count is provided to a selector. Optional dependency order, compatibility and no-skill activation need independent annotations/benchmarks; ordinary SkillRet qrels do not by themselves supply all of those labels.

A reranker's local NONE means no candidate in its supplied pool fits. It does not prove that no appropriate skill exists in the full corpus. Global abstention/skill activation needs genuine no-skill cases or counterfactual utility evidence.

## Scale experiment

Sizes: 10, 100, 1,000, 6,006. Two valid constructions are possible and answer different questions:

1. Fixed global nested libraries, evaluated on a documented common eligible query panel; disclose any coverage/selection bias.
2. Per-query, per-seed gold-conditioned nested libraries with controlled distractors; this isolates interference while guaranteeing that discovery is possible, but is NOT the official fixed-pool protocol.

Choose and label the protocol before running; do not mix both in one curve. In protocol 2, the harness uses gold only to construct the controlled library, never to score or hint to the model. All methods receive the same library for the query. Random and semantically overlapping distractors form separate conditions. At 10 skills, returning all 10 makes Recall@10 and Completeness@10 trivial; prioritize @5, ordering and final selection there.

## Latency, resources and cost

Report per-query end-to-end p50/p95, retrieval-only, reranker-only and final-selection time; cold startup separately; fixed warm-up; fixed concurrency; CPU threads; batch sizes; actual GPU offload; RAM/VRAM peaks; indexing time and index size. Do not count cached identical model answers as fresh-inference latency. Keep provider prompt-cache discounts separate from client response-cache hits.

Record input/output/reasoning tokens for LLMs, embedding tokens separately, and text bytes as a tokenizer-independent descriptor. A CPU BM25 method has no LLM input tokens unless a downstream selector is used. Relative latency must use the same machine and baseline configuration; it does not eliminate hardware dependence. Provide absolute latency too.

Record provider, exact model response ID/version, deployment precision, endpoint/tier, batching, token prices, query counts, retries, storage and allocated GPU time. Use CAD totals, with exchange-rate source/date or an explicitly hypothetical exchange-rate assumption.

## Failure handling and reproducibility

A missing query must not silently disappear from averages. Preserve a fixed denominator and report failure rate. Unknown IDs, duplicates, malformed JSON, timeouts, OOM, refusal and context overflow are explicit statuses. A retry policy is fixed in advance and all attempts count toward latency/cost. For controlled reranking, reject invalid permutations or use a declared fallback that is reported separately.

Keep immutable run directories keyed by configuration and data hashes; append new runs rather than overwriting. Save prompt/readout code, model/tokenizer hashes, candidate ordering and raw outputs. Test stable tie handling against the official evaluator. Qualitative review is blinded to method where practical. New judged relevance belongs in an auxiliary qrels file and may not overwrite official qrels.

## Minimum metric tests for Codex

Two gold IDs at ranks 1 and 4: Recall@5=1, Completeness@5=1, MRR@5=1; Completeness@3=0. Two gold IDs with one retrieved: Recall@5=0.5, not 1. First gold at rank 6: MRR@5=0. All gold missing: zero retrieval metrics. Duplicating the same relevant ID must not increase recall. Test identical rankings, reversed rankings, empty results, all-relevant results, ties, and fixed-denominator missing queries. Cross-check official metrics on hand-constructed cases before a dataset run.
