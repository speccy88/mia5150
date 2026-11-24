# SkillRet Decisions API scaling protocol

## Experimental design

The study uses 200 fixed queries from the previous BM25 and Jev development trial. The binary judgments come from the SkillRet test split at revision `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`, which contains 6,006 skills and 4,392 test queries. The query order, texts and qrels were checked against the previous trial.

For each query and seed (42, 43, 44), October's catalog sampler starts with all known relevant skill IDs in sorted order. It shuffles the remaining sorted IDs with Python `random.Random`, seeded by the integer value of the full SHA-256 digest of the compact JSON array `[seed, query_id, "membership"]`. It takes a prefix of the shuffled IDs to reach the target size. Catalogs are nested within each query and seed and shared across retrieval methods. The sizes are 50, 100, 250, 500, 1,000, 2,000 and 5,000, for 4,200 catalogs and 12,600 logical reranking cases. Each case requests 20 judgments, or 252,000 logical judgments.

## Retrieval and metadata

All methods use stripped names and descriptions rendered as `name | description`, with the description capped at 1,024 characters. The October sparse BM25 implementation uses catalog-specific document frequencies and average document length with `k1=1.5`, `b=0.75` and case-folded word tokenization. Vector retrieval uses BAAI BGE-small-en-v1.5 at revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, the pinned FP32 ONNX encoder, L2-normalized CLS pooling, 512-token truncation and the October query prefix. Hybrid uses reciprocal rank fusion with a constant of 60 and up to 100 BM25 and vector results each. First-stage ties are resolved by skill ID.

The historical Jev trial used a colon separator, truncated the combined name and description to 1,024 characters, and used BGE token clipping to at most 128 tokens including special tokens. Its BM25 used English stopwords. Those settings and the full catalog size of 6,006 distinguish that result from this scaling study.

## Decisions requests and cost

The REST client sends the fixed model `gpt-6-luna` to `https://api.openai.com/v1/decisions`. Each request has one shared input containing the query and rendered metadata for 20 candidates, plus one independent predicate question per candidate. Skill IDs determine display and question order. The request contains only the query, candidate IDs and metadata, model, and [frozen instructions](rubric.json). Responses must provide all named predicates with finite probabilities in [0, 1]. The reranker sorts descending by probability, then uses original retrieval rank and takes five. The score evidence retains all 20 candidates.

The SHA-256 of the exact canonical JSON request keys the persistent cache, which only reuses successful results for identical payloads. The synthetic check and ten fixed benchmark smoke cases precede the full run. The runner uses SQLite for durable attempt reservations, usage, failures and validated results. Concurrency and request rates are bounded. Uncertain dispatches retain their budget reservations.

The published rates per million input tokens are USD 0.10 uncached, USD 0.01 cached and USD 0.125 for a cache write at the documented 30-minute TTL. Decisions output tokens carry no charge. The model documentation describes a 10% regional-processing premium and doubled input rates for requests above 272,000 input tokens. The runner uses the standard endpoint configuration, an observed smoke projection and an in-flight reserve against a USD 15 cap. [Decisions guide](https://developers.openai.com/api/docs/guides/decisions) · [Model pricing](https://developers.openai.com/api/docs/models/gpt-6-luna) · [API reference](https://developers.openai.com/api/reference/python/resources/decisions/methods/create)

## Evaluation

Recall is the fraction of binary relevant skills retrieved per query and is averaged with equal query and seed weights. Candidate Recall@20 is the upper bound on Recall@5 given the pool. Completeness@5 is the fraction of cases retrieving every relevant skill in five results. MAP@K uses all positive qrels in the per-query AP denominator, as in the existing SkillRet evaluator. Per-seed metrics and sample standard deviations across the three seeds accompany the means.

Paired differences compare each case with its own retrieval baseline. The 95% percentile bootstrap resamples 200 query IDs with replacement using 10,000 replicates and seed 42, holding all paired observations and seeds for each sampled ID together. Valid-only metrics use successful scored cases and their matched baseline. The operational ranking uses a retrieval fallback after a failed attempt. Unattempted cases remain missing and prevent a complete operational estimate. API latency and token totals come from actual attempts. Local cache reuse is tracked separately from fresh HTTP latency and provider cache usage.

The development queries were previously exposed, and the sampler always includes the known positives. The study measures sensitivity to additional distractors on this panel. Historical prompt, October and Jev measurements retain separate evaluation-population labels in the presentation figure.
