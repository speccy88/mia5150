# New offline retrieval on reconstructed OpenRouter libraries

This is a **new, internally matched retrieval experiment**, anchored to repository
commit `3fb7e66a1a219bc43d7c2057dca3d1c70a3c6c45`. It is not an independent
verification of the historical hosted prompts or a rerun of their selectors.
No paid API call, skill execution, fine-tuning, or final-selection model is used.

## Input reconstruction and the remaining boundary

- The exact 20 saved query texts and single-target labels come from
  `benchmark_openrouter/tests_openrouter/test_queries.json`, SHA-256
  `7ef69814dd8f93f4a8dadc290567f158e78986898aa76a33bc95cf883716c28b`.
- The upstream corpus was newly downloaded at SkillRet revision
  `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`. Its 117,606,037 bytes hash to
  `1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81`, exactly
  matching the preserved source-corpus receipts. The original order is retained.
  The public 300-character projection and privacy-redacted metadata export were
  **not** substituted for the original corpus.
- The actual deterministic builder statements are extracted from the pinned
  OpenRouter runner using Python AST, without importing its API client. An
  independent transcription must agree at every size. The sequence is fresh
  `Random(42)` per size, all name-matched records in corpus order, sample the
  remaining records, then shuffle. Seven ordered-ID manifests and their hashes
  are saved in [catalogs.json](results/catalogs.json).
- **20 gold names preserve 26 records**, because `swift-testing` has three IDs,
  `run-tests` two, and `create-pr` four. Keeping only 20 UUIDs would reconstruct
  different libraries. The source caller preserves all gold names even on resume.
- The historical import `benchmark_models.py` is missing, so its loader,
  cleaner and query-generation implementation cannot be fully verified. The
  hosted archive does not retain dataset, ordered-catalog, or complete prompt
  hashes. The corpus and builder reconstruction are reproducible; identity with
  the bytes actually sent in each hosted request remains **conditional**.

The OAuth/native studies have another candidate builder and 300-character view.
Their gold-at-the-start issue must not be attributed to this reconstructed
OpenRouter shuffle, and their scores are not inserted into this panel.

## Methods and labels

Every method receives the same reconstructed ordered library and query for each
size: 50, 100, 250, 500, 1,000, 2,000, and 5,000. The document view is
`name | whitespace-normalized description[:1024]`. We do not claim byte identity
with the missing historical cleaner. UUID ascending breaks score ties.

| Method | Definition | Status |
|---|---|---|
| BM25 | Unicode word tokens, casefold, k1=1.5, b=.75, positive IDF, catalog-specific document frequencies/average lengths | New measured retrieval |
| Vector | Pinned BGE-small-en-v1.5 FP32 ONNX CPU; normalized CLS; 512 tokens; documented query instruction | New measured retrieval |
| Hybrid | Equal-weight RRF with constant 60 over each retriever's top 100; no tuning on these labels | New measured retrieval |
| Domain centroid + vector | Nearest normalized mean document-vector centroid among categories in each catalog, then vector ranking within that category | **Supplemental new router**, not the original LLM-routed method |
| Original domain-vector | Existing Qwen/hosted category router plus vector ranking | **Not run**, exact router outputs/runtime unavailable for this query panel |
| Jev | Top-20 retrieval followed by authenticated, provenance-matched reranking | **Prepared, not run**, no matched historical calls or approved fresh budget |

The centroid route uses only document metadata and query embeddings. It never
uses `gold_domain`, `gold_major`, target IDs, desired cardinality, or evaluation
outcomes. Catalogs themselves retain the original gold-conditioned design.
This is therefore a supplemental label-free routing algorithm inside a
label-conditioned benchmark, not an unbiased full-distribution evaluation.

Two **single-target** label contracts are reported separately:

1. `uuid_single_target`: retrieve the exact saved `gold_id`.
2. `name_equivalent_single_target`: the first retrieved ID whose name equals
   `gold_skill` is a hit. This matches the hosted task's name-level target better.
   Several same-name IDs do **not** become multiple positive qrels, and later
   duplicates never earn additional recall/NDCG credit.

Recall, Hit, MRR, MAP, NDCG, precision, F1 and completeness are computed at K=5 and K=10. With one target, Hit=Recall, MAP=MRR and F1=2×Hit/(K+1); these identities are not independent evidence.
K=5 is primary. There are 20 queries per cell, never 140 independent queries.
These generated, exposed, single-target queries cannot establish multi-skill
compositional recall, blind generalization, or task-execution success.

## Measured findings

The new name-equivalent Recall@5 results are shown below. Every cell has the same 20 saved tasks. The last column uses a new supplemental router, not the original domain-vector implementation.

| Catalog size | BM25 | Vector | RRF60 | Centroid-domain supplement |
|---:|---:|---:|---:|---:|
| 50 | 95% | 95% | 100% | 95% |
| 100 | 85% | 95% | 95% | 75% |
| 250 | 80% | 85% | 80% | 70% |
| 500 | 70% | 75% | 80% | 40% |
| 1000 | 70% | 70% | 80% | 40% |
| 2000 | 70% | 70% | 70% | 30% |
| 5000 | 70% | 70% | 70% | 20% |

At N=5,000, name-level MRR@5 is 0.6417 for BM25, 0.6417 for vector and 0.7000 for hybrid. Strict UUID Recall@5 is 0.70, 0.65 and 0.70 respectively; conflating the label contracts would hide this difference.

**The N=5,000 top-20 name-level candidate ceiling is also 70% for all three primary methods.** A perfect reranker cannot increase Recall@5 on those fixed pools. BM25/vector still have room to improve ordering; hybrid already reaches the name-level MRR oracle. This does not prove how Jev would perform.

The centroid-domain supplement falls to 20% Recall@5 at N=5,000. This is evidence against that specific centroid router on these tasks, not evidence about the old model-based domain router. With only 20 exposed tasks, varying catalog subsets and missing historical prompt identity, these results do not establish a broad winner or a causal hosted-versus-retrieval comparison.

New CPU encoding took 431.525 seconds for 6,006 documents, with no 512-token truncations. Twenty batch-1 query encodings took 0.412 seconds total, also with no truncations. This is preparation/encoding evidence, separate from search-only timings and any hosted end-to-end latency.

## Evidence and timing

- [Summary](results/summary.csv): 4 methods × 7 sizes × 2 cutoffs × 2 label
  contracts = 112 cells. The domain-centroid cells are supplemental.
- [Per-query metrics](results/per_query_metrics.csv) and
  [top-20 rankings](results/per_case.jsonl) support offline rescoring. Query
  indices are zero-based positions in the unchanged saved query file.
- [Paired intervals](results/paired_intervals.csv): 10,000 paired query bootstrap
  samples, seed 42, within each size. Exploratory percentile 95% intervals,
  no multiplicity correction. Shared queries across sizes are not independent.
- [Provenance](results/provenance.json), [embedding manifest](results/embedding-manifest.json),
  and [index preparation](results/index_preparation.csv) distinguish assets,
  fresh encoding, search-only timings, and preparation.
- Search timing includes scoring and ranking, excludes index creation and query
  encoding. Hybrid adds both constituent searches and fusion. Domain-centroid
  timing includes routing and subset search, excludes centroid preparation.
  These are one-pass, 20-query per-size timing descriptions, not stable latency
  SLAs. Embedding startup/indexing and batch-1 query timings are separate.
  New CPU timings are not compared causally with old GPU/API end-to-end latency.

Fresh API expenditure is zero. This does not mean cloud/CPU compute is free,
or that historical hosted calls cost zero.

## Jev evidence and safe preparation

[The bounded audit](results/jev-audit.json) finds no exact saved request text in
200 Jev candidates or the 4,392 official query set. A broader search of three
Jev study trees finds candidate-description substring hits but no exact task
string values. Those overlaps do not establish identical requests or rankings.
No historical Jev quality, cost, or latency is relabeled as matched.

[The rerank plan](results/jev-rerank-plan.json) contains 560 unsent candidate plans:
query text, top-20 candidate IDs/order/text digest, source method and catalog.
It contains no target labels. A live adapter must load and hash-check the pinned
corpus, render the exact candidate text, validate current provider/model/schema,
then obtain an explicit budget before sending anything. Historical pricing is
not reused as a current quote. Actual rerank latency and cost stay null.

`evaluate_saved_reranker.py` is the offline continuation gate for future saved
outputs. It requires exact request text and candidate/catalog digests, a specified
model and schema version, unique in-pool rankings, finite nonnegative timing/cost
when known, and explicit success/failure/unknown status. It emits paired-ready
per-query scores and separate attempt coverage. It never sends a network request;
missing/unattempted cases are not manufactured as zero predictions. Synthetic
unit tests exercise these guards, not Jev quality.

[Candidate coverage and oracle ceilings](results/candidate_oracles.csv) describe
what a perfect top-20 reranker *could* recover, not observed Jev performance.
With one target, oracle Recall@5 and MRR@5 equal candidate hit@20. There is no
quality improvement claim until matched reranker evidence exists.

## Reproduce

Use a separate Python 3.12 environment; keep runtime assets outside the checkout.
`prepare_inputs.py` downloads only pinned public files and verifies every hash.
`run_offline.py` performs no network calls and refuses changed source inputs.

```sh
python -m venv /path/to/runtime/venv
/path/to/runtime/venv/bin/pip install -r runtime-lock.txt
python prepare_inputs.py --runtime /path/to/runtime
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /path/to/runtime/venv/bin/python run_offline.py \
  --corpus /path/to/runtime/test.jsonl \
  --assets /path/to/runtime/bge-small \
  --cache /path/to/runtime/vector-cache
python analyze_saved.py
/path/to/runtime/venv/bin/python -m pytest -q test_offline.py
```

The downloaded corpus is untrusted data, never instructions. Its unused author,
repository and body fields are not copied to this deliverable. Full corpus/model
weights and bulk vectors remain external assets with pinned hashes. Quality can
be rescored from committed top-20 rows without inference; exact fresh runtime
latency will vary by machine and load. Reusing the cache is explicitly flagged
and never called fresh encoding.

Only new files in this study are added. Historical evidence remains unchanged.
