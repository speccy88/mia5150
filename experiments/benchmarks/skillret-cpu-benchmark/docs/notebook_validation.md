# Fresh recovery notebook: validation and reproduction

## Evidence boundary

The original full-test measurements were completed and independently audited before the working tree became unavailable. Their surviving aggregate readouts are retained as historical observations. No old raw rankings, per-query metrics, latency samples, embeddings, missing metric values, or confidence intervals are reconstructed from those aggregates.

The notebook's primary analysis uses newly saved raw results from the bounded recovery run. It requires the same first 64 frozen query IDs across all seven models plus BM25 and all four catalog sizes, or the complete common first48, then first32 fallback. The first48 prefix was preregistered before extension-quality outcomes as a runtime-budget fallback. Larger individual checkpoints remain separate. Original 4,392-query aggregates and the earlier common-eight exercise are in a visibly separate historical appendix.

## Run only after inference stops

```sh
.venv/bin/python src/build_notebook.py --python-execute
```

This executes each ordinary Python cell sequentially in-process and captures its actual output. It does not import model code, perform inference, download anything, or change sandbox settings. No Jupyter kernel is started in recovery. The original environment blocked kernel socket binding; the notebook preserves that honest validation boundary rather than claiming kernel-based execution.

Use `--build-only` while inference is active. That writes an explicitly unexecuted scaffold with no fresh findings. The builder refreshes the top narrative only after all calculations succeed. The available nbformat schema validator is used when installed; otherwise required v4 structures and every code-cell's Python syntax are checked and the narrower check is recorded.

## Independent checks

- Processed dataset artifacts match their reconstruction manifest hashes
- Full and selected qrels agree; all IDs are valid, gold sets are nonempty, source counts match, and no duplicate relevance pairs occur
- The 64 selected IDs and original 256 IDs match the outcome-independent SHA256 query-order rule; the 32 timing IDs are their exact prefix
- Catalogs have correct unique membership, preserve every gold skill, and saved ranks belong to the relevant query-specific nested membership
- Every stored ranking is unique and all nine relevance metrics are independently recomputed from ranks and released qrels, without the runner's metric code
- Primary quality requires an exact model/method/catalog/query grid, not a matching row count alone
- Every binary reranker uses the verified BGE-hybrid candidate set and its independently recomputed initial shortlist MRR; available raw pair scores independently reproduce native score sorting and stable tie order
- Native audit tables count each query–candidate pair once across catalogs, distinguish intentional query shortening from candidate loss, and keep unavailable audits explicit
- Completed results have a completion receipt; growing or explicitly partial files are excluded
- Full-catalog latency requires the exact same 32 IDs × 2 repeats. Quality and timing source/configuration fingerprints are checked, and fresh timed decision shortlists are compared with BGE quality rankings
- Quality–latency merges are condition-level, one-to-one, with both query denominators shown
- Missing timing/setup/model results remain missing, rather than extrapolated or set to zero

Fresh bootstrap intervals resample paired query-level MRR differences with replacement, 2,000 draws at seed 20261002, using linear percentile endpoints. They are descriptive query-sampling intervals, not hardware/training uncertainty, causal evidence, multiplicity-adjusted significance, or out-of-domain guarantees. The old aggregate-only results are excluded.

## Companion output

Successful execution writes `docs/notebook_paired_bootstrap.json` (schema version 2) with exact selected fresh cohort IDs/hash, model/method/catalog, baseline identity, candidate depth, paired point differences, 95% interval endpoints, wins/ties/losses, seed, resample count, method and source SHA256s. The report must verify its selected cohort and independently recomputed point differences before including these intervals.

Cohort hashes use SHA256 of sorted query IDs joined by newlines with no trailing newline.

## Lightweight tests

```sh
.venv/bin/python -m unittest discover -s tests -p test_notebook.py -v
```

Synthetic fixtures test metric edge cases, corrupted candidate metrics, changed shortlist membership, duplicate/missing model grids, exact common64→48→32 fallback, exact timing pairs, deterministic bootstrap, Unicode line-separator-safe JSONL reading, generated notebook syntax/schema, and truthful sequential execution capture. They do not execute models or scan actual result files.
