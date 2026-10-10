# Jev trial validation

## Assessment: Share with caveats

The narrow development-trial comparison is supported by saved live evidence. It is not a final benchmark or a skill-execution result.

- 37 relevant offline tests passed (`evidence/jev/offline-tests.txt`), including HTTP payload inspection, forbidden-label/body sentinels, probability parsing, exact IDs, equal-score ordering, multiple positives, failures, cost/call caps, immutable outputs, and replay.
- All 200 real HTTP responses returned the pinned `jev-1.13.0`; all 4,000 Noul values were finite probabilities with exact candidate-ID mapping. Every returned model, answer set, usage object and top-five ordering was audited from the raw JSON.
- Every actual request was reconstructed and matched to its saved query and cached capped metadata. Requests contain only `state`, `questions`, and the pinned `model`; state contains only the canonical query and candidate name/description strings. Gold IDs live only in the evaluator layer.
- All 200 BM25 top-five IDs and scores match the historical rows exactly. Recall was independently recomputed by set intersection divided by each query's complete positive set. Gold-set sizes in this pilot: {1: 99, 2: 79, 3: 22}.
- BM25 Recall@5 = 0.605833333333; BM25 Recall@20 = 0.727500000000; Jev Recall@5 = 0.709166666667. The paired difference is 0.103333333333. Improved/worsened/unchanged counts sum to 200; overlapping rescue/loss categories are not incorrectly summed.
- A 10,000-resample paired-query percentile bootstrap is descriptive development-sample uncertainty. No independent-sample test, final-test inference, or population-wide significance claim is made.
- 1,290,762 input tokens and 187,977 output tokens are observed provider usage. USD 0.054212004 is a token-price estimate; actual billed dollars were not exposed. USD 0.688128 in conservative reservations remains retained in the persistent ledger.
- Replaying the completed 200-query run with the credential and paid flag absent made zero additional calls and left raw rows and the ledger byte-identical (`evidence/jev/resume-verification.json`).
- SHA-256 checks confirm all 455 inventoried preexisting files remained unchanged, including previous results/reports, data, code, manifests and the two notebooks. The 4,192 reserved final IDs are unchanged.

## Caveats

This fixed pilot was already development-consumed; queries are synthetic and skills public. Metadata caps and limited labels affect the result. Candidate Recall@20 limits attainable recall; 80 queries lack at least one gold candidate, including 32 with none. Jev can label a plausible but unannotated skill relevant, which still counts wrong under the existing metric. A future final evaluation needs a separately approved, prespecified protocol. Vector→Jev was not run.

## Audit

```bash
.venv/bin/python scripts/audit_jev.py results/jev/f2dc6fc3f49ca24c4327
```

This reads saved results and makes no network request. See `evidence/jev/audit-f2dc6fc3f49ca24c4327-200.json` for machine-readable checks.
