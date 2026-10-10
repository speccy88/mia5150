# Dense20 + novel BM25 at budget30: existing experiment verified

The exact requested candidate comparison was already completed over all **4,392 historically exposed public queries**. It was not rerun as a new experiment. A bounded, monitored CPU-only audit verified its original source-cache hashes, saved-pool hashes, construction, metrics and positive-preservation guarantee. The audit exited normally; no GPU model, API, download or training was launched.

Both arms contain exactly30unique candidates for every query:

- Control: cached vector SkillRet-Embedding0.6B top30.
- Treatment: preserve the same vector top20 in order, then append the first10novel BM25 candidates, scanning at most the cached top100.

Construction is label-independent. The audit first reconstructed and checked every saved pool, then read qrels only to calculate diagnostic coverage and ceilings. No gold candidates were inserted, no pool was repaired using labels, and original raw artifacts were preserved.

| Metric | Dense30 | Dense20 + novel BM25 10 | Difference |
|---|---:|---:|---:|
| Macro candidate recall@30 | 0.873330 | 0.874962 | +0.001632 |
| Oracle Recall@5 | 0.873330 | 0.874962 | +0.001632 |
| Oracle Recall@10 | 0.873330 | 0.874962 | +0.001632 |
| Relevant pairs covered / 7,187 | 6,114 | 6,167 | +53 |
| Completeness@30 | 0.779372 | 0.787341 | +0.007969 |
| Oracle NDCG@10 | 0.892345 | 0.892718 | +0.000373 |

The original comparison recorded **175 improving queries, 4,094 ties and 123 regressions**. The treatment gained183positive pairs but lost130relative to dense30. Those negative outcomes remain in the evidence; a positive aggregate does not mean universal improvement.

## Which positives are preserved?

The prefix guarantee holds for every query: treatment first20equals cached dense first20. Therefore **all5,923relevant query-skill pairs present in dense20 are retained, with zero losses**. The treatment adds244relevant pairs beyond dense20.

The guarantee does not extend to dense ranks21–30. The treatment exchanges those slots for novel lexical candidates and loses130pairs relative to dense30. Preserving every known positive from dense30 by consulting qrels would make candidate selection label-dependent and contaminate the comparison. We did not do that.

Within these saved pools, an ideal ranker can place every retrieved positive within its top5, so oracle Recall@5 and Recall@10 equal the pool recall. These are **label-based ceilings**, not achieved local reranker scores. The small mean ceiling increase does not demonstrate a new retrieval-and-reranking improvement.

## Decision and remaining scope

No additional local reranker panel was launched. The exact full-population coverage test already exists, and its oracle NDCG@10 advantage is only0.000373with123regressing queries. Another small convenient panel would add weak evidence and risk selective interpretation. The conditional allowance for one bounded reranker comparison was left unused.

This is exposed public-test analysis, not untouched confirmation. No unused confirmation cohort was opened. The original historical200-query Jev, LoRA validation, smaller hosted/local screens and routing pilot remain separate datasets and claims.

EG2 text-only270M and Jina v5 text-nano239M remain a later shortlist only: neither was installed or downloaded. No activation-format or GPU compatibility settings were changed. WSL remains6GiB RAM,2processors,2GiB swap with dropCache reclaim. Existing live host/guest/GPU/no-active-swap guards were used unchanged.

## Evidence and reproducibility

- Existing exact comparison: `round2/candidate-budget30-v1/{freeze,pools,evaluation,per-query}.json`.
- New audit: `round2/dense20-preservation-audit-v1/audit.json` and `per-query.csv`.
- New guarded run: `runs/dense20-preservation-audit-v1/`; exit code0.
- Audit implementation: `round2/audit_dense20_preservation.py`. Its one-shot output directory protects the saved receipt from overwrite.
- Original protocol SHA256: `c601df43beee7295cd6518c285f93ca4a41f1041f26292d823758799c9ae87b8`.

The audit receipt contains SHA256 values for all original artifacts and both source ranking caches. It verifies every one of the4,392pool constructions and retains per-query counts and oracle recalls. The English report and raw evidence are ready for the existing Notion handoff; no upload or publication occurred here.

Compute is released. The GPU was checked at0MiB used and0% utilization after the audit. The host monitor was stopped. No new spend or reservation occurred; previous shared commitments remainUSD1.97/USD4.
