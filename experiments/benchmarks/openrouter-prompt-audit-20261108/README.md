# OpenRouter prompt-result audit

Offline re-analysis of the nine-model run committed at
`3fb7e66a1a219bc43d7c2057dca3d1c70a3c6c45`.

- [Publication validation](VALIDATION.md): 41 tests, input preservation, reproducibility and privacy checks.
- [Audited report](results/REPORT.md): findings, definitions, limitations, costs and inference scope.
- [Machine-readable audit](results/audit.json): frozen input hashes, independent counts and checks.
- [Corrected metrics JSON](results/metrics.json) and [complete metric table](results/summary.csv): K=1,3,5,10, all model/size cells, three explicit scoring views and clustered intervals.
- [Common-nine-model panel](results/common_panel_summary.csv) and [paired model contrasts](results/paired_differences.csv) and [paired size contrasts](results/size_differences.csv): comparable cells with denominators.
- [Per-query grid](results/per_query.jsonl): every planned case, including NA/unattempted and preflight exclusions.
- [Figures](results/figures/README.md): presentation-ready 16:9 figures and caveats.
- [Conditional offline retrieval comparison](../openrouter-matched-retrieval-20261108/README.md): separate reconstructed-input experiment, not verified original-prompt equivalence.

## Reproduce without provider calls

Python 3.10+ and NumPy are sufficient for the numeric audit. Matplotlib is needed only for figures.

```sh
python experiments/benchmarks/openrouter-prompt-audit-20261108/audit.py
python -m unittest discover -s experiments/benchmarks/openrouter-prompt-audit-20261108/tests -v
python experiments/benchmarks/openrouter-prompt-audit-20261108/visualizations/plot_audit.py
```

The audit imports no provider SDK and has no network or inference operation.
Do not run the historical paid benchmark runner as part of reproduction.
Defaults are 5,000 query-cluster bootstrap draws with seed 5150. Identical
inputs and dependencies produce identical numeric artifacts. An optional
`--output-dir` writes to another directory; `--bootstrap-replicates` explicitly
changes interval simulation and is recorded in audit.json. Original inputs
must match `source/input_manifest.json` or the audit fails before writing.

## What is and is not recoverable

The CSV has 1,224 unique rows: 1,184 completed responses and 40 preflight
context exclusions. The 1,260-cell planned grid additionally has 36 absent
Claude cases. There are only 20 independent questions, repeated across sizes.
Seventy-two responses report invalid entries without their original positions;
raw responses and the original parser module are missing. The original
`details.csv`, `metrics.json`, configuration, query list and original figures
are not modified. Historical REPORT.md bytes are preserved under `source/`.

The primary descriptive view excludes reported-invalid rows. The full
stored-ranking view is an explicitly labeled parser-filtered sensitivity.
Neither is a recovered raw-response ranking. Lower-bound sensitivity inserts
known invalid entries at the front, under that conditional model only.
All metric arithmetic on the saved rank lists reproduces the historical CSV.
The audit corrects reporting, denominators, definitions, uncertainty and
unsupported claims, rather than manufacturing new raw evidence.

Estimated costs sum to US$35.140929. Actual billed cost, cache hit rates,
reasoning-token accounting and omitted retry charges cannot be verified.
See the report before interpreting any comparison or value claim.
