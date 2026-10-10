# Integration validation — 2 October 2026

These checks were performed when importing the two user-provided archives.
They validate retained evidence and code; no model inference, model download,
paid request, or new benchmark measurement was performed.

| Check | Observed result |
| --- | --- |
| Both ZIP CRC checks | Passed |
| Original code manifest | All 34 entries passed size and SHA-256 checks |
| Original results manifest | All 160 entries passed size and SHA-256 checks |
| Full local recorded-run audit | Passed for 14 completed stages, 3,712 result/timing rows |
| Project offline pytest suite | 73 passed |
| Standard-library evidence verifier | Passed: 32 queries, 6,006 skills, 53 qrels, 1,792 quality rows, 56 summaries, 14 full-catalog summaries, 896 timing observations |
| Source selection | Executed core unchanged; newer final report sources selected explicitly |
| Documentation links | All published Markdown relative links resolve |
| Credential-pattern scan | No matches in selected files or extracted PDF text |

The verifier independently recomputes nine relevance metrics and checks the
exact grids, catalog membership, shared native shortlists, aggregate means,
matched latency percentiles/means/QPS, and current file hashes. The additional
36 verifier tests include corrupted IDs, metrics, pools, timing observations,
and unsafe checksum paths. The other 37 tests are the supplied synthetic
analysis/reporting and weight-free adapter checks.

The full local audit ran on the untouched extracted results package using
`tests/audit_skillret_recovery.py`. It also checks native pair scores,
shortlist ordering, input-fitting audits, run configurations and scorer cache
fingerprints. Its full inputs remain in the local archive extraction; the
published compact snapshot intentionally has a narrower verification scope.
Raw logs and validation receipts are saved locally in
`local-experiments/evidence/cpu-benchmark-integration-2026-10-02/`.

Local checks used the existing macOS sandbox environment: Python 3.12.6,
pytest 9.1.1, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6, and nbformat 5.11.1.
Those are integration-test versions, separate from the recorded Linux CPU
inference environment and its model-specific dependency pins. No clean-room
installation or macOS inference portability is claimed.

Run the same offline checks from this folder:

```sh
python3 verify_recorded.py
shasum -a 256 -c SHA256SUMS
python -m pytest -q
```

The destination copy is checked with the same pytest suite, verifier, file
manifest and credential scan before publication.
