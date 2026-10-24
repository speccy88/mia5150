# Independent completed-local sensitivity validation

**Result: passed.** Read-only verification of saved outputs at commit
[`e8690e35de1821980f82c20ce13318d4e64bfcf2`](https://github.com/speccy88/mia5150/commit/e8690e35de1821980f82c20ce13318d4e64bfcf2).
No model inference was performed and no repository file was changed by this audit.

The verifier imports no experiment metric implementation. It recomputes set scores,
strict output alias mappings, prompt hashes, catalog membership, top-five retrieval
metrics, routing gold retention, official qrel equality and generated label unions.
It checks the exact planned case grid rather than treating the ledger row count as
sample size. It also reproduces the exact/macro/retrieval, token-mean, paired-latency
and latency-quantile values in 100 summary cells, and all 80 seed sensitivity cells.
Primary paired bootstrap intervals were separately reproduced by
`independent_raw_audit.py`; they are not reimplemented in this second verifier.

## Findings

- All **5,680 condition-cases** are reconciled: 5,080 small-model and 600 4B cases.
- There are **200 distinct exposed official requests**, a 24-request subset reused
  for sensitivity, and 12 separate generated silver probes. Repeats are not new
  independent requests.
- 5,662 prompt hashes, 4,985 valid alias outputs, 4,541 retrieval metrics, 1,133
  route-retention checks, 567 context preflight exclusions and 109 parseable invalid
  output cardinalities pass their checks.
- 4B statuses: 538 valid, 60 overflow, one invalid selection, one invalid/empty
  domain. Small-model statuses are preserved in the JSON, including the single
  generated persistence-unavailable outcome. It is not reclassified as success.
- Primary findings are unchanged. No new scoring blocker was found in the final
  local summaries or the revised closure matrix.

This verifies recorded timing arithmetic, not latency measurements performed by the
auditor. It does not establish human label validity, a fresh holdout, natural-catalog
selection at small sizes, native skill activation, task execution, or hosted results.
Full 4,392-query retrieval remains a separate stage.

## Exact inputs

1. Primary raw: `MIA5150_Primary200_Raw_2026-10-24.jsonl`, Library
   `libfile_cf2e5ab47a548191b135ef9964ee2d5d`, 11,535,421 bytes,
   SHA256 `6b77f74019790a27243ca2e243aa6626acddbe8070c69930658192de8b4302f2`.
   Git blob `61d1d606dc46da62efedd884af5cb174f3485c62` matches committed bytes.
2. Primary catalogs/public inputs: `MIA5150_Primary_Evidence_2026-10-24.zip`, Library
   `libfile_311f82f23de081919eafbea11cdeb556`, 22,648,933 bytes,
   SHA256 `5f8288adf1312bc0653fe794b027e72f1cc3d6f839eee5ee3e5aed0294f8af01`.
3. Final non-primary packet: `MIA5150_Completed_Local_Sensitivity_2026-10-24.zip`,
   Library `libfile_3b9b3352b2308191af4015a0cd0b2c63`, 1,185,712 bytes,
   SHA256 `12c9d3c35b135d5bbad4d7100205320aff20cc5601408d78c2c2e7c790b6fc2a`.
   All 34 declared manifest entries passed byte-count/SHA256 checks.
   - Small non-primary canonical export: 1,080 cases, 3,140,289 bytes,
     SHA256 `51cc8227aa7f1b7357db0de81402781697a8d64bb52568bc8bcf97f3e8021b1f`.
     The complete repeated small-model ledger was not independently byte-compared;
     this exact Library export is the independent input for these 1,080 cases.
   - 4B canonical export: 600 cases, 1,697,925 bytes,
     SHA256 `93b484be6c3a10bdf5cd3d3e1366ab107ec7476bc5349a733ed2233de5e691ff`.
     Git blob `32f84963175943fdbf1e29bdb3a8eee330ad8672` matches the committed
     `results/qwen35_4b_q4-cases.jsonl` byte for byte.

The packet's selection/seed CSVs, status, validation and report text were compared
with the GitHub connector's immutable committed files, normalizing only line endings.
The public skills, queries and qrels byte hashes are checked against the frozen
protocol. Evaluator labels, primary panel and protocol are unchanged from the primary
packet. Generated labels are checked as unions of their constituents' positive qrels,
not as human judgments.

## Reproduction

Python 3.12.14 and NumPy 2.3.5 were used. Standard-library modules plus NumPy are the
only dependencies. Extract the primary evidence ZIP beneath `PRIMARY/evidence_extract`;
place the primary raw JSONL beneath `PRIMARY/raw_library`. Extract the final sensitivity
packet beneath `SENSITIVITY`. The original ZIP already contains the primary raw file
under `raw/`, which may be copied into the required raw-library location.

```sh
python independent_sensitivity_audit.py \
  --primary-root PRIMARY \
  --packet-root SENSITIVITY \
  --output independent_sensitivity_validation.json
```

Verifier source SHA256:
`602a5220231dbb3cf34130928af3dfbf3a9e378e2c2811fce466155cdc2fb118`.

Only this README, the standalone verifier and its result are included in this
sanitized audit ZIP. No raw records, credentials, original proposal identities,
or private input documents are packaged.
