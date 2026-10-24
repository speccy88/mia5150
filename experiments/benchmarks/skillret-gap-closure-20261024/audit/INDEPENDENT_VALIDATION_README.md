# Independent primary-result validation

Read-only audit completed 24 October 2026. No inference, model execution, paid requests, new annotations or repository modification was performed by the reviewer.

## Result

PASSED: all 4,000 planned primary cases on 200 unique exposed queries. Independently recomputed every selected-set metric, 3,197 recorded retrieval rows, 797 routing-retention values, 3,984 prompt hashes, all 3,496 valid raw alias outputs, 400 context preflights, 87 invalid-list cardinalities, all 20 weighted-exact table cells, eight paired-bootstrap contrasts/intervals, and all 20 token/paired-timing summary cells. Arithmetic agreement tolerance is 1e-12. Recorded timing values were checked for aggregation, not remeasured.

## Exact target

- Repository: https://github.com/speccy88/mia5150
- Commit: `ed12adad509ea0e5a082a906c4974c13ab5b3719`.
- Study: `experiments/benchmarks/skillret-gap-closure-20261024/`.
- Primary raw repository file: `results/primary200-canonical.jsonl`.
- Primary Git blob: `61d1d606dc46da62efedd884af5cb174f3485c62`.
- Raw SHA-256: `6b77f74019790a27243ca2e243aa6626acddbe8070c69930658192de8b4302f2`.
- Protocol identity: `44acc3d80d6a904da7f9d554d5de9995617c629073c64c5b1653b5ab9ddf916f`.
- Public input source: `experiments/scaling/skillret-seed5150-scaling/scaling5150/provenance-audit/public-inputs/skills-metadata.jsonl`.

The reviewer obtained exact primary bytes via Library because the GitHub connector could not read an 11.5 MB text file. The independently computed Git blob SHA proves those bytes equal the target committed raw primary file. Study catalog/label files and metadata were extracted from the supplied evidence ZIP. Two report CSVs were read through the GitHub connector and saved with normalized LF line endings; hashes below describe those local audit bytes, not an assertion of original CSV byte identity.

## Reproduce without model assets

Tested with Python 3.12.14 and NumPy 2.3.5. The script imports no experiment implementation or metric module. Use a checkout or extracted artifacts from the pinned commit, with large raw files hydrated. From repository root, set `STUDY=experiments/benchmarks/skillret-gap-closure-20261024`, then run:

```sh
python path/to/independent_raw_audit.py \
  --raw "$STUDY/results/primary200-canonical.jsonl" \
  --study "$STUDY" \
  --public-inputs experiments/scaling/skillret-seed5150-scaling/scaling5150/provenance-audit/public-inputs \
  --report "$STUDY/report" \
  --output independent_raw_validation.json
```

The script intentionally pins the primary raw Git blob. A later report refresh can be checked against the same immutable primary rows, but altered scientific inputs require a separately versioned audit. Its timing aggregation uses the primary panel's dominant source hash, which matched the reviewed report; if a later campaign changes the report's source eligibility, review that change rather than weakening assertions.

## Limits

This validates the primary result, not the complete append-only attempt history, 4B/extra-seed/generated/full-4,392-query stages, hosted models, a fresh held-out panel, human label quality, or native skill execution. The exposed-query and incomplete-qrel limitations remain. One unavailable generated-case persistence result is outside this primary audit. Final campaign status still requires the implementation owner's terminal receipts.

## Audited input byte hashes

```json
{
  "raw_library/MIA5150_Primary200_Raw_2026-10-24.jsonl": {
    "sha256": "6b77f74019790a27243ca2e243aa6626acddbe8070c69930658192de8b4302f2",
    "bytes": 11535421
  },
  "evidence_extract/study/catalogs.json.gz": {
    "sha256": "027f8c72faa54c305181b2d2e0edf05235a04424946b6fd3b90b649cd94610cf",
    "bytes": 3422016
  },
  "evidence_extract/study/catalog-document-ids.json": {
    "sha256": "f20e8af2911655022e7830eed8e6adb6f6164f9a76dbbe7357523e71ca8a3550",
    "bytes": 258261
  },
  "evidence_extract/study/evaluator-labels.json": {
    "sha256": "fb495516ab851317d9be039cbd51a9bd30a93980951bfd5ee30fca4462944ec6",
    "bytes": 19490
  },
  "evidence_extract/study/panel.json": {
    "sha256": "5075a02a689917b10f0ef18bc7ef291de9fc1b4ea780062a822ac83cb3b51adb",
    "bytes": 250524
  },
  "evidence_extract/study/protocol.json": {
    "sha256": "f23359c2a464c58fce285d0c52ae2e30eddb168785e031afe5411dd18322fcc2",
    "bytes": 4011
  },
  "evidence_extract/public-inputs/skills-metadata.jsonl": {
    "sha256": "74819f4f683cb4dfcfc34085f6caa610202e78c49e8e32a2f5045a825addeb2f",
    "bytes": 2079130
  },
  "report/selection-summary.csv": {
    "sha256": "2327606d242a2478771aedfd948d55143443f9b1980fe518c51281a5273fe269",
    "bytes": 41430
  },
  "report/paired-primary.csv": {
    "sha256": "0c53ee0b6c92e092b2233af62717b73caa49556f80ae09df1c9ba49cd48b2c85",
    "bytes": 2985
  }
}
```
