# Independent final retrieval and supplementary validation

**Passed** at immutable commit
[`0f4e9818b6cefb94e09da30ad76b9ff5ac97999c`](https://github.com/speccy88/mia5150/commit/0f4e9818b6cefb94e09da30ad76b9ff5ac97999c).
The attached standalone streaming verifier imports no experiment metric code and
performs no inference or search reexecution.

## Verified

- All 52,704 planned query/seed/size cases, with no duplicates or missing cells:
  4,392 distinct official requests × three seeds × four sizes.
- All catalog membership hashes, gold preservation, strict top-five IDs and
  case identifiers reconstructed independently from the pinned public inputs.
- All 158,112 ranker Recall@5, MRR@5, complete-set coverage, hit and denominator
  calculations recomputed against 7,187 positive qrels.
- All 36 macro-recall, positive-pair micro-recall, MRR and recorded search-time
  means reproduced to 1e-12.
- All 2,400 overlapping primary BM25/vector/hybrid top-five rankings and retrieval
  metrics match exactly: 800 per method, zero differences.
- All 26,352 additional-seed full-catalog rankings equal seed 42. This is expected:
  N=6,006 contains the same entire catalog regardless of seed, not new independent
  library samples.
- Compressed bytes match the committed Git blob, and the decompressed bytes match
  the published uncompressed hash.

At N=6,006 the full-query macro Recall@5 values are **47.73% BM25, 51.73% vector,
58.70% hybrid**. These are retrieval-coverage outcomes, not final selector accuracy.
Primary and local model/seed selection results remain unchanged.

## Exact evidence

Raw Library file: `libfile_1c718d66ccc88191a5519e6cb6a93a4e`,
`MIA5150_Full4392_Retrieval_2026-10-24.jsonl.gz`.

|Input|Identity|
|---|---|
|Compressed bytes|15,054,930|
|Compressed SHA256|`a55bfecfa99508cebcc0d6c0238575c60bfbd919d46c2e69654f66dcd0b53a35`|
|Committed Git blob|`268b1281e2851a811aab7be4cdcc32d363c88536`|
|Uncompressed bytes|91,864,324|
|Uncompressed SHA256|`dcf3630cf9d8fade4375e335c43aaf070ee507ce0482579276efd8df23ca60e1`|
|Recorded execution source|`2694a77fdde84cf6b35e2b91c22621050810ecb971d60eade7f959dc5f164de9`|
|Verifier source SHA256|`bc13f03e4505af2517cf347b418f6a750b7a8d4dee752bdc9dcf42b464d29355`|

The shared primary inputs are unchanged:
- Primary raw Library `libfile_cf2e5ab47a548191b135ef9964ee2d5d`, SHA256
  `6b77f74019790a27243ca2e243aa6626acddbe8070c69930658192de8b4302f2`.
- Primary evidence ZIP Library `libfile_311f82f23de081919eafbea11cdeb556`, SHA256
  `5f8288adf1312bc0653fe794b027e72f1cc3d6f839eee5ee3e5aed0294f8af01`.
- `report/full-retrieval.csv` fetched from the immutable commit above, Git blob
  `13b670e0239b826fb6ae35077ab28288a473cb24`.

## Reproduction

Python 3.12.14, standard library only. Extract the primary evidence ZIP beneath
`PRIMARY/evidence_extract`; put the primary raw JSONL beneath `PRIMARY/raw_library`.
Save the full-retrieval gzip and the committed summary CSV locally, then run:

```sh
python independent_full_retrieval_audit.py \
  --raw MIA5150_Full4392_Retrieval_2026-10-24.jsonl.gz \
  --primary-root PRIMARY \
  --report full-retrieval.csv \
  --output independent_full_retrieval_validation.json
```

The ZIP contains both standalone verifiers, their two passed results and this
README. The large raw inputs are deliberately excluded.

## Supplementary tables

The previously completed supplementary check is also saved as a standalone script:
`independent_supplementary_audit.py`, source SHA256
`e97b023005c83984bbe6f2fa89abaef3c60c82d9455f937d2024a446a58f8b4c`.
Its receipt is `supplementary_table_validation.json`.

It reproduces 40 seed-averaged bootstrap cells, 40 paired-model bootstrap cells and
20 disjoint primary first-failure count partitions from the already independently
validated raw local cases. Nominal descriptive intervals are not treated as
multiple-comparison-adjusted significance or causal model-size effects.

Use the same primary inputs and the completed-local packet, Library
`libfile_3b9b3352b2308191af4015a0cd0b2c63`, SHA256
`12c9d3c35b135d5bbad4d7100205320aff20cc5601408d78c2c2e7c790b6fc2a`.
Extract the latter under `SENSITIVITY`. Save the three committed CSVs from
`report/seed-query-bootstrap.csv`, `report/model-paired-bootstrap.csv` and
`report/error-pipeline.csv` beneath `REPORT`. Python 3.12.14 / NumPy 2.3.5:

```sh
python independent_supplementary_audit.py \
  --primary-root PRIMARY --packet-root SENSITIVITY --report REPORT \
  --output supplementary_table_validation.json
```

## Acceptance and limits

The full-benchmark breadth gap is closed for this pinned retrieval representation
and these three rankers. Together with the prior independent 5,680-case local
selector audit, the planned local evidence is reproducible and internally consistent.
No correction to a headline score is required by this review.

This is **not all research gaps closed**. Requests were already exposed; small
catalogs are gold-conditioned; positive-only qrels do not establish no-skill
abstention calibration or complete semantic relevance. Human-authored/adjudicated
requests, agreement, genuinely independent evaluation, native activation and actual
task execution remain separate evidence requirements. Hosted execution is still
reported as zero calls and credential-blocked, not a failed or successful experiment.

Search timings are checked as recorded, not remeasured. The full pass uses a
versioned sparse BM25 implementation and offline batched query embeddings. Do not
pool these search-only times with the primary fresh embedding/retrieval/selection
latency, or use the 52,704 repeated conditions as the number of independent queries.
