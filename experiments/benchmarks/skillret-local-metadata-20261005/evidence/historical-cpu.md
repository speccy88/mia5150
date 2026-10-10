# SkillRet CPU benchmark

This folder contains the CPU-only SkillRet tests supplied in
`SkillRet_Benchmark_Code.zip` and `SkillRet_CPU_Benchmark_Results.zip`, imported
on 2 October 2026. It compares lexical and embedding retrieval with four
binary relevance adapters for finding useful skills from short discovery
headers. It includes the experiment code, tests, pinned source metadata, and
compact recorded evidence. Integration checks are offline; the numbers below
come from the supplied run, with no new model inference.

## Recorded experiment

The dataset is `ThakiCloud/SKILLRET`, revision
`6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`: 6,006 test skills,
4,392 source test queries, and 7,187 positive relevance judgments. Each
searchable document is `name + "\n" + description[:1024]`; full `SKILL.md`
bodies are excluded. This differs from the official full-document baseline
and the other studies in this repository.

The primary comparison uses the same **32 hash-selected test queries** for
**14 pipelines** at catalog sizes **10, 100, 1,000, and 6,006**: 56 conditions
and 1,792 per-query rows. The smaller catalogs are nested, query-specific,
gold-preserving samples. They measure controlled distractor scaling; the
6,006-skill catalog is shared across queries.

The pipelines are:

- BM25 lexical retrieval.
- BGE-small, multilingual E5-small, and EmbeddingGemma INT8, each with dense
  retrieval, BM25/dense reciprocal-rank fusion, and a category-centroid routing
  ablation (`domain_vector`). Routing selects one published skill `major`
  category from the query embedding, then searches that category.
- Julia-1, GLiNER2.5-small, Verdict ONNX, and Laya multilingual, each scoring
  binary relevance on the same BGE/BM25 hybrid top-20 shortlist. These are
  pairwise relevance adapters; native listwise choice routing was not tested.

This gives seven neural model families plus BM25. There is no training,
fine-tuning, paid API selector, or skill execution in this study.

## Recorded full-catalog results

These selected rows use all 6,006 skills and the same 32 quality queries.
The [full table](reference_run/report/SkillRet_Benchmark_Matched32_FullCatalog.csv)
also contains the dense and category-routing ablations. Rates are macro means;
MRR@5 is truncated at five.

| Pipeline | MRR@5 | Recall@5 | Median latency (ms) |
| --- | ---: | ---: | ---: |
| BM25 | 0.7396 | 0.6042 | 6.39 |
| BGE hybrid | 0.8385 | 0.7083 | 99.74 |
| E5 hybrid | 0.8750 | 0.7396 | 106.21 |
| EmbeddingGemma INT8 hybrid | 0.8828 | 0.7708 | 509.35 |
| Julia rerank | 0.1026 | 0.1719 | 4,335.37 |
| GLiNER rerank | 0.2042 | 0.4219 | 7,357.38 |
| Verdict rerank | 0.1714 | 0.1979 | 17,349.07 |
| Laya rerank | 0.1193 | 0.1927 | 14,682.03 |

Warm latency uses the same **32 queries × two repeats**, at full catalog only.
It includes fresh query encoding, retrieval, and adapter scoring where used;
model loading and corpus/index preparation are excluded. Smaller-catalog
timings were not measured. Serial QPS is reciprocal mean latency, and process
peak RSS includes the whole pipeline rather than model weights alone.

Run receipts identify an Intel Xeon Platinum 8573C, nine visible CPUs,
affinity to CPUs 0–3, four numerical threads, Linux x86-64, and Python 3.12.14.
Dedicated CPU quota and an enforced RAM limit were not established.

The four tested adapters reduced ranking quality relative to their BGE hybrid
shortlist in this configuration. This does not establish their quality on
other prompts, native tasks, or listwise routing. The cohort is small: the
recorded paired 95% bootstrap interval for BGE hybrid minus BM25 MRR@5 spans
zero, [-0.015625, 0.234505]. Rankings here are descriptive, not a universal
model ranking. The [original four-page report](reference_run/report/SkillRet_CPU_Benchmark.pdf)
contains further analysis.

## Evidence boundaries

Retrieval ran on 64 queries. Julia and GLiNER also completed a 48-query
extension. The archive records a telemetry-related stop during Verdict48;
Laya48 did not complete. Those larger runs are supplements, and the primary
table remains the complete shared 32-query comparison. See the saved
[finalization record](reference_run/protocol/report_finalization.json).

Earlier 4,392-query aggregate observations survive separately in
[retained_observations.json](docs/retained_observations.json). Their original
raw records are unavailable, so they cannot be rescored or paired with this
32-query run. The supplied analysis code preserves that distinction.

Relevance uses the released positive qrels; missing judgments are scored as
non-relevant. Discovery-header retrieval is not downstream agent execution
success. Query sampling intervals do not cover seed, hardware, or model
configuration variation.

`setup_privacy.sh` is a prospective environment setting supplied after the
measurements. It was not part of the measured configuration and was not
runtime-tested after the recorded stop. The
[privacy addendum](docs/PRIVACY_ADDENDUM.md) retains that provenance.

## Folder contents and provenance

- `src/`: dataset preparation, pinned model download, CPU retrieval, relevance
  adapters, and analysis/report builders. Executed core sources are unchanged.
- `tests/`: synthetic metric, cohort, reporting, and offline evidence checks;
  `src/test_decision_adapters.py` provides weight-free adapter contracts.
- `reference_run/`: original aggregate/per-query CSVs, the PDF and receipts,
  matched timing observations, frozen IDs/qrels/candidate memberships, and
  compact metadata. Query text, skill text, model weights, vectors, and logs
  are omitted from the published snapshot.
- `provenance/`: both original archive manifests and
  [SOURCE_SELECTION.json](provenance/SOURCE_SELECTION.json), which records
  archive hashes, copied file hashes, and the filtering of ID-only evidence.
  Final `report_fresh.py` and `build_compact_pdf.py` come from the newer results
  archive; the other code comes from the delivered code archive.
- `SHA256SUMS`: hashes of the current published files. The original manifests
  describe their complete input archives, not this selected publication.

[ORIGINAL_CODE_README.md](ORIGINAL_CODE_README.md) preserves the code-only
archive's documentation. This README describes the integrated snapshot;
documentation-only link corrections are recorded in the source selection map.
Recorded CSVs retain their original CRLF bytes through `.gitattributes`.

The original ZIPs and their complete extractions are preserved locally under
`local-experiments/artifacts/skillret-cpu-benchmark-2026-10-02/` in the MIA5150
workspace. They are not committed. The complete recorded run supports the
additional pair-score/source-fingerprint audit in
`tests/audit_skillret_recovery.py`; that audit needs the full local extraction.

## Verify without models or downloads

From this folder, the standard-library verifier checks file hashes, exact
query/pipeline/catalog coverage, all nine metrics against ranks and qrels,
candidate membership, shared reranker shortlists, aggregate means, and
32 × two-repeat timing summaries:

```sh
python3 verify_recorded.py
shasum -a 256 -c SHA256SUMS
```

The offline test suite also needs NumPy, SciPy, pandas, nbformat, and pytest.
`requirements-tests.txt` records the direct versions used for the local
integration checks; it is separate from the original Linux model environments.

```sh
python3 -m venv .venv-tests
.venv-tests/bin/python -m pip install -r requirements-tests.txt
.venv-tests/bin/python -m pytest -q
```

In the MIA5150 workspace, use the existing sandbox environment without an
installation, from `local-experiments/skillret-cpu-benchmark/`:

```sh
ALLOW_PAID_RUNS=0 ALLOW_PAID_NOTEBOOK_RUNS=0 HF_HUB_OFFLINE=1 \
  ../.venv/bin/python -m pytest -q
```

These checks validate retained evidence and code contracts. They do not claim
a clean-room model installation or end-to-end inference rerun.

## Reproduce a new CPU run

Use a fresh copy of the code for each new run to preserve existing results.
The model runners target Linux x86-64 and use Linux CPU affinity and
`/proc/cpuinfo`; macOS/Windows end-to-end inference is unverified. Original
model pins remain in `requirements-main.txt`, `requirements-gliner.txt`, and
`requirements-julia.txt`. GLiNER and Julia require separate environments.
See [MODEL_DOWNLOADS.md](MODEL_DOWNLOADS.md) for exact revisions and installation.

After creating those environments, start with the pinned dataset and BM25:

```sh
. ./setup_privacy.sh
.venv/bin/python src/prepare_dataset.py
.venv/bin/python src/run_retrieval.py bm25
```

The optional neural runs and analysis steps are documented in
[FULL_REPRODUCTION.md](docs/FULL_REPRODUCTION.md). Complete BGE retrieval before
the relevance adapters because it writes their shared shortlist. Run stages
serially and choose a runtime budget before extending the query cohort.

`src/controller.py` is retained run history: it has a fixed 2 October 2026
deadline and external checkpoint gates, so it is not the default launcher.
HTML rebuilding needs the Data Analytics renderer (`--plugin-root`);
PDF rebuilding needs ReportLab, pypdf, and the original Linux font paths.
Those presentation dependencies are optional. The committed CSVs and PDF
can be read directly.

