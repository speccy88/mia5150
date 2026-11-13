# MetaEncoder v1 / SkillRet

**Measured study: complete primary retrieval, partial secondary experiments.** All 6,006 skills and 4,392 queries were processed on RunPod. Binary and ordinal decisions completed the frozen 200-query top-10 pilot, but not full-query or complete top-20/50 evaluation. Final diagnostics were generated on RunPod at 2026-11-13 04:25:33 UTC. Delivery and resource lifecycle evidence are recorded separately; this report does not equate partial experimental scope with the entire requested benchmark being complete.

## Primary retrieval

| Configuration | Queries | Catalog | NDCG@10 | Recall@10 | Completeness@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| MetaEncoder v1, 6,656-dimensional BF16 inference / FP32 output, content-preserving chunk aggregation | 4,392 | 6,006 | 0.720593 | 0.791249 | 0.656421 | 0.792056 |
| Historical SkillRet-Embedding-0.6B full text | 4,392 | 6,006 | 0.810990 | 0.876442 | 0.788024 | 0.856349 |

MetaEncoder's measured primary retrieval scores are lower than the supplied historical baseline. This does not establish that every MetaEncoder capability is inferior: native API equivalence, dimension reductions and decision reranking are judged below using their own measured scope. Historical local latency is not a controlled RunPod latency comparator.

## Conditions and caveats

- Both use the same pinned public test corpus and canonical full-text convention. This is an exposed public benchmark, not a blind unseen test.
- MetaEncoder document inputs exceeding the trained 16,384-token boundary use deterministic content-preserving chunks and normalized mean aggregation. Nine documents exceed that boundary; no query does. A separate fixed-first-chunk comparison isolates this protocol change.
- Prefix-truncated embedding dimensions are exploratory and not a documented native feature.
- Retrieval ranks by cosine similarity with a fixed original-catalog-row tie rule. BF16 inference returns FP32 normalized arrays.
- A subgroup evaluation implementation error stopped the first scoring pass after embedding completion. It was corrected by filtering ranking dictionaries to each subgroup’s query IDs while retaining the full skill catalog, then scoring was rerun from cached arrays. No expensive embedding inference needed repeating for this correction.
- Decision reranking cannot recover positives absent from candidate pools. Incomplete scoring is an operational shortfall, not evidence that the decision model regressed.

## All completed retrieval configurations

| Configuration | Queries | Dimension | NDCG@10 | Recall@10 | Completeness@10 | MRR@10 | Cached matrix multiply (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_256 | 4392 | 256 | 0.548590 | 0.624469 | 0.468807 | 0.618847 | 0.019397 |
| full_512 | 4392 | 512 | 0.635098 | 0.707688 | 0.551002 | 0.709480 | 0.028259 |
| full_1024 | 4392 | 1024 | 0.688011 | 0.758500 | 0.612933 | 0.762030 | 0.284812 |
| full_2048 | 4392 | 2048 | 0.707705 | 0.778271 | 0.640255 | 0.780271 | 0.523945 |
| full_4096 | 4392 | 4096 | 0.717491 | 0.787606 | 0.650956 | 0.789751 | 0.820637 |
| full_6656 | 4392 | 6656 | 0.720593 | 0.791249 | 0.656421 | 0.792056 | 1.339122 |
| truncated_6656 | 4392 | 6656 | 0.720690 | 0.791325 | 0.656421 | 0.792056 | 1.476223 |

Sources: [aggregate measurements](results/retrieval_summary.json), [full-dimension per-query metrics](results/full_6656_metrics.json), [top-100 rankings](results/rankings/README.md). All configurations use the same completed primary inference; dimensions below 6,656 do not incur additional inference. Times measure a single cached CPU NumPy matrix multiplication for all 4,392 queries against 6,006 skills, excluding ranking sort, model loading, query inference and network transfer. They are not warm request-latency estimates.

## Native retrieve

The native interface completed 20 queries against the full corpus. Sample NDCG@10=0.804087, Recall@10=0.858333, Completeness@10=0.750000 and MRR@10=0.885000. These are different-query-scope metrics, **not** improvements over the full 4,392-query manual result. Native calls include fresh query encoding, unlike cached matrix multiplication. Measured native latency p50=0.292786 s and p95=0.334097 s includes query encoding and search. Exact top-100 order agreement was 0%; top-100 set agreement was 35%. Batch-shape/BF16 numerics and different tie handling are plausible, **not proven**, explanations. None of the queries exceeded the trained sequence boundary, so query truncation is not an explanation in this run. Equivalence remains unresolved. See [diagnostics](results/native_equivalence_diagnostics.json).

## Runtime and validation

RunPod H100 PCIe 80 GB; allocated 16 vCPU and 251 GB RAM, Intel Xeon Platinum 8352Y. Actual package versions and device details are preserved in manifests. The 10 tests in `test_metrics` and `test_protocol` passed on RunPod; [log](results/tests.log). Embedding arrays were validated for expected shape, finite values, FP32 dtype and L2 normalization before export. Nine long documents were chunked, and no query exceeded the trained boundary.

## Decision experiments: matched completed pilot

The deterministic 200-query pilot was frozen before decision evaluation. All top-10 candidates were successfully scored for binary and ordinal modes. Binary and ordinal questions share one inference call per pair; their costs must not be added as independent calls.

| Configuration | Queries | Candidates/query | NDCG@10 | Recall@10 | Completeness@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| Matched cosine pilot | 200 | 10 | 0.707071 | 0.780000 | 0.635000 | 0.784242 |
| Binary relevance reranking | 200 | 10 | 0.661998 | 0.780000 | 0.635000 | 0.695988 |
| Ordinal usefulness reranking | 200 | 10 | 0.627428 | 0.780000 | 0.635000 | 0.650004 |

Neither decision mode improves ranking on the complete pilot. Paired 95% percentile bootstrap intervals for NDCG@10 change versus matched cosine are binary **[-0.083338, -0.011923]** and ordinal **[-0.118645, -0.044243]**, using 1,000 resamples and seed 42. These describe this pilot and frozen prompts, not every possible decision formulation. At cutoff 10, reranking the same ten items cannot change recall or completeness.

Binary classification on the 2,000 top-10 pairs has precision 0.175591, recall 0.952991 and F1 0.296543. This high-recall, low-precision behavior does not support treating a positive decision as a reliable relevance judgment. The ordinal 0–4 usefulness scale is a model output, not graded ground truth. Independent selection results are preserved in `v2_selection_*_metrics.json`; no selection threshold was optimized on held-out labels.

Combined binary-plus-ordinal pair latency on top-10 is p50 **0.816247 s**, p95 **2.772160 s**. Across all completed pairs it is p50 0.813960 s, p95 2.826201 s. These are pair-call latencies, not whole-query reranking latencies.

### Incomplete wider candidate pools

There are 3,762 successful pairs out of the intended 4,000 top-20 or 10,000 top-50 pairs. **No query has a fully scored top-20 or top-50 pool.** Both wider arms therefore reuse the same incomplete scored candidates; their identical measurements are not independent experiments.

| Partial scored-candidate output | Intended queries | Intended pool | NDCG@10 | Recall@10 | Completeness@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| Binary, incomplete | 200 | 20 / 50 | 0.649607 | 0.797500 | 0.665000 | 0.669232 |
| Ordinal, incomplete | 200 | 20 / 50 | 0.603104 | 0.790000 | 0.645000 | 0.605569 |

These retain the intended 200-query denominator but rank only successfully scored candidates. They measure partial execution output, **not a valid completed-pool improvement or regression**. The full 4,392-query decision experiments were not run. Candidate absence imposes a hard ceiling: full-corpus retrieval macro coverage is 0.844376 at 20 and 0.898983 at 50. A reranker cannot recover an absent relevant skill.

### Choice

Choice ran for four queries at each of 5, 10 and 20 candidates. At each size, three completed queries have one required skill (successful-only accuracy 0.666667) and one has two required skills (accuracy 1.0); no three-skill query completed. These tiny subsets are insufficient for general conclusions. `choice_v2_summary.json` retains intended subgroup denominators of 104, 69 and 27 and conditional-on-coverage accuracy. Unfinished cases remain zero in the intended-denominator measure. No full ranking or MRR was fabricated from one choice.

Sources: [matched diagnostics](results/decision_matched_diagnostics.json), [operational summary](results/decision_v2_summary.json), [choice](results/choice_v2_summary.json). Raw responses/checkpoints are retained in the final Drive evidence.

## Dimensions and calibration

The smallest exploratory dimension satisfying the frozen criterion of absolute NDCG@10 **and** Recall@10 differences at most 0.01 from 6,656 is **4,096**. This reduces storage/search width without new neural inference, but prefix truncation is not an officially supported native dimensionality feature. Full-size vectors are retained. Single cached search timings above are hardware/runtime-specific, not a robust repeated latency study.

Official scalar-temperature calibration used disjoint alternating query subsets of the frozen pilot: 100 fit queries and 100 held-out queries, with 1,881 observed candidate pairs in each. Temperature changed from 0.02 to 0.190546. No model weights were trained.

| Held-out calibration metric | Uncalibrated | Calibrated |
|---|---:|---:|
| ECE | 0.472919 | 0.438594 |
| Brier score | 0.376372 | 0.248985 |
| Negative log-likelihood | 1.307319 | 0.690904 |

Calibration improves these metrics but leaves substantial miscalibration. It is a secondary, labeled test-subset experiment, **not** an untouched 4,392-query calibrated test. Incomplete candidate collection limits representativeness. See [calibration](results/calibration_v2.json) and the reliability diagram.

## Historical comparison and uncertainty

The existing `skillret-embedding-jev-fusion/results/dense_top50.jsonl` rankings were rescored on RunPod using the same exact 4,392 IDs. Source SHA-256: `2941fb3aea15ce1d22bf8559ebfcd07476b55e9e948ab3cc889f812f892752b8`. No historical model was rerun.

The primary MetaEncoder-minus-baseline NDCG@10 difference is -0.090397, with paired 95% interval **[-0.097776, -0.083004]**; Recall@10 difference is -0.085193, interval **[-0.093997, -0.077185]**. RunPod computed 1,000 paired query resamples, seed 42, percentile intervals. The exact query-ID hash and implementation are in [paired_bootstrap.json](results/paired_bootstrap.json). This is a comparison under the declared model-specific document handling, not a controlled architecture-only causal study.

| Required skills | NDCG@10 | Recall@10 | Completeness@10 | MRR@10 |
|---|---:|---:|---:|---:|
| 1 | 0.787469 | 0.887074 | 0.887074 | 0.755800 |
| 2 | 0.676756 | 0.730769 | 0.513799 | 0.820642 |
| 3 | 0.594842 | 0.603785 | 0.195971 | 0.845199 |

Full aggregate cutoffs, per-query measurements and coverage definitions are in [full_6656_metrics.json](results/full_6656_metrics.json). No primary query is missing. Required-skill subgroup metrics were corrected by filtering rankings by subgroup query IDs, not by restricting the skill catalog.

## Runtime, cost, figures and delivery

[Performance diagnostics](results/performance_diagnostics.json) separate 2,960 individually encoded documents (encoding timer 2,215.144693 s) from 3,046 batch-amortized documents (855.936910 s). Query encoding for 4,392 queries took 311.846131 s. Warm single-document p50/p95 are 0.589648/1.758699 s; batch-amortized query p50/p95 are 0.068025/0.104461 s and are **throughput allocations, not single-request latency**. Recorded peak allocated GPU memory reaches 60,099,139,584 bytes. Download/startup/checkpoint time is not included in these encoding timers.

Eight RunPod-generated PNGs and CSV/JSON inputs are under `figures/`: capability NDCG, recall/completeness, dimensionality, decision changes, required-skill groups, latency/memory, reliability, and quality/compute. Read scope labels before comparing bars. Phase-compute estimates are not provider invoices; shared inference and overlapping decision pools must not be double-counted. A final provider bill and complete per-arm monetary attribution are unavailable in the measured result JSON.

Lossless vectors are delivered through [MEM/MIA5150/MetaEncoder-v1-SkillRet-6006-4392](https://drive.google.com/drive/folders/16zcgqtaJSVOAv46mWwf7zli1GOhdf-h4). See [DRIVE_ARTIFACTS.md](DRIVE_ARTIFACTS.md) for actual verified files, hashes, final evidence and delivery status. GPU compute was verified stopped at approximately 04:31 UTC on November 13; final storage termination and publication confirmation must be verified separately, not inferred from stopping compute.

Remaining scientific limitations: incomplete larger/full-query decisions, tiny choice sample, unresolved native ranking differences, no native reduced-dimension feature, no controlled repeated search-latency study, and non-blind public-test exposure. Decision text can exceed the trained sequence range; successful API execution is not proof of trained-length validity. Model inference, vector processing, metrics, bootstrap, calibration and figure generation ran on RunPod; this report transcribes those saved measurements without off-RunPod statistical recomputation.

## Remaining delivery blockers

GPU compute was verified stopped at 04:31:19 UTC. Deletion of the pod and its associated 150 GB workspace requires owner approval; retained storage continues at USD 0.042/hour. Final Drive artifacts are verified (20 files, 642,211,029 bytes). GitHub publication is not yet confirmed: the available connector cannot set the required anonymous author and the transfer environment lacks authenticated git push. Local generic-author commits preserve the publication until a supported authenticated route is available.
