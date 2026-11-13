# MetaEncoder v1 / SkillRet

**Publication reconciliation pending.** The primary retrieval run processed all 6,006 skills and evaluated all 4,392 test queries on RunPod. Secondary decision experiments and final artifact delivery remain under verification. The full task is not yet complete.

## Primary retrieval

| Configuration | Queries | Catalog | NDCG@10 | Recall@10 | Completeness@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| MetaEncoder v1, 6,656-dimensional BF16 inference / FP32 output, content-preserving chunk aggregation | 4,392 | 6,006 | 0.720593 | 0.791249 | 0.656421 | 0.792056 |
| Historical SkillRet-Embedding-0.6B full text | 4,392 | 6,006 | 0.810990 | 0.876442 | Not yet reconciled here | 0.856349 |

MetaEncoder's measured primary retrieval scores are lower than the supplied historical baseline. This does not establish that every MetaEncoder capability is inferior: native API equivalence, dimension reductions and decision reranking must be judged using their own measured scope. Historical local latency is not a controlled RunPod latency comparator.

## Conditions and caveats

- Both use the same pinned public test corpus and canonical full-text convention. This is an exposed public benchmark, not a blind unseen test.
- MetaEncoder document inputs exceeding the trained 16,384-token boundary use deterministic content-preserving chunks and normalized mean aggregation. Nine documents exceed that boundary; no query does. A separate fixed-first-chunk comparison isolates this protocol change.
- Prefix-truncated embedding dimensions are exploratory and not a documented native feature.
- Retrieval ranks by cosine similarity with a fixed original-catalog-row tie rule. BF16 inference returns FP32 normalized arrays.
- A subgroup evaluation implementation error stopped the first scoring pass after embedding completion. It was corrected by restricting rankings to each subgroup, then scoring was rerun from cached arrays. No expensive embedding inference needed repeating for this correction.
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

Sources: [aggregate measurements](results/retrieval_summary.json), [full-dimension per-query metrics](results/full_6656_metrics.json), [top-100 rankings](results/full_6656_top100.jsonl). All configurations use the same completed primary inference; dimensions below 6,656 do not incur additional inference. Times measure a single cached CPU NumPy matrix multiplication for all 4,392 queries against 6,006 skills, excluding ranking sort, model loading, query inference and network transfer. They are not warm request-latency estimates.

## Native retrieve

The native interface completed 20 queries against the full corpus. Sample NDCG@10=0.804087, Recall@10=0.858333, Completeness@10=0.750000 and MRR@10=0.885000. These are different-query-scope metrics, **not** improvements over the full 4,392-query manual result. Native calls include fresh query encoding, unlike cached matrix multiplication. Detailed equivalence diagnostics remain pending.

## Runtime and validation

RunPod H100 PCIe 80 GB; allocated 16 vCPU and 251 GB RAM, Intel Xeon Platinum 8352Y. Actual package versions and device details are preserved in manifests. The 10 tests in `test_metrics` and `test_protocol` passed on RunPod; [log](results/tests.log). Embedding arrays were validated for expected shape, finite values, FP32 dtype and L2 normalization before export. Nine long documents were chunked, and no query exceeded the trained boundary.

## Delivery and remaining work

Lossless vectors are uploaded to [MEM/MIA5150/MetaEncoder-v1-SkillRet-6006-4392](https://drive.google.com/drive/folders/16zcgqtaJSVOAv46mWwf7zli1GOhdf-h4). Final evidence, secondary scope, paired bootstrap, figures, cost and resource teardown remain in progress. See [DRIVE_ARTIFACTS.md](DRIVE_ARTIFACTS.md) for exact delivery verification. This is a primary-retrieval checkpoint, not a claim that the entire requested study has finished.

