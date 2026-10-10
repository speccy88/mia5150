# MIA5150 local experiments

Generated 2026-10-05T23:12:36.627323+00:00. Paid spend: **0**. Hardware: RTX2060SUPER8GB/SM75, i5-9500T, 16GB Windows RAM; Ubuntu26.04 WSL2 capped at6GB RAM/2CPU/2GB swap.

This report separates released-model deployment comparisons from our own adaptation experiments. Public benchmark feedback was already exposed; none of these SkillRet results is blind final-test confirmation. Read `IMPLEMENTATION_MAP.md` for implementation boundaries.

**Main finding:** Equal fusion changes full-public NDCG@10 from 0.730655 to 0.730792: +0.000136, with exploratory cluster 95% CI[-0.006117,+0.006769]. This is essentially flat. Actual hard-negative LoRA improves its 32-query development mean from 0.678483 to 0.691718, but transfer changes from 0.762598 to 0.759710; both paired intervals include zero. No reliable new benchmark improvement is established.

The corrected replay attempt is resource-blocked. Its interrupted checkpoints are preserved; no completed adapted evaluation or gain is claimed. Released specialist models outperform the generic baseline, which is a deployment finding, not a gain caused by our training.

## All 4,392 corrected public queries: fixed 6,006-skill header catalog

| System | Queries | NDCG@10 | Recall@10 | Complete@10 | MRR@5 | Failed/empty |
|---|---:|---:|---:|---:|---:|---:|
| public-full-bm25 | 4392 | 0.6199 | 0.7034 | 0.5676 | 0.6748 | 0.0% |
| public-full-qwen_embed_06 | 4392 | 0.5694 | 0.6421 | 0.4713 | 0.6434 | 0.0% |
| public-full-skillret_embed_06 | 4392 | 0.7307 | 0.8009 | 0.6794 | 0.7888 | 0.0% |
| public-full-rrf | 4392 | 0.7308 | 0.8040 | 0.6867 | 0.7850 | 0.0% |

The dataset revision is `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`. Dense models use native query instructions, FP16, maximum 1,024 tokens, exact cosine search and one query per forward pass. RRF uses k60 and the top100 from each channel. These are local bounded-context configurations, not the original8B/BF16/full-context reference.

## Public 128-query development screen

| System | Queries | NDCG@10 | Recall@10 | Complete@10 | MRR@5 | Failed/empty |
|---|---:|---:|---:|---:|---:|---:|
| bm25-locked-header | 128 | 0.5988 | 0.7070 | 0.5469 | 0.6392 | 0.0% |
| bm25-locked-full | 128 | 0.4729 | 0.5195 | 0.3594 | 0.5520 | 0.0% |
| qwen_embed_06-header-0 | 128 | 0.5287 | 0.5924 | 0.3984 | 0.6174 | 0.0% |
| skillret_embed_06-header-0 | 128 | 0.7074 | 0.7799 | 0.6484 | 0.7768 | 0.0% |
| rrf-skillret-bm25 | 128 | 0.7226 | 0.7773 | 0.6484 | 0.7969 | 0.0% |
| skillrouter_embed_06-header-0 | 128 | 0.7087 | 0.7513 | 0.5625 | 0.8081 | 0.0% |
| skillreason_embed_06-header-0 | 128 | 0.6917 | 0.7591 | 0.5781 | 0.7943 | 0.0% |
| tfidf-fields | 128 | 0.5146 | 0.6055 | 0.4609 | 0.5607 | 0.0% |
| tfidf-flattened | 128 | 0.5171 | 0.5690 | 0.3984 | 0.5911 | 0.0% |

Header and full-document evidence are distinct tracks. Full neural documents and each dense field are capped at 1,024 tokens; the report does not imply all source text was visible. Sparse full-document baselines use the available complete text. Field-six is an adaptation of uniform six-channel field cosine, not the older-pool paper leaderboard or its learned MLP. Candidate recall/completeness and rescued/displaced positives at20/50/100 are in `results/candidate-coverage.json`.

## Shared 32-query / identical 20-candidate reranking pool

| System | Queries | NDCG@10 | Recall@10 | Complete@10 | MRR@5 | Failed/empty |
|---|---:|---:|---:|---:|---:|---:|
| skillret_rerank_06-header-32 | 32 | 0.7992 | 0.8490 | 0.7812 | 0.8359 | 0.0% |
| skillrouter_rerank_06-header-32 | 32 | 0.8245 | 0.8490 | 0.7812 | 0.8672 | 0.0% |
| skillreason_rerank_06-header-32 | 32 | 0.8252 | 0.8490 | 0.7812 | 0.8698 | 0.0% |
| r3_rerank_fp16_local-header-32 | 32 | 0.7955 | 0.8177 | 0.7500 | 0.8438 | 0.0% |
| jina-native | 32 | 0.8246 | 0.8490 | 0.7812 | 0.8646 | 0.0% |

The pool is frozen SkillRet0.6B dense retrieval without gold insertion. Jina uses its reviewed native joint-candidate attention and learned projector, capped at 8,192 total tokens, rather than a substitute pooling method. Common source headers are identical; native token layouts differ. Failed outputs retain the denominator. Native role-specific instructions are saved in source snapshots.

## Bounded 8-query quantized pointwise screen

| System | Queries | NDCG@10 | Recall@10 | Complete@10 | MRR@5 | Failed/empty |
|---|---:|---:|---:|---:|---:|---:|
| jevk5-pointwise | 8 | 0.6788 | 0.6875 | 0.6250 | 0.7500 | 0.0% |
| plumb-pointwise | 8 | 0.6788 | 0.6875 | 0.6250 | 0.7500 | 0.0% |

Matched8-query controls: skillret_embed_06-header-0 NDCG@10=0.6533, skillret_rerank_06-header-32 NDCG@10=0.6764, skillrouter_rerank_06-header-32 NDCG@10=0.6806, skillreason_rerank_06-header-32 NDCG@10=0.6788.

JevK5 uses Q5_K_M and native option-letter log-probabilities with T1.22; Plumb uses Q4_K_M/T2.07. Missing logits are failures, not fabricated probability floors. The Qwen8 native yes/no implementation was not evaluated: its download hit the resource guard. The completed capped15+NONE choice diagnostics retain original/reversed outputs. The21-option multi-pass implementation was not run. Neither choice setup establishes multi-skill selection or genuine no-skill-needed activation.

## Actual rank4 q/v LoRA:32-query grouped development set

| System | Queries | NDCG@10 | Recall@10 | Complete@10 | MRR@5 | Failed/empty |
|---|---:|---:|---:|---:|---:|---:|
| lora-hard-base | 32 | 0.6785 | 0.6667 | 0.4375 | 0.8516 | 0.0% |
| lora-hard-adapted | 32 | 0.6917 | 0.6667 | 0.4375 | 0.8656 | 0.0% |

Adapters use seed5150, alpha8, lr5e-5, FP16/gradient scaling, gradient checkpointing and a384-example ceiling. All labelled positive pairs are kept; only negatives are sampled to meet the corrected replay cap. Native deployment reload and exact adapter/optimizer/scaler/RNG restoration were checked. The released base already saw public training data: these results measure new adaptation, not clean pretraining generalization.

**Superseded replay:** the original replay cap dropped8 of91 unique positive pairs. Its earlier0.6946 result is retained for audit but excluded from supported gains. Corrected replay retains all91 planned positives, but its runs were stopped by resource guards before evaluation. Checkpoint restoration was verified; completed training and improvement are not claimed. Older lexical files without the clean designation also include a negative-partition defect; see the exposure ledger.

## LoRA transfer: same32 public queries, full512 evidence and fixed20 pools

| System | Queries | NDCG@10 | Recall@10 | Complete@10 | MRR@5 | Failed/empty |
|---|---:|---:|---:|---:|---:|---:|
| lora-transfer-base | 32 | 0.7626 | 0.8333 | 0.7500 | 0.7708 | 0.0% |
| lora-transfer-hard | 32 | 0.7597 | 0.8021 | 0.7188 | 0.7917 | 0.0% |

Training/test skill IDs are disjoint, but this public query panel was already inspected. This is a transfer-regression diagnostic, not untouched external confirmation. Development and transfer results must both inform any promotion.

## Clean learned lexical fusion: separate128-query training-derived development set

| System | Queries | NDCG@10 | Recall@10 | Complete@10 | MRR@5 | Failed/empty |
|---|---:|---:|---:|---:|---:|---:|
| adaptation-clean-no-adaptation | 128 | 0.6260 | 0.6628 | 0.4219 | 0.7676 | 0.0% |
| adaptation-clean-hard-negative | 128 | 0.6343 | 0.6745 | 0.4453 | 0.7922 | 0.0% |
| adaptation-clean-label-smoothing | 128 | 0.6370 | 0.6745 | 0.4453 | 0.7965 | 0.0% |
| adaptation-anchoring | 128 | 0.6530 | 0.6784 | 0.4531 | 0.8194 | 0.0% |

Training uses512 queries and repository/exact-name components. Cross-partition multi-positive queries and overlapping normalized query text are excluded; development-group documents cannot enter training negatives. Label smoothing, replay and L2 anchoring are explicitly different interventions.

## Fixedfive-skill structural selection

| Method | NDCG@5 | Complete@5 | SetF1 | Exact set |
|---|---:|---:|---:|---:|
| top5 | 0.6760 | 0.5312 | 0.3389 | 0.0000 |
| mmr | 0.6772 | 0.5234 | 0.4544 | 0.0938 |
| dsr-embedding-quality | 0.6455 | 0.4688 | 0.3105 | 0.0000 |
| domain1 | 0.2989 | 0.1875 | 0.1394 | 0.0000 |
| domain2-global | 0.6760 | 0.5312 | 0.3389 | 0.0000 |

No gold set size is provided to the selector. One-domain restriction loses relevant skills; global fallback recovers by retaining unrestricted hits. Similar skills may both be required, so diversity is not automatically beneficial. No compatibility, dependency or activation labels were invented.

## Bounded escalation diagnostic

Calibration16queries and evaluation16queries are disjoint. A label-free calibration-median margin threshold escalates 31.2% of evaluation queries. NDCG@10: dense 0.7899; policy 0.8313; always rerank 0.8661. Estimated warm latency {'mean': 0.6593824263750321, 'p50': 0.09101114750001216, 'p95': 2.1081083560000025}. This sum of separate component timings excludes model switching/startup and is not measured deployment end-to-end latency.

## Scaling and full-catalog prompting

The same eight queries, three seeds and random/hard-overlap nested libraries are used across10/100/1000/6006 skills. Gold-conditioned library construction is explicitly a separate experiment. No label or required-skill count enters the scorers. At10 skills, complete-list Recall@10 is trivial.

scale-prompt: context=8192; statuses={'ok': 33, 'context_overflow': 141, 'invalid_or_failed': 18}.

Overflow and malformed outputs are counted as failures; no hidden retrieval reduces the full-catalog prompt. The32768-token track was not run. The four-query prompted/validated-ID ranking and pairwise diagnostic was interrupted by the host-RAM guard before completion; no quality score is claimed. It does not use grammar-constrained decoding.

## Measured deployment timing

Durations include recorded failures and use the stated query cohort. Reranking excludes retrieval/loading; encoding includes exact search but excludes document indexing. Missing timings are unavailable, not zero.

| System | Cohort | Warm p50 seconds | Warm p95 seconds | Failure rate |
|---|---|---:|---:|---:|
| qwen_embed_06 | 128 public development | 0.0735 | 0.1554 | 0.0% |
| skillret_embed_06 | 128 public development | 0.0748 | 0.1465 | 0.0% |
| skillrouter_embed_06 | 128 public development | 0.0865 | 0.1679 | 0.0% |
| skillreason_embed_06 | 128 public development | 0.0688 | 0.1533 | 0.0% |
| BM25-header | 128 public development | 0.0013 | 0.0022 | 0.0% |
| BM25-full | 128 public development | 0.0016 | 0.0025 | 0.0% |
| skillret_rerank_06-header-32 | 32 public development | 1.8338 | 2.0962 | 0.0% |
| skillrouter_rerank_06-header-32 | 32 public development | 1.8510 | 2.0985 | 0.0% |
| skillreason_rerank_06-header-32 | 32 public development | 1.7546 | 2.0219 | 0.0% |
| r3_rerank_fp16_local-header-32 | 32 public development | 1.9898 | 2.4938 | 0.0% |
| jina-native | 32 public development | 0.3185 | 0.3976 | 0.0% |
| jevk5-pointwise | 8 public development | 4.5511 | 5.9924 | 0.0% |
| plumb-pointwise | 8 public development | 4.4901 | 5.8166 | 0.0% |

## Paired uncertainty and promotion

Gold-overlap connected-component bootstrap,2000 resamples, seed5150. The following intervals are exploratory and not corrected for the multiple screened methods. A positive development mean alone is insufficient for promotion.

| Contrast | NDCG@10 delta | Cluster95% interval | Wins/losses/ties |
|---|---:|---|---|
| fusion_minus_dense | +0.0152 | [-0.0253, +0.0546] | 36/28/64 |
| skillret_rerank_06_minus_dense | +0.0858 | [+0.0389, +0.1442] | 11/0/21 |
| skillrouter_rerank_06_minus_dense | +0.1111 | [+0.0490, +0.1832] | 11/0/21 |
| skillrouter_rerank_06_minus_skillret_reranker | +0.0254 | [+0.0005, +0.0577] | 4/1/27 |
| skillreason_rerank_06_minus_dense | +0.1119 | [+0.0492, +0.1857] | 11/0/21 |
| skillreason_rerank_06_minus_skillret_reranker | +0.0261 | [+0.0015, +0.0586] | 5/2/25 |
| r3_rerank_fp16_local_minus_dense | +0.0821 | [+0.0091, +0.1672] | 9/3/20 |
| r3_rerank_fp16_local_minus_skillret_reranker | -0.0037 | [-0.0498, +0.0437] | 4/3/25 |
| jina_minus_dense | +0.1112 | [+0.0483, +0.1869] | 10/2/20 |
| jina_minus_skillret_reranker | +0.0254 | [-0.0110, +0.0669] | 5/2/25 |
| lora-hard | +0.0132 | [-0.0098, +0.0447] | 5/3/24 |
| lexical-hard-negative | +0.0083 | [-0.0310, +0.0480] | 26/27/75 |
| lexical-label-smoothing | +0.0109 | [-0.0286, +0.0508] | 27/27/74 |
| lexical-anchoring | +0.0270 | [-0.0086, +0.0646] | 29/21/78 |
| public-full-bm25-minus-skillret | -0.1107 | [-0.1206, -0.1012] | 710/1769/1913 |
| public-full-qwen_embed_06-minus-skillret | -0.1613 | [-0.1705, -0.1517] | 410/2115/1867 |
| public-full-rrf-minus-skillret | +0.0001 | [-0.0061, +0.0068] | 1077/866/2449 |
| lora-transfer-hard | -0.0029 | [-0.0306, +0.0254] | 3/4/25 |

Deployment decision: retain SkillRet0.6B as the reference. EqualRRF has no demonstrated full-query NDCG gain and is not promoted as an improvement. Released rerankers remain quality candidates pending broader paired validation. Do not promote a LoRA, domain restriction, diversity policy or escalation rule solely on these small development means. The three leading released rerankers should not be ordered confidently by a sub-0.001 difference.

## Resource and reproducibility limits

All expensive stages ran sequentially through `guard.py`. Successful runs must show no new swap activity and the requested reserves. Failed loading attempts are retained and excluded from completed-run claims. Direct-to-GPU loading and completed-file cache hints reduced transient pressure without changing WSL limits. `results/resources.json` reports measured maxima/minima; raw telemetry is retained. WorkerCPU/RSS excludes child llama servers; whole-guest CPU ticks and whole-device GPU measurements remain available. Warm timings, indexing and startup are separate; these are deployment configurations rather than a precision-controlled hardware speed contest.

The8B BF16 reference, hosted/API models, RunPod, extensive multiseed training, certified independent confirmation, and full-query neural reranking tournament were not represented by these bounded pilots. The runnable reranker accepts a4392-query limit and full-public candidate artifact for a future longer run. Full32K dense-body reproduction, SkillReason4B training and labelled activation/compatibility/graph studies remain outside the completed comparisons.

SRA and SkillReason are distinct external benchmarks. Both external adapters are implemented. SRA data acquisition completed, but neither external full-catalog index was evaluated after repeated host-resource failures. No external quality score or independent confirmation is claimed. Additional conditional retrieval encoders should be justified by the saved miss-coverage evidence, not added to an opaque combined score.

Historical Jev200/6006 results, the historical LoRA10123/256val study and the custom80-case routing pilot are retained as separate context. No new result was merged into their metrics. No paid service, credential extraction, publication, model upload or Git push occurred.

## Receipts

See `evidence/tests.json`, `evidence/notebook-smoke.json`, `evidence/plan-integrity.json`, `evidence/exposure-ledger.json`, `requirements-lock.txt`, exact `runs/*/command.json` and source snapshots. `PACKAGE_SHA256.json` covers the portable deliverables. The package excludes bulk data, ALL model/adapter weights and optimizer binaries, environments, caches, credentials and Git internals. Local checkpoints remain at /home/user/code/mia5150-local/checkpoints; hashes and sizes are in evidence/local-checkpoint-inventory.json. No Library upload is authorized or performed.

## Unfinished local coverage

Resource-stopped: corrected replay, full-body1024 retrieval, six-channel field encoding (only the name channel completed), Qwen8 download, and cached4B ID/pairwise diagnostics. Not run after those failures:21-option multi-pass choices,32K prompting, SRA/SkillReason full-catalog transfer and actual end-to-end escalation latency. The existing escalation latency remains a sum of separate component measurements, not an end-to-end measurement. See EXECUTION_STATUS.md and evidence/remaining-resource-blockers.json.
