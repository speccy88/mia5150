# SkillRet-0.6B full-text baseline and Jev RRF60 findings

Report snapshot: 2026-10-07T01:51:40Z. Baseline status: **validated**.

The full dense result contains 4,392 queries, 6,006 candidate skills and 7,187 relevant pairs. Maximum absolute NDCG deviation from the released target is 0.000323. It passes the prespecified 0.005 NDCG reproduction gate.

Jev has not produced a complete evaluated comparison in this phase. Its score, latency, estimated cost and improvement are unknown. No synthetic Jev result is reported.

## Primary comparison

| System and scope | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | Completeness@5 | MAP@10 | MRR@5 | Median latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — SkillRet-0.6B (4392 queries) | 0.7905 | 0.8110 | 0.8244 | 0.8764 | 0.7097 | 0.7539 | 0.8533 | 0.0698 |
| B — SkillRet -> Jev (not run) | — | — | — | — | — | — | — | — |
| C — fixed RRF60 (not run) | — | — | — | — | — | — | — | — |

Baseline latency is active query encoding plus dense dot-product search. It excludes the recorded inter-forward safety cooldown, cache writes, model loading and catalogue indexing; telemetry below records actual wall time. B/C latency is API batch time when supplied, excludes retrieval, and is shared by Jev and fusion rather than a second API run. Missing or failed selected hosted queries retain empty rankings and the selected denominator. A pilot is never extrapolated into a full result.

## All measured metrics

| Metric | A full dense | A matched hosted scope | B Jev | C RRF60 |
|---|---:|---:|---:|---:|
| NDCG@5 | 0.7905 | — | — | — |
| NDCG@10 | 0.8110 | — | — | — |
| NDCG@15 | 0.8188 | — | — | — |
| Recall@5 | 0.8244 | — | — | — |
| Recall@10 | 0.8764 | — | — | — |
| Recall@15 | 0.9018 | — | — | — |
| Completeness@5 | 0.7097 | — | — | — |
| Completeness@10 | 0.7880 | — | — | — |
| Completeness@15 | 0.8272 | — | — | — |
| MAP@5 | 0.7413 | — | — | — |
| MAP@10 | 0.7539 | — | — | — |
| MAP@15 | 0.7572 | — | — | — |
| Precision@5 | 0.2593 | — | — | — |
| Precision@10 | 0.1399 | — | — | — |
| Precision@15 | 0.0962 | — | — | — |
| MRR@5 | 0.8533 | — | — | — |
| MRR@10 | 0.8563 | — | — | — |

## Released targets and measured deviations

| Metric | Released target | Actual dense | Actual minus target |
|---|---:|---:|---:|
| NDCG@5 | 0.7902 | 0.7905 | 0.000323 |
| NDCG@10 | 0.8112 | 0.8110 | -0.000210 |
| NDCG@15 | 0.8187 | 0.8188 | 0.000148 |
| Recall@5 | 0.8238 | 0.8244 | 0.000578 |
| Recall@10 | 0.8774 | 0.8764 | -0.000958 |
| Recall@15 | 0.9016 | 0.9018 | 0.000191 |
| Completeness@5 | 0.7083 | 0.7097 | 0.001399 |
| Completeness@10 | 0.7894 | 0.7880 | -0.001376 |
| Completeness@15 | 0.8265 | 0.8272 | 0.000686 |
| MAP@5 | 0.7409 | 0.7413 | 0.000369 |
| MAP@10 | 0.7538 | 0.7539 | 0.000065 |
| MAP@15 | 0.7570 | 0.7572 | 0.000231 |
| Precision@5 | 0.2591 | 0.2593 | 0.000190 |
| Precision@10 | 0.1401 | 0.1399 | -0.000232 |
| Precision@15 | 0.0962 | 0.0962 | 0.000036 |

A deviation above 0.005 triggers investigation of split, catalogue, full document construction, exact prefix, normalization, truncation, qrels and metric conventions. FP16/runtime adaptations and the dataset revision must also be considered. Metrics were recomputed from saved rankings and checked against the baseline's stored aggregates.

## Candidate coverage and failure evidence

Dense top20: macro candidate Recall=0.917502; relevant-pair coverage=6,476/7,187 (0.901071); all-gold queries=3,742/4,392.
Dense top50: macro candidate Recall=0.953893; relevant-pair coverage=6,781/7,187 (0.943509); all-gold queries=4,007/4,392.

Valid A/B/C results use exactly the same twenty candidates. Reranking cannot recover a relevant skill missing from that pool. Dense top50 coverage describes the upstream retrieval ceiling; a 20-item reranking must not be called a top50 retrieval result.

`failure_analysis.csv` contains 12 deterministic examples. Baseline-only categories are coverage misses, relevant skills at ranks 6–20, and incomplete multi-skill top5 results. Actual hosted comparisons add both positive and negative cases when available. Blank Jev/RRF columns are explicit unrun/failed values, not zero-quality measurements.

## Statistical comparison

Paired bootstrap confidence intervals are not run because there is no actual paired RRF comparison. The placeholder has null metrics.

## Text, tokens, versions and sources

The official document is stripped `name | description | skill_md`. The exact query prefix is `Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: `, with a real newline and one final space. Full text is retained in source; model input uses deterministic right-side token truncation at the configured limit. Vectors use last-token pooling and float32 L2 normalization; ranking uses dot products.

- model: `ThakiCloud/SKILLRET-Embedding-0.6B`
- model_revision: `67f1594c46c40e13a5eb2f4092cc85c98c9ed32d`
- dataset: `ThakiCloud/SKILLRET`
- dataset_revision: `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`
- official_code_revision: `0cc508e6bfcfff1fe6352d90ff5957ba38a64bba`
- max_seq_length: `32768`
- weight_dtype: `float16`
- normalization_dtype: `float32`
- attention: `sdpa; explicit repeat_kv; memory-efficient CUDA kernel; math fallback disabled`
- rope_theta: `not recorded yet`
- effective_rope_theta: `1000000`
- batch_size: `1`
- cpu_threads: `1`
- Runtime packages: torch=2.8.0+cu128; transformers=4.57.1; numpy=2.5.3; tokenizers=0.22.2; accelerate=1.15.0; safetensors=0.8.0.

Current source-file hashes match the run protocol: True.
- queries: `4d07a35f40e597f8468c6dab480c95a1f9b40ecbde80274081ba2e7c643d0f6d`
- skills: `1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81`
- qrels: `edbfae0b8109de460d54e27e6235b1b47316f34956744db0571ad1b4ddb860af`

[Pinned model](https://huggingface.co/ThakiCloud/SKILLRET-Embedding-0.6B/tree/67f1594c46c40e13a5eb2f4092cc85c98c9ed32d); [pinned dataset](https://huggingface.co/datasets/ThakiCloud/SKILLRET/tree/6583d7d2ed07644d0fb8938ed8178f3a7dc42a12); [official evaluation source](https://github.com/ThakiCloud/SKILLRET/blob/0cc508e6bfcfff1fe6352d90ff5957ba38a64bba/skillret/eval.py). See `SOURCE_AUDIT.md` for provenance and earlier metadata-cache exclusions.
Documents tokenizer counts: min=117, median=1677.5, p95=5723.5, max=49012, raw total=13247447; over8192=108; over32768=2; truncated at this run's limit=2.
Queries tokenizer counts: min=57, median=252.0, p95=393.0, max=862, raw total=1128023; over8192=0; over32768=0; truncated at this run's limit=0.

## Execution history, progress and resource evidence

Recorded task-handoff history: the first feasibility attempt stopped with `No available kernel` on SM75 native grouped-query SDPA. The implementation switched to Transformers' equivalent explicit `repeat_kv` path, with memory-efficient SDPA and the quadratic math fallback disabled. This is an attention implementation adaptation, not a changed candidate catalogue or scoring objective.

A later configuration audit found that Transformers 4.57 ignored the saved nested `rope_parameters` theta=1,000,000 and instead used its default theta=10,000. Run04 was intentionally stopped (exit -15) after 84 document vectors and before any query vectors or final result; its partial cache beginning `04df631` is invalid for ranking/reproduction and is preserved for audit only. Run03 feasibility therefore establishes resource fit only, not model correctness. The corrected loader explicitly sets theta=1,000,000 and verifies the model's actual inverse-frequency tensor before encoding. A changed loader hash, runtime, actual document/query text hashes and effective theta segregate the new cache. Only the corrected protocol can support a final reproduction claim.

Run `06-full-correct-rope` stopped on the sustained host CPU guard after five consecutive samples above 80%. Its valid completed vectors were retained. A 0.05-second pause after each new forward was added without changing the embedding function, token protocol or cache identity. No resource threshold was relaxed.
Run `07-full-cpu-cooldown` left all 6,006 document vectors and 961 query vectors, then ended without an exit or guard-stop receipt. Subsequent inspection found no worker. The cause is unknown; this attempt is not recorded as completed or as a guard success. Its last saved host CPU was 56%, GPU use was 1,303 MiB, and swap counters were unchanged at zero. The verified cache was resumed in run `08-resume-query-cache`; final completion is determined by the ranking and gate artifacts above.
Saved feasibility status=passed; tested raw token lengths=[49012, 36738, 28114]; effective cap=32768; Torch peak reserved=2442.0 MiB, allocated=1997.24267578125 MiB. These are feasibility measurements, not full-run performance. If this receipt belongs to run03, it predates the RoPE fix and proves resource fit only.
Cache `04df631ea318ebe961ea0e9a3e547e7407e062250019138d671f9852f270506c` (invalid_wrong_rope_theta_preserved_do_not_use): 84/6,006 document vectors and 0/4,392 query vectors journaled; recorded encoder time=160.66s. Cached-vector counts are not fresh inference counts for this invocation. Invalid caches must not be included in valid baseline progress.
Cache `c364634bcfd07525e074a927760ef2dd0f52ce0b0f92a09debaed13a6f0d683e` (current_corrected_protocol): 6,006/6,006 document vectors and 4,392/4,392 query vectors journaled; recorded encoder time=1630.41s. Cached-vector counts are not fresh inference counts for this invocation. Invalid caches must not be included in valid baseline progress.
Cache `d7b56e8e189a54585ccb77b8afb9d1a60ca3a8137b5e811c2c552996e8eb35c1` (requires_current_protocol_match): 0/6,006 document vectors and 0/4,392 query vectors journaled; recorded encoder time=0.00s. Cached-vector counts are not fresh inference counts for this invocation. Invalid caches must not be included in valid baseline progress.
Run `01-audit`: 49 telemetry samples; exit={'returncode': 0, 'utc': 1791331346.3137684}; guard stop=None.
Host/guest minimum available=3.280/4.438 GiB; observed GPU peak=0 MiB, max temperature=47 C; max host CPU=73.0%; five-sample CPU violation=False; largest host telemetry age=2.150s; observed swap counter changes in/out=0/0.
Run `02-feasibility`: 13 telemetry samples; exit={'returncode': 1, 'utc': 1791331387.6325123}; guard stop=None.
Host/guest minimum available=3.029/3.907 GiB; observed GPU peak=2173 MiB, max temperature=49 C; max host CPU=44.0%; five-sample CPU violation=False; largest host telemetry age=1.488s; observed swap counter changes in/out=0/0.
Run `03-feasibility-repeatkv`: 75 telemetry samples; exit={'returncode': 0, 'utc': 1791331551.2675405}; guard stop=None.
Host/guest minimum available=2.781/3.851 GiB; observed GPU peak=2557 MiB, max temperature=64 C; max host CPU=73.0%; five-sample CPU violation=False; largest host telemetry age=2.136s; observed swap counter changes in/out=0/0.
Run `04-full-baseline`: 143 telemetry samples; exit={'returncode': -15, 'utc': 1791331736.1565034}; guard stop=None.
Host/guest minimum available=2.396/3.713 GiB; observed GPU peak=2381 MiB, max temperature=66 C; max host CPU=90.0%; five-sample CPU violation=False; largest host telemetry age=2.127s; observed swap counter changes in/out=0/0.
Run `05-feasibility-correct-rope`: 74 telemetry samples; exit={'returncode': 0, 'utc': 1791331897.6746442}; guard stop=None.
Host/guest minimum available=2.693/3.568 GiB; observed GPU peak=2557 MiB, max temperature=65 C; max host CPU=52.0%; five-sample CPU violation=False; largest host telemetry age=2.140s; observed swap counter changes in/out=0/0.
Run `06-full-correct-rope`: 1262 telemetry samples; exit={'returncode': 1, 'utc': 1791333288.9617424}; guard stop={'error': 'host CPU above 80% for five samples', 'utc': 1791333288.7456439}.
Host/guest minimum available=2.837/3.586 GiB; observed GPU peak=2387 MiB, max temperature=67 C; max host CPU=93.0%; five-sample CPU violation=True; largest host telemetry age=2.142s; observed swap counter changes in/out=0/0.
Run `07-full-cpu-cooldown`: 182 telemetry samples; exit=None; guard stop=None.
Host/guest minimum available=3.066/3.707 GiB; observed GPU peak=1303 MiB, max temperature=60 C; max host CPU=77.0%; five-sample CPU violation=False; largest host telemetry age=2.152s; observed swap counter changes in/out=0/0.
Run `08-resume-query-cache`: 474 telemetry samples; exit={'returncode': 0, 'utc': 1791334940.8014925}; guard stop=None.
Host/guest minimum available=3.030/3.515 GiB; observed GPU peak=1303 MiB, max temperature=61 C; max host CPU=83.0%; five-sample CPU violation=False; largest host telemetry age=2.047s; observed swap counter changes in/out=0/0.

Guard policy: at least 2 GiB free host and guest memory, no active swap, at most 7,168 MiB VRAM, temperature at most 80 C, no five consecutive host CPU samples above 80%, and host telemetry younger than eight seconds. Telemetry is sampled, not a continuous guarantee. Only this experiment's telemetry/guard receipts were read; worker and historical logs were not read by the reporter.

## Usage, limitations and next action

No paid hosted comparison is available. Estimated hosted cost is null because no actual smoke/pilot usage exists. The reporter itself does not inspect credentials or call endpoints; [hosted_stage_status.json](hosted_stage_status.json) records the saved execution checkpoint.
Recorded execution: 0 paid inference calls; 1 authenticated model-list calls; incurred API cost CAD 0. Status: `blocked_refill_status_unknown`.
After the initial process-environment absence check, the human explicitly authorized the saved TypeSafe key to be handed only to a child process as TYPESAFE_API_KEY. The authorized handoff was performed; its value was never printed or persisted, and the child environment ended with the process. This was a specific authorization, not a persistent environment change. Model discovery evidence is saved under `hosted-execution/model_list*`; returned aliases do not establish an actual paid-response model-version pin.
Billing reassessment accepts the user's reported USD 5 signup credit as an experiment-consumption ceiling, not a verified live balance. Current official Jev 1.13 pricing is verified at USD 0.042/M input tokens with free output. No balance lookup or renewed budget approval is required. The sole pending account fact is whether automatic refill was previously enabled: the local consumption ledger cannot cap a separate threshold-triggered credit purchase. No setting change is requested. See ../BILLING_GATE_REASSESSMENT.md for sources, exact access limitations and conservative request bounds.

## Hosted context preparation

The provider documents rounded 64k total and 32k state-plus-longest-question limits without defining their exact boundaries. This study's local interpretation is 64,000/32,000. The check counts canonical UTF-8 JSON request bytes plus 2,048 input-framing allowance and 2,048 output allowance, for both the complete request and each same-state single-question request. These are study choices, not native Jev token counts, a proved token upper bound or verified server boundaries. See the [context source audit](../CONTEXT_SOURCE_AUDIT.md).

Historical full-text-only plans admitted 6/10 smoke queries (4 blocked, 5 candidate admission failures) and 104/200 pilot queries (96 blocked, 143 candidate admission failures). Those local conservative failures were not provider rejections and did not prove native context overflow. No truncation or candidate/query substitution occurred in those original plans. They remain preserved for comparison.

The user subsequently authorized deterministic batching and logged right truncation of skill_md for candidates that fail the local check. Batch splitting cannot fix a single candidate's own same-state question size. The opt-in policy preserves query text, IDs, name, description, rubric and the fixed top20 pool; only skill_md may become its longest fitting Unicode prefix. Original text, retained spans and original/prepared hashes and counts are preserved. A query still blocks if immutable content plus the required nonempty body cannot fit. The dense baseline is unchanged.
Latest preparation evidence: [context_preparation_audit.json](context_preparation_audit.json). These recorded counts supersede the historical readiness counts above; they are not inferred from a policy description.
An optional [context preparation summary](context_preparation_summary.md) presents the saved audit.

| Scope | Selected | Prepared | Blocked | Query-candidate pairs | Truncated pairs | Full-text pairs | Queries with truncation | Unique truncated skills | Removed skill_md UTF-8 bytes | Retained skill_md byte fraction | Requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Smoke | 10 | 10 | 0 | 200 | 5 | 195 | 4 | 5 | 39777 | 0.975853 | 40 |
| Pilot | 200 | 200 | 0 | 4000 | 143 | 3857 | 96 | 103 | 1423995 | 0.957207 | 779 |

Smoke manifest: `jev-plan-smoke-context-v1/preparation_manifest.json`; plan hash `4202da9d6067127246f25bcc3586f0157d03949cdc176862b1243f00f80d690a`.
Pilot manifest: `jev-plan-pilot-context-v1/preparation_manifest.json`; plan hash `6da7ec1a2062a4a1d6c38aae6c14ea268188fe05933fc8fd3b5f5bec9cb1f1b8`.
Local preparation readiness does not verify provider token admission, available credit, pricing/FX/fees, the live model pin or other hosted execution gates. Truncation is driven by the local estimate, not demonstrated native overflow. Any later quality analysis must report full-text and truncated subsets while preserving the fixed selected-query denominator. Preparation itself makes no network request; any separately authorized model-list request is counted in the execution status above.
Dense retrieval median/p95=0.069773/0.134777 seconds per query; document encoding=1289.45 seconds (one-time cache work); model startup this run=2.41 seconds.
No conclusion is available about Jev ordering gains, multi-skill completeness gains, redundancy with dense retrieval, category-specific harms or whether its added latency/API dependency is justified. The full study is unfinished.
This is an exposed public benchmark reproduction, not a newly blinded evaluation. Keep earlier runs intact. The full local baseline passed its reproduction gate and the CAD 10 all-in experiment budget is approved. The hosted transport accepts credentials only through TYPESAFE_API_KEY, including the explicitly authorized temporary child-process handoff. Paid execution still requires the remaining verified account/provider evidence and persistent budget reservations; no top-up or RunPod action is authorized. No claim about Jev or RRF quality is supported until actual paid-response results exist.
