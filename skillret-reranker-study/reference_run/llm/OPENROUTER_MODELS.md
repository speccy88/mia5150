# Five OpenRouter models: measured pilot

**68 valid selector decisions out of 100 planned** on five fixed SkillRet pilot queries, N=50. Models use real Deep Agents SkillsMiddleware and candidate-only StateBackend through native ChatOpenRouter. Valid formatting does not imply correct selection. All results below are regenerated from saved rows.

There are 79 attempted selector cases, 32 exact sets, 6 gated cases and 15 unrun cases. Qwen3.8 27B expansion stopped after two distinct listing requests hit Darkbloom shared-pool rate limits; successful short-prompt results do not complete its comparison. Unknown IDs and malformed calls count as failures, and are never repaired. Larger live libraries remain unrun.

| Model | Pinned provider | Current canary | Valid / attempted / planned selectors | Exact sets | Valid activation / attempted | Actual native reads | All-config USD |
|---|---|---|---:|---:|---:|---:|---:|
| qwen38-27b | darkbloom/fp4 | pass | 2/4/20 | 2 | 3/5 | 3 | $0.003998 |
| qwen38-flash | alibaba | pass | 15/18/20 | 5 | 4/5 | 5 | $0.090067 |
| deepseek41-flash | inference-net | pass | 20/20/20 | 9 | 4/5 | 4 | $0.020007 |
| mimo-v2-6-pro | deepinfra/fp8 | pass | 13/18/20 | 6 | 0/5 | 0 | $0.249035 |
| glm53-flash | relace | pass | 18/19/20 | 10 | 4/5 | 4 | $0.023391 |

Costs include initial and revised canaries, selectors, domain routers and activation; model subtotals include the earlier credential check. Whole-project paid ledger: **US$0.386496410**, approximately **CAD 0.579744616** at the declared planning factor 1.50. 236 reserved attempts; unresolved reserve **US$0.000000000**. This conversion is an estimate; bank fees/exchange and local hardware electricity are not measured. Persistent ceilings remain USD 3, CAD 5, 300 requests. No automatic retries or provider fallback.

## Selection and retrieval

The first pre-credit full-document BM25 result remains **Recall@5 0.516667 (51.67%) across 200 queries / 6,006 skills**; Recall@10 0.576667. It is a partial-protocol check, not full published-score reproduction. It uses a different representation and denominator from this small metadata selector study.

| Model | Method | Attempts / planned | Valid | Exact / valid | Candidate recall |
|---|---|---:|---:|---:|---:|
| qwen38-27b | listing | 2/5 | 0 | 0/0 | 1.000 |
| qwen38-27b | bm25 | 1/5 | 1 | 1/1 | 1.000 |
| qwen38-27b | vector | 1/5 | 1 | 1/1 | 1.000 |
| qwen38-27b | domain_vector | 0/5 | 0 | — | — |
| qwen38-flash | listing | 5/5 | 4 | 1/4 | 1.000 |
| qwen38-flash | bm25 | 5/5 | 4 | 1/4 | 0.900 |
| qwen38-flash | vector | 5/5 | 5 | 3/5 | 0.800 |
| qwen38-flash | domain_vector | 3/5 | 2 | 0/2 | 0.667 |
| deepseek41-flash | listing | 5/5 | 5 | 1/5 | 1.000 |
| deepseek41-flash | bm25 | 5/5 | 5 | 2/5 | 0.900 |
| deepseek41-flash | vector | 5/5 | 5 | 3/5 | 0.800 |
| deepseek41-flash | domain_vector | 5/5 | 5 | 3/5 | 0.800 |
| mimo-v2-6-pro | listing | 5/5 | 3 | 2/3 | 1.000 |
| mimo-v2-6-pro | bm25 | 5/5 | 4 | 1/4 | 0.900 |
| mimo-v2-6-pro | vector | 5/5 | 3 | 3/3 | 0.800 |
| mimo-v2-6-pro | domain_vector | 3/5 | 3 | 0/3 | 0.667 |
| glm53-flash | listing | 5/5 | 4 | 2/4 | 1.000 |
| glm53-flash | bm25 | 5/5 | 5 | 4/5 | 0.900 |
| glm53-flash | vector | 5/5 | 5 | 3/5 | 0.800 |
| glm53-flash | domain_vector | 4/5 | 4 | 1/4 | 0.625 |

All models share the same five queries (three single-positive, two multi-positive), N=50 gold-preserving libraries, candidate order, capped metadata, prompts and tool schema. Retrieval shortlists have at most five candidates. Listing candidate coverage is 1 by construction. Domain routing occurs only when listing/BM25/vector outputs for that query are valid; otherwise it is recorded as gated, not treated as a model answer. Thus conditional accuracy denominators can differ. CSVs give precision/recall/F1, single/multiple strata, planned coverage, tokens, latency and routing cost. Logical domain cost includes the router even if reused.

## Integration and controlled changes

Initial Qwen Flash and DeepSeek canaries returned s-prefixed native names instead of original UUIDs. Strict validation rejected them. One fixed input/schema UUID clarification, already used by the local Ollama trial, was applied uniformly to all five models before benchmark selection. Original failures remain saved; no output was repaired or sampled repeatedly for correctness. Real native metadata injection and tool isolation remain active.

GLM's pinned endpoint rejected reasoning=none with HTTP 400. Provider key cumulative usage confirmed no generation charge, and an explicit reconciliation settled that reservation at zero. GLM uses reasoning effort=low; other models request none. Output remains bounded at 512 tokens, routing at 96, temperature 0, timeout 90 seconds. This reasoning difference limits model-to-model latency/cost comparison. Provider names, model IDs, completion IDs, returned reasoning tokens, quantization tags and current price snapshots are retained. Cache-write pricing is conservatively reserved in addition to uncached prompt price for every input token.

Each successful metadata workflow has one model call and only select_skills offered. Separate activation offers bounded native read_file plus select_skills, at most four calls and two candidate reads. Actual native reads include completed ToolMessage outputs before later denials. A read is evidence of progressive disclosure, never skill execution. Downloaded instructions and links were not executed.

## Limits and saved failures

Five pilot queries are too few to establish a model or method winner. N=250/1000 live scaling remains unrun in this follow-up. The 4,192 reserved final-query texts were not consumed. Hosted endpoints/weights and cache state can change; temperature zero is not bitwise reproducibility. Local Ollama comparison is exploratory: it shares UUID clarification and query/library selection, but has different model sizes, quantization, runtime and generation controls. See the separate [local report](OLLAMA_TRIAL_REPORT.md).

- qwen38-27b selection q-04792: blocked; TooManyRequestsResponseError: Provider returned error.
- qwen38-27b selection q-04792: blocked; selector_prerequisite_failed.
- qwen38-27b selection q-04981: blocked; TooManyRequestsResponseError: Provider returned error.
- qwen38-27b activation q-04792: blocked; parallel_reads_forbidden.
- qwen38-27b activation q-04981: blocked; parallel_reads_forbidden.
- qwen38-flash selection q-04981: unknown_ids; unknown_ids.
- qwen38-flash selection q-04981: blocked; selector_prerequisite_failed.
- qwen38-flash selection q-00562: unknown_ids; unknown_ids.
- qwen38-flash selection q-00562: blocked; selector_prerequisite_failed.
- qwen38-flash selection q-00391: malformed_output; malformed_output.
- qwen38-flash activation q-04981: blocked; parallel_reads_forbidden.
- qwen38-flash canary fixture: unknown_ids; strict output validation.
- deepseek41-flash activation q-04981: blocked; parallel_reads_forbidden.
- deepseek41-flash canary fixture: unknown_ids; strict output validation.
- mimo-v2-6-pro selection q-04981: unknown_ids; unknown_ids.
- mimo-v2-6-pro selection q-04981: unknown_ids; unknown_ids.
- mimo-v2-6-pro selection q-04981: blocked; selector_prerequisite_failed.
- mimo-v2-6-pro selection q-00562: unknown_ids; unknown_ids.
- mimo-v2-6-pro selection q-00562: unknown_ids; unknown_ids.
- mimo-v2-6-pro selection q-00562: unknown_ids; unknown_ids.
- mimo-v2-6-pro selection q-00562: blocked; selector_prerequisite_failed.
- mimo-v2-6-pro activation fixture-0: blocked; parallel_reads_forbidden.
- mimo-v2-6-pro activation fixture-1: blocked; parallel_reads_forbidden.
- mimo-v2-6-pro activation fixture-2: blocked; parallel_reads_forbidden.
- mimo-v2-6-pro activation q-04792: blocked; parallel_reads_forbidden.
- mimo-v2-6-pro activation q-04981: blocked; parallel_reads_forbidden.
- glm53-flash selection q-00562: unknown_ids; unknown_ids.
- glm53-flash selection q-00562: blocked; selector_prerequisite_failed.
- glm53-flash activation q-04981: blocked; parallel_reads_forbidden.
- glm53-flash canary fixture: blocked; BadRequestResponseError: Reasoning is mandatory for this endpoint and cannot be disabled..

## Reproduce or resume

```bash
.venv/bin/python scripts/run_openrouter_models.py --stage all
.venv/bin/python scripts/report_openrouter.py
.venv/bin/python scripts/audit_openrouter.py
.venv/bin/python scripts/verify.py
.venv/bin/python pilot.py package
```

The first command requires the privately configured key and ALLOW_PAID_RUNS=1. Matching cases, including failed outputs, are reused. Do not delete ledgers or prior results. Configurations are in manifests/configs; raw traces in evidence/traces; per-model results in results/openrouter. Both notebooks default to saved/offline evidence without model calls.
