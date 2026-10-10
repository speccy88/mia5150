# CLM / SkillRet development trial

Run: `clm_runpod_20260927T033653888785Z`. This is an additional development experiment, not the final course benchmark.

## Actually tested

The existing metadata BM25 implementation ran on the same fixed 200 pilot queries and all 6,006 skills. Every top-five result matches the saved metadata baseline. The final reserved query split was not used for model inputs or evaluation.

| Measurement | Result |
|---|---|
| BM25 Recall@5, all 200 | 60.5833% |
| BM25 candidate Recall@20, all 200 | 72.7500% |
| Queries missing any / all gold in BM25 top20 | 80 / 32 |
| CLM queries attempted | 200/200 |
| BM25→CLM Recall@5, attempted queries | 38.1667% |
| Absolute recall change | -22.4167% |
| Paired bootstrap 95% interval for change | -28.9167% to -15.9146% |
| CLM complete@5 / precision@5 / set F1 | 25.5000% / 11.2000% / 16.8095% |
| Improved / worsened / unchanged | 14 / 76 / 110 |
| CLM failures | 0 |
| Median / p95 CLM seconds | 0.27934269281104207 / 0.4476817158516496 |
| Pod created | True |
| Pod termination verified | True |
| GPU / VRAM | NVIDIA A40 / 44.42 GiB |
| Peak sampled GPU memory | 36661.0 MiB |
| BF16 runtime evidence | True |
| Billing smoke top prediction | technical (expected billing; incorrect predictions retained) |
| Reported GPU price/hour | US$0.49 |
| Pod elapsed seconds / approximate compute USD | 698.4723899364471 / 0.09506985307468308 |
| Approximate compute USD including related attempts | 0.1799866181254387 |

Status: **COMPLETE**.


The Mac has Apple Silicon and 16 GiB unified memory. It supports preparation, existing BM25, evaluation, tests and packaging. The requested Linux/NVIDIA vLLM BF16 configuration is unavailable on this Mac. No quantized substitute is used.

## Pinned official configuration; runtime claims require remote evidence

| Component | Pin |
|---|---|
| CLM repository | `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7` |
| CLM-v0.1-8B checkpoint revision | `e939398d4556fcd9400c76fa8c5a513202f42b0a` |
| Qwen/Qwen3-8B revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| vLLM package | `0.30.0` |
| Encoder settings | BF16, LAST pooling, prefix caching, 2,048 tokens, no quantization or CPU offload |
| Head and action cache | CPU, 512 MiB; fresh CLM process before the primary trial |

Runtime evidence: encoder and CLM ran; inspect the saved smoke results, exact commands, runtime verification and server logs. GPU residency is supported by the full unquantized weight load, zero CPU-offload setting, CUDA startup logs and sampled GPU memory when those artifacts exist. The runner records each attempt once and preserves errors; it never retries a ranking to obtain a preferred answer.

The runner refreshes Runpod prices before creation and records the actual choice in `runpod-selection.json`. Cost is an elapsed-time estimate, not an invoice; storage is reserved conservatively and is not measured separately. The table above uses this run's evidence rather than the earlier preparation snapshot.

Related attempts are preserved in `related_attempts.json` when present. The first live attempt stopped because an overly strict harness gate treated a valid but incorrect smoke prediction as a service failure. Its pod was deleted and verified absent. The correction records semantic smoke accuracy without blocking the benchmark; it does not alter model weights, precision, prompts or query inputs. See `evidence/clm_sources/smoke-gate-correction.json` and the saved integration diff.

| Candidates | Cold action seconds | Warm median seconds | Warm p95 seconds |
|---|---|---|---|
| 10 | 0.5360 | 0.1020 | 0.1478 |
| 100 | 1.0324 | 0.0540 | 0.0611 |
| 1000 | 9.5276 | 0.1634 | 0.1796 |

The cache experiment includes only one cold-action request per size; its warm distribution uses 20 different query states. Any speed difference includes CLM embedding/projection reuse and retained vLLM prefix caching, so it is not an isolated measurement of only one cache layer.

## Protocol and comparison with Jev

Dataset revision: `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`. Candidate strings are the existing BGE-capped metadata, reused byte-for-byte. BM25 supplies 20 candidates; CLM ranks them and the local evaluator takes five. Candidate IDs stay in an external mapping. No qrels, gold labels, expected answer count or full skill body is sent to CLM. Candidate display order is independent of BM25 rank and relevance.

Recall divides recovered relevant IDs by all relevant IDs for each query, then macro-averages. Precision@5 divides by five. Complete@5 requires every relevant ID. Existing `reference_metrics.py` also supplies set precision/recall/F1. Attempted service/context failures score zero; unrun queries are counted and excluded. The report preserves full baseline denominators and the attempted paired denominator separately. Bootstrap intervals resample paired per-query recall differences with seed 42; they are not independent-sample tests.

Failure labels are multi-label: one multi-positive query can gain one target, lose another and also have an upstream missing target. Missing-any and missing-all candidate-gold counts are distinct. No gold is used to form the primary BM25 candidate set.

CLM uses separate state/action embeddings with learned projection heads. Reused candidate vectors can reduce encoder work, which the cache experiment measures. The running Jev test and its output path were not located in this checkout. No Jev experiment was restarted or altered, and no CLM-versus-Jev superiority claim is supported. A controlled comparison must match query IDs, dataset revision, candidate memberships, metadata and top-k, and preserve both methods' failures. Compare by `query_id` after those protocol checks.

## Adaptations and execution status

- Upstream CLM defaults to truncation. The runner passes `--max-tokens 0` and records overlength inputs as explicit exclusions at 2,048 tokens.
- Only SSH port 22 is exposed. CLM 8700 and vLLM 8090 bind to localhost.
- `/v1/rank` exposes ranking and latency but not input-token usage. Cache health snapshots are preserved; unavailable metrics are not inferred.
- Cache trials restart CLM between sizes, retaining the vLLM prefix cache. One cold-action request is followed by 20 different-state warm-action requests. This measures action reuse rather than a completely cold encoder.
- Direct CLM N=50/250/1000 and vector→CLM are **not run or added**. They remain secondary to completing the primary trial.
- Local offline tests validate code, not live CLM operation. See `evidence/clm-tests.json` and its log. Live evidence is reported separately above and in `remote/out/`.

## Cost and cleanup

The CLM-specific persistent ledger reserves both USD and estimated CAD before creation, targets under US$3, and caps cumulative reservations at US$5 / CAD$7.50 using the explicit 1.50 planning conversion. Previous OpenRouter/Jev ledgers are untouched. The default time limit is at most 90 minutes, with price/storage/cleanup reserves. There is no network volume.

The runner retrieves evidence throughout the run and in `finally`, deletes only its attributed pod, and verifies deletion with a fresh API read. An independent local watchdog attempts deletion at the deadline if the runner dies. Current Runpod v2 and CLI 2.14 do not expose the scheduled-termination flag shown in the older guide. This guard therefore requires the Mac to remain online; it is not a provider-enforced spending cap. Keep the runner active until `cleanup.json` says `terminated: true`. A failed deletion is an error, never reported as successful cleanup.

## Reproduce

Set `RUNPOD_API_KEY` privately in the project `.env`; keep `ALLOW_PAID_RUNS=1`. Do not paste credentials into chat. From the project folder:

```bash
.venv/bin/python -m pytest -q tests/test_clm_reranker.py tests/test_reference_helpers.py
.venv/bin/python scripts/run_clm_trial.py --backend auto
```

For a free local preparation-only check: `.venv/bin/python scripts/run_clm_trial.py --prepare-only`.
Each invocation makes a new result directory. Completed or failed cases are not silently retried or overwritten. The key stays local; the remote transfer excludes evaluation labels, `.env`, credentials and unrelated work.

## Recommendation

The measured paired recall change is -22.42 percentage points. This configuration does not improve the primary recall metric; keep it as a documented negative development result rather than a default replacement. This does not establish superiority to Jev or generalize beyond this development sample. Inspect the paired confidence interval and failures before drawing a stronger conclusion.

Official sources: [CLM code and API](https://github.com/Contrastive-LM/CLM/tree/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7), [checkpoint](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/tree/e939398d4556fcd9400c76fa8c5a513202f42b0a), [encoder](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218), [Runpod v2 contract](https://api.runpod.io/v2/openapi.json).
