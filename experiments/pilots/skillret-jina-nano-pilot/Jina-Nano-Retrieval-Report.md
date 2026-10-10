# Jina v5 text-nano: bounded local retrieval pilot

**Date:** 6 October 2026, UTC. **Machine:** Elite-Windows11, RTX 2060 SUPER, existing Ubuntu-26.04 WSL2. This report includes the separately requested one-query EmbeddingGemma 2 sensitivity check.

## Outcome

**Jina v5 text-nano successfully completed 431 FP32 CUDA embedding calls with its retrieval adapter.** There were no truncated inputs, new swap pages, or resource-guard stops. Its warm SkillRet query median was **52.10 ms**, compared with the earlier EmbeddingGemma 2 configuration's **72.78 ms**. This is an observed comparison of two runtime configurations, not an isolated architecture-speed claim.

Jina matched the historical Qwen control on the small bilingual fixture: **18/20 top-ranked successes, Recall@5 100%, MRR 0.950**. EG2 had 19/20 and MRR 0.975.

In the fixed SkillRet candidate pool, Jina's Recall@5 was **48.44%**, above base Qwen's **44.27%**, but below EG2's **52.08%** and the existing SkillRet-tuned model's **56.77%**. Jina's complementary hits did not improve on simply giving Qwen the same maximum candidate budget. **This pilot does not support adding Jina as another SkillRet retrieval channel at present.** Its lower observed latency is useful, but the existing tuned model remains the stronger quality control in this restricted pool.

EG2's previously truncated query was re-encoded at a 1,024-token cap. A backbone hook confirmed all **836 tokens** were consumed on three repetitions. The vector changed, but the query's measured recall and all aggregate Recall/Completeness/MRR metrics stayed unchanged. Original EG2 results were preserved.

## License, source review, and identity

The [official repository](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano/tree/8a7f00aac812071b69403df470f1038ec85f8925) was public and ungated. Its license metadata is **CC-BY-NC-4.0**, a noncommercial license; this local evaluation does not establish permission for commercial deployment. No gated terms, credentials, paid API, or external service were used.

Pinned revision:

`8a7f00aac812071b69403df470f1038ec85f8925`

Base `model.safetensors`: **423,543,680 bytes**, SHA-256:

`f07de152e254b5d03b1273e0bca0a11dfade686508503500d460813428539651`

Retrieval `adapter_model.safetensors`: **13,586,992 bytes**, SHA-256:

`bc1afe96601bbb5198cd5553fa17d74bd77df3e30b515e50139980dd3a933cba`

All checkpoint/adapter artifacts were verified against official LFS SHA-256 values. Configuration, tokenizer, and code hashes are retained in the manifests. The official loader loads all four adapters; only **retrieval** was activated and tested. Total loaded parameters were **238,897,920**.

The repository uses custom code. Before execution, the pinned official loader, configuration, encoding path, imports, and file/network-sensitive operations were reviewed. Five Python files were hashed and checked again against the downloaded copies. The review found local adapter loading and standard Torch/Transformers/PEFT inference, with a `snapshot_download` branch used only when the model path is not local. Local paths and offline environment settings avoided that branch. No arbitrary execution or shell-operation calls were observed. This was a bounded source review, not a formal security audit.

`trust_remote_code=True` was used only to load those reviewed local files. Hugging Face home/module caches were isolated under the new pilot directory; runtime downloads were disabled. No unrecognized source was executed and no additional approval was required.

## Runtime and fixed protocol

No new packages were necessary. The existing Python environment supplied **Torch 2.8.0+cu128, Transformers 4.57.1, PEFT 0.21.2, huggingface-hub 0.36.2, and safetensors 0.8.0**. SkillRet's installed packages, models, fixtures, source, and results were not edited.

The RTX 2060 SUPER is SM75. The official loader defaults to FP32, and this run explicitly used **FP32, eager attention, one-item batches**, and one configured intra/inter-op thread. All parameters were verified on CUDA in FP32. No BF16 assumption, FP16 trial, FlashAttention2, quantization, training, or adapter merging was used.

Official retrieval formatting and behavior:

- `task="retrieval"` and active adapter `retrieval`.
- Query prefix: `Query: `.
- Document prefix: `Document: `.
- Last non-padding-token pooling, followed by L2 normalization.
- Full **768-dimensional** embeddings; no dimension truncation.
- Explicit **1,024-token cap**.

Actual forward input lengths were captured and checked against the tokenized lengths. The longest small-fixture query/document had 19 tokens; SkillRet documents reached 250 tokens and queries 771 tokens. **No Jina input was truncated.** Document content remained exactly the prior `name | description` representation.

The compatibility smoke encoded a northern-lights query, a relevant explanation, and an unrelated bicycle passage three times. All vectors were finite, unit-normalized and 768-dimensional, and the relevant passage scored higher on every repeat. Only then did the script continue to the fixed retrieval fixtures.

## Fixed 20-query/28-document result

The exact EG2/prior-Qwen fixture was reused. Documents were encoded once, queries three times. Query vectors were numerically identical across repetitions, and rankings were identical. Historical Qwen scores were independently recomputed from saved vectors.

| System | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|
| Historical Qwen3-Embedding-0.6B Q8_0 | 90% (18/20) | 100% | 0.950 |
| EmbeddingGemma 2 FP32 | 95% (19/20) | 100% | 0.975 |
| Jina v5 text-nano FP32 | 90% (18/20) | 100% | 0.950 |

These twenty hand-written bilingual cases are a smoke fixture, not a representative or statistically decisive model comparison. Equal aggregate metrics do not imply identical rankings or errors. Qwen was a historical control with a different runtime/precision, not a contemporaneous speed run.

## Fixed SkillRet candidate pool

The exact **32-query/270-document** EG2 pool was reused, with identical source hashes and cached controls. Queries had previously been selected by deterministic hashing from the frozen 128-query scorer set. Documents were the union of Qwen top five, BM25 top five, and hash-selected distractors. Relevance labels were not used to choose or add candidates.

The original corpus contains **6,006 skills**. The restricted pool includes **34 of 60 known relevant query/document pairs**, giving a **56.77% macro-recall ceiling** when the original relevance denominators are retained. Missing relevant documents remain missing. This experiment measures candidate ranking and complementarity, **not new candidate discovery or full-catalog retrieval**.

Jina encoded each document once and all queries twice. Repeat vectors and rankings were identical. Qwen and SkillRet-tuned controls were recomputed from existing vectors against the same pool. Qwen's top five still matched its original cached top five for all 32 queries.

| System in shared pool | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Completeness@5 |
|---|---:|---:|---:|---:|---:|
| Base Qwen | 34.38% | 44.27% | 53.65% | 56.77% | 31.25% |
| SkillRet-tuned embedding | **43.75%** | **56.77%** | **56.77%** | **56.77%** | **43.75%** |
| EG2, including full-query sensitivity | 39.06% | 52.08% | 56.77% | 56.77% | 34.38% |
| Jina v5 text-nano | 38.02% | 48.44% | 49.48% | 54.17% | 37.50% |
| Qwen + Jina RRF | 34.38% | 50.52% | 52.08% | 55.73% | 37.50% |

Recall is macro-averaged over all 32 queries. Completeness requires every known relevant document for a query to appear within the cutoff. RRF used equal weights, constant 60, and deterministic ID tie-breaking, with no label fitting.

At top five, Jina found **six relevant query/document pairs missed by Qwen**, but lost **three Qwen found**. Per-query recall improved on four queries, worsened on two, and tied on 26. The union of both top-five lists, with a maximum budget of ten, reached **52.60% recall**, below **53.65% for Qwen's own top ten**. The union of top-ten lists reached **54.69%** at a maximum budget of twenty, below **56.77% for Qwen's own top twenty**.

Although same-cutoff RRF improved Recall@5 over base Qwen, the equal-budget union comparison and the stronger tuned control provide no compelling reason to add another production retrieval channel from this sample. These are public benchmark queries in a Qwen/BM25-derived pool, not a blind confirmation set.

## EG2 truncation sensitivity

Only `q-02348` was re-encoded. Its original EG2 raw length was 836 tokens, previously capped at 512. The new run used the same FP32/eager model and prefix, raised the cap to 1,024, and reused all cached document vectors and the other 31 query vectors.

A corrected backbone pre-hook observed **[1, 836]** inputs on all three repetitions. Repeat vectors were identical. The cosine similarity between old and new query vectors was **0.976874**. Rankings beyond the top five changed, but the known relevant candidate remained first. Query Recall@1/5/10/20 stayed **1/3**, because its other two relevant documents were outside the candidate pool.

All aggregate EG2 Recall, Completeness and MRR values remained exactly unchanged: Recall@5 **52.08%**, Recall@10 **56.77%**, MRR **0.646875**. Original scores were preserved; the sensitivity array is a separate artifact.

The first shape-audit hook was attached to an outer layer that the library bypassed, yielding no shape observations. That instrumentation issue was corrected, and only the same query was repeated. The first inference result is retained but is not used as proof of actual input length. The verified run recorded a sampled **1,329 MiB GPU peak**, **51 C maximum GPU temperature**, **4.2508 GiB minimum host available RAM**, **3.8458 GiB minimum guest available RAM**, and zero swap/guard stops.

Jina's 1,024-token run, cached Qwen's 1,024-token run, and the corrected EG2 input now all contain the full selected queries, despite naturally different tokenizer counts. No full EG2 rerun was performed.

## Latency and actual CPU cost

Jina imports took **6.298 s wall / 4.677 s process CPU**. Model and adapter loading took **4.557 s wall / 4.120 s CPU**. The first embedding call took **0.460 s wall / 0.341 s CPU**. These are observed process-start components, not controlled cold-disk measurements.

| Warm phase | Calls | Median wall | P95 wall | Total wall | Process CPU |
|---|---:|---:|---:|---:|---:|
| Small-fixture documents | 28 | 50.24 ms | 76.40 ms | 1.524 s | 1.523 s |
| Small-fixture queries, 3 repeats | 60 | 49.67 ms | 64.91 ms | 3.165 s | 3.162 s |
| SkillRet documents | 270 | 52.26 ms | 75.66 ms | 15.317 s | 15.247 s |
| SkillRet queries, 2 repeats | 64 | 52.10 ms | 82.20 ms | 3.684 s | 3.678 s |

All **431 calls** totaled **24.569 s timed wall / 24.368 s process CPU**, including nine smoke calls. Timers include actual tokenization/encoding, GPU execution, CPU transfer and CUDA synchronization; separate raw-token counting, file writing and scoring are excluded.

Jina was faster than the earlier EG2 configuration on this workload, but still consumed approximately one CPU core during the sequential encode loop. CUDA inference does not imply negligible CPU use. Host CPU percentages also include background activity.

## Resource envelope

The machine remained the i5-9500T with approximately 16 GB Windows RAM, RTX 2060 SUPER 8 GB, and Ubuntu 26.04.1 under WSL2. Existing WSL limits remained 6 GB RAM, two virtual CPUs and 2 GB swap.

All prior guards remained unchanged: stop below 2 GiB host/guest available RAM, on any new swap, after five host-CPU samples above 80%, above 80 C GPU temperature or 7,168 MiB VRAM, or on stale/failed telemetry. Jobs ran sequentially, at niceness 10, with 60-second per-request limits.

Successful Jina telemetry:

- **1,183 MiB** sampled GPU-memory peak; **36%** GPU-utilization peak.
- **54 C** maximum GPU temperature; CPU temperature unavailable.
- **1.8743 GiB** peak worker RSS.
- **4.3384 GiB** minimum Windows available RAM.
- **3.7363 GiB** minimum guest available RAM.
- **44.10%** maximum sampled host CPU.
- **Zero new swap pages, zero guard stops**; 44 samples, 37 with worker data.

PyTorch peak allocated/reserved memory was **1,045,415,936 / 1,121,976,320 bytes**. These differ in scope from `nvidia-smi`; sampling can miss subsecond peaks. CUDA-resident parameters, loaded CUDA libraries and GPU telemetry are preserved as execution evidence.

## Reproduction, limitations, and artifacts

The successful Jina worker used:

```bash
/home/user/code/frognano-benchmark/.venv/bin/python \
  /home/user/code/jina-nano-pilot/jina_guard.py \
  evaluation \
  /home/user/code/mia5150-local/.venv/bin/python \
  /home/user/code/jina-nano-pilot/jina_eval.py
```

The dedicated Windows host monitor must be active. The loader reads the reviewed local model/config, sets eager attention and FP32, then calls:

```python
model.encode(
    texts=[text],
    task="retrieval",
    prompt_name="query",  # or "document"
    max_length=1024,
)
```

No Jina inference attempt failed. An initial read-only source-fetch shell command returned 404s and was replaced by a structured pinned downloader before any custom code ran. The separately described EG2 shape-hook instrumentation issue was corrected. No model/runtime dependency installation or guard relaxation was needed for Jina.

Unrun: full-catalog encoding, training/adaptation, other Jina tasks, larger batches, long-context sweeps, alternative precision/attention, dimension reduction, paid APIs, and additional models. No service or security setting was changed. The license and restricted-pool design limit conclusions about deployment and general retrieval quality.

Deliverables:

- `C:\Users\user\Documents\Codex\2026-10-05\task\Jina-Nano-Retrieval-Report.md`
- `C:\Users\user\Documents\Codex\2026-10-05\task\jina-nano-results\`
- `C:\Users\user\Documents\Codex\2026-10-05\task\jina-nano-host.jsonl`
- Retained WSL project/model: `/home/user/code/jina-nano-pilot/`

Raw evidence contains source review and hashes, model/adapter manifests, unchanged fixtures, embeddings, exact rankings, per-call wall/CPU/token records, independently recalculated metrics, telemetry, and both EG2 sensitivity attempts. The parent task handles English Notion publication.

**Compute released at 21:31 UTC.** Jina evaluation and the verified EG2 sensitivity worker both exited with code zero. All task-owned workloads and the Windows monitor stopped. Final GPU state was **0 MiB used, 0% utilization, 45 C**; the final Windows sample showed **9,506,607,104 bytes available**. Assets are retained for reuse. No further model was started and no authorization blocker remains for these completed tests.
