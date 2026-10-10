# EmbeddingGemma 2: bounded text retrieval pilot on RTX 2060 SUPER

**Date:** 6 October 2026, UTC. **Machine:** Elite-Windows11, existing Ubuntu-26.04 WSL2. Results independently audited at 21:11–21:14 UTC.

## Outcome

**EmbeddingGemma 2 successfully ran text-only FP32 inference on the RTX 2060 SUPER.** The successful run completed 431 single-item embedding calls, including compatibility checks, the exact prior bilingual retrieval fixture, and a small SkillRet candidate-ranking pilot. No resource guard stopped the successful run; no new swap activity occurred.

On the fixed 20-query/28-document fixture, EmbeddingGemma 2 achieved **Recall@1 95% (19/20), Recall@5 100%, MRR 0.975**, versus independently rechecked historical Qwen3-Embedding-0.6B Q8_0 results of **90%, 100%, and 0.950**. One additional top-ranked success on twenty hand-written cases is a smoke-test result, not evidence of broad superiority.

On the restricted SkillRet candidate set, EmbeddingGemma 2 improved Recall@5 over base Qwen, but remained below the existing SkillRet-tuned embedding control. Rank fusion with Qwen showed modest complementary value. **The result supports keeping it as a lightweight experimental channel, not replacing the tuned SkillRet retriever or launching a broad catalog sweep yet.**

## Official requirements and tested identity

The [official model card](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md) describes a modular approximately 740M-parameter model with a roughly 270M text-only configuration. It explicitly supports BF16/FP32 and warns against FP16 activations. The RTX 2060 SUPER is SM75, so this test explicitly used **FP32**, eager attention, and batch size one. It did not use BF16 emulation, FP16 activations, FlashAttention2, quantization, remote model code, or multimodal inputs.

The official repository was public and `gated=false`; no credentials or gated-term acceptance were needed. Model revision:

`914f7f89142e33e77833254d9c9b90c3cef7303b`

Official `model.safetensors`: **1,488,915,288 bytes**, SHA-256:

`197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`

The repository distributes one complete checkpoint. That file was downloaded and verified, but the official `config_kwargs={"vision_config": None, "audio_config": None}` configuration instantiated only **271,002,624 text-model parameters**. Both modality towers were asserted absent and all loaded parameters were asserted CUDA-resident FP32. All model/config/tokenizer files are pinned in `model-manifest.json`; large artifacts were checked against official LFS SHA-256 values. License metadata: Apache-2.0.

## Runtime and isolation

The existing SkillRet Python runtime was reused from `/home/user/code/mia5150-local/.venv`, including PyTorch **2.8.0+cu128**, without installing or upgrading packages there. New small dependencies went only into `/home/user/code/embeddinggemma2-pilot/overlay`, using the existing `uv` executable and hash-verified official PyPI wheels. SkillRet models, source, fixtures, and result files were only read.

Effective packages included Transformers **5.19.0**, SentenceTransformers **6.1.0**, huggingface-hub **1.33.0**, tokenizers **0.23.2**, Pillow **12.3.0**, torchvision **0.23.0**, and safetensors **0.8.0**. Torchvision 0.23.0 matches the reused PyTorch 2.8 runtime. The complete effective dependency graph and wheel manifest are retained.

Transformers wheel SHA-256:

`afcd2dd5f603ed28c1e1fcb00a338ccbb4ef5f878ed289635df8b58187afb518`

SentenceTransformers wheel SHA-256:

`eb8122f4d180f552eda26dc3d77e84e8c11dc2b1d456a406b9f24abb70ceeadd`

These released wheel identities describe the runtime; no source checkout commit is claimed. The native packaged model implementation was inspected before execution. `trust_remote_code=False` and offline model loading were used.

Hardware was the i5-9500T with approximately 16 GB host RAM and an RTX 2060 SUPER with 8,192 MiB VRAM. Ubuntu reported 26.04.1 LTS and kernel `6.18.40.1-microsoft-standard-WSL2`. Existing WSL limits remained 6 GB RAM, two virtual CPUs and 2 GB swap. No display driver, security/network setting, or existing service was changed.

The existing guard policy was retained: stop below 2 GiB host or guest available RAM, on any new swap activity, after five consecutive host-CPU samples above 80%, above 80 C GPU temperature, above 7,168 MiB GPU memory, or on stale/failed telemetry. Workloads ran sequentially with niceness 10, one configured intra-op thread and one inter-op thread, and a 60-second per-request limit.

## Protocol and prefixes

The official SentenceTransformer pipeline was used: projected token embeddings, mean pooling **including prompt tokens**, and L2 normalization to **768 dimensions**. No Matryoshka dimension truncation was tested.

Exact prefixes:

- Queries: `task: search result | query: ` (`SearchQuery`).
- Documents: `title: none | text: ` (`Document`).

The previous SkillRet `name | description` document text was retained unchanged as the document content. Skill names therefore remained inside that content; this pilot did not separately tune the model's real-title formatting or task prefix.

The configured input limit was **512 tokens**, below the advertised 8K capacity, to keep this first experiment bounded. Installed library source confirms truncation is enabled and the maximum sequence length controls the tokenizer. All small-fixture inputs and all 270 SkillRet documents fit. One SkillRet query, `q-02348`, had **836 raw tokens** and was truncated on both repetitions. The cached Qwen control used a 1,024-token cap; its longest selected query was 801 tokens and none was truncated. This is a known comparison asymmetry, not a hidden equivalence claim.

The raw-count diagnostic emitted a length warning for that query; actual encoding used the configured truncation policy and completed with finite normalized vectors. No indexing error occurred.

The smoke test embedded a northern-lights query, relevant explanation, and unrelated bicycle passage three times. All vectors were finite, 768-dimensional, unit-normalized, and the relevant passage scored above the unrelated passage each time. Only after those assertions passed did the same process continue to the retrieval pilots.

## Fixed bilingual fixture

The exact prior **20 queries and 28 documents**, including their original relevant-document indices, were copied and hashed. No cases were rewritten for the new model. Documents were encoded once and all queries three times. All three query embedding arrays were numerically identical; rankings were identical.

| Metric | EmbeddingGemma 2 FP32 | Historical Qwen Q8_0 |
|---|---:|---:|
| Recall@1 | **95% (19/20)** | 90% (18/20) |
| Recall@5 | 100% | 100% |
| MRR | **0.975** | 0.950 |

The Qwen control was recalculated from its saved vectors, not copied uncritically from a report. Its official GGUF revision was `370f27d7550e0def9b39c1f16d3fbaa13aa67728`; Q8_0 artifact SHA-256 was `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`. It used 1,024-dimensional embeddings and its own prescribed query instruction. This historical control uses a different runtime and precision; it was not rerun for a contemporaneous speed comparison.

## SkillRet candidate complementarity

The source corpus contains **6,006 skills**. This experiment selected **32 queries** from the existing frozen 128-query scorer set using SHA-256 ordering with the fixed prefix `eg2-pilot-v1:`. Candidate documents were the union of each selected query's cached Qwen top five, cached BM25 top five, and 32 hash-selected distractors, yielding **270 distinct documents**. Selection used no relevance labels. All outputs were written into the new pilot directory.

Relevance labels were opened for scoring after selection. The candidate pool contains **34 of 60 relevant query/document pairs**. Using the original per-query relevance denominators, its **maximum possible macro Recall is 56.77%**. No missing relevant documents were inserted. The experiment therefore measures ranking/complementarity inside an existing candidate set, not new full-catalog candidate discovery.

Cached Qwen and SkillRet-tuned document/query vectors were scored against exactly the same pool. Cache hashes, source corpus/query hashes, and row alignment were checked. Qwen's recalculated top five matched its original cached top five for **all 32 queries**. EmbeddingGemma 2 encoded the 270 documents once and the queries twice; repeat vectors and rankings were identical.

| Shared-pool system | Macro Recall@1 | Macro Recall@5 | Macro Recall@10 | Completeness@5 |
|---|---:|---:|---:|---:|
| Base Qwen embedding | 34.38% | 44.27% | 53.65% | 31.25% |
| Existing SkillRet-tuned embedding | **43.75%** | **56.77%** | 56.77% | **43.75%** |
| EmbeddingGemma 2 | 39.06% | 52.08% | 56.77% | 34.38% |
| Qwen + EmbeddingGemma 2, RRF | 38.02% | 55.73% | 56.77% | 40.63% |

Recall is macro-averaged over all 32 queries, with no dropped failures. Completeness means every known relevant document for a query appears in the top five. RRF used equal channel weights, rank constant 60, and deterministic ID tie-breaking, without fitting on labels.

At top five, EmbeddingGemma 2 found **eight relevant query/document pairs missed by Qwen**, while losing **two that Qwen found**. It improved per-query recall on seven queries, worsened it on one, and tied on 24.

The union of both top-five lists, with a maximum budget of ten, reached **54.69% recall**, versus **53.65% for Qwen's own top ten**. At a maximum budget of twenty, the union of the two top-ten lists and Qwen's top twenty both reached the pool ceiling. This equal-budget comparison shows the complementary gain is modest. The existing tuned SkillRet model already reached the pool ceiling at five.

These public benchmark queries are not a blind confirmation set. The small pool is built partly from Qwen/BM25 candidates and does not represent full-catalog difficulty. No claim of general SkillRet superiority, replacement readiness, or full-catalog recall is justified.

## Latency, CPU cost, and memory

Import took **5.598 seconds wall / 5.340 seconds process CPU**. Model-and-processor load took **4.018 seconds wall / 4.467 seconds process CPU**. The first embedding call took **0.913 seconds wall / 0.777 seconds CPU**. These are observed process-start/load components, not controlled cold-disk measurements.

Timers cover actual encoding, tokenization within the encoder, GPU execution, result transfer to NumPy, and CUDA synchronization. Separate raw-token counting, file writes, and audit/scoring work are excluded from per-call latency.

| Warm phase | Calls | Median wall | P95 wall | Total wall | Process CPU |
|---|---:|---:|---:|---:|---:|
| Small-fixture documents | 28 | 69.91 ms | 86.60 ms | 2.011 s | 2.006 s |
| Small-fixture queries, 3 repeats | 60 | 69.11 ms | 78.67 ms | 4.208 s | 4.197 s |
| SkillRet documents | 270 | 70.19 ms | 90.09 ms | 19.652 s | 19.597 s |
| SkillRet queries, 2 repeats | 64 | 72.78 ms | 93.11 ms | 4.885 s | 4.879 s |

All **431 calls**, including the nine smoke calls, totaled **32.224 seconds timed wall** and **32.009 seconds process CPU**. CUDA residency does not make this configuration CPU-free: it consumed approximately one CPU core during the sequential encode loop. Ancillary runtime threads existed despite one configured intra/inter-op thread. Host CPU sampling includes other machine activity and is not attributed entirely to the model.

Successful-run resource telemetry, sampled approximately once per second:

- GPU-memory peak: **1,257 MiB**.
- GPU-utilization peak: **44%**.
- GPU-temperature maximum: **54 C**; CPU temperature unavailable.
- Worker RSS peak: **1.6483 GiB**.
- Minimum Windows available RAM: **4.1990 GiB**.
- Minimum guest available RAM: **3.8168 GiB**.
- Maximum sampled host CPU: **49.23%**.
- **Zero new swap pages, zero guard stops**; 50 telemetry samples, 43 with worker data.

PyTorch recorded **1,169,106,432 allocated bytes** and **1,199,570,944 reserved bytes** at peak. These differ in scope from `nvidia-smi`. CUDA parameter residency, loaded CUDA libraries, and GPU telemetry provide execution evidence. Subsecond memory/temperature peaks may exceed sampled values.

## Failures, unrun stages, and reproducibility

Three resolved preparation issues remain in the evidence:

1. The small dependency resolver initially rejected an old nonstandard PyPI version string. It was corrected to skip invalid versions.
2. The first SentenceTransformer load required Pillow, even with both modality encoders disabled.
3. After Pillow was supplied, the processor required torchvision. Version 0.23.0 was installed into the isolated overlay, matching existing Torch 2.8.

The two processor-load failures occurred before embeddings were produced. Their model-weight loads are not counted as successful inference or included in the successful latency table. No guard threshold was relaxed and no PyTorch replacement was required.

Unrun: model training/adaptation, full-catalog EmbeddingGemma encoding, long-context sweeps, large batches, BF16/FP16 trials, multimodal inference, other attention backends, MRL dimension experiments, title/prefix tuning, and Jina text-nano. No credentials, paid services, external servers, or gated terms were involved.

Exact successful launch, with the supplied Windows `eg2-host.jsonl` monitor active:

```bash
/home/user/code/frognano-benchmark/.venv/bin/python \
  /home/user/code/embeddinggemma2-pilot/eg2_guard.py \
  evaluation-compatible \
  /home/user/code/mia5150-local/.venv/bin/python \
  /home/user/code/embeddinggemma2-pilot/eg2_eval.py
```

Core model configuration:

```python
SentenceTransformer(
    local_model_path,
    local_files_only=True,
    trust_remote_code=False,
    config_kwargs={"vision_config": None, "audio_config": None},
    model_kwargs={
        "dtype": torch.float32,
        "attn_implementation": "eager",
        "device_map": "cuda",
    },
)
model.max_seq_length = 512
# encode: batch_size=1, normalize_embeddings=True,
# prompt_name="SearchQuery" or "Document"
```

Local deliverables:

- Report: `C:\Users\user\Documents\Codex\2026-10-05\task\EmbeddingGemma2-RTX2060S-Report.md`
- Raw evidence: `C:\Users\user\Documents\Codex\2026-10-05\task\embeddinggemma2-results\`
- Host telemetry: `C:\Users\user\Documents\Codex\2026-10-05\task\eg2-host.jsonl`
- Retained model/runtime: `/home/user/code/embeddinggemma2-pilot/`

Raw evidence includes pinned manifests, dependency hashes, reviewed source, frozen fixtures, cached-control score matrices, saved embeddings, per-call wall/CPU/token records, complete rankings, independent metrics, resource telemetry, and failed-stage logs. Downloaded assets remain available for reuse. The parent task handles English Notion publication.

**Compute released at 21:16 UTC.** The successful worker exited with code zero; all task-owned setup/download/evaluation processes and the Windows monitor stopped. Final NVIDIA reading: driver 610.88, **0 MiB used, 0% utilization, 45 C**. The final host sample showed 9,443,422,208 available bytes. No authorization blocker remains for the completed pilot.
