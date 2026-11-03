# Skill Discovery — lot local Windows

Status at 2026-11-03T04:54:01.513021+00:00.

This file supplies a section of the combined report. Generating it does not send messages or publish anything.

## Status of the 18 requested models

| Requested model | Status | Recorded rows | Measured responses | Outputs without errors | Top-1 / measured responses |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | complete export | 140 | 80 | 66 | 49/80 (61.3%) |
| Qwen3-1.7B | complete export; TRANSPORT_ERROR=1, GPU_SERVER_NOT_RUNNING=13 | 140 | 66 | 22 | 15/66 (22.7%) |
| Qwen3.5-4B | complete export | 140 | 80 | 64 | 0/80 (0.0%) |
| Ling-3.0-tiny | complete export | 140 | 80 | 39 | 0/80 (0.0%) |
| Qwen3.5-9B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Llama 3.2 3B | gated access: user acceptance required | 0 | 0 | 0 | 0/0 (—) |
| Gemma 3 4B | gated access: user acceptance required | 0 | 0 | 0 | 0/0 (—) |
| H2O-Lightning-4B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Gemma 4 E2B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Qwen3-8B | en attente | 0 | 0 | 0 | 0/0 (—) |
| NeoHorse-1-9B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Ornith-1.5-9B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Nemotron 3.5 Bolt E4B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Ternary Bonsai 2 9B | en attente | 0 | 0 | 0 | 0/0 (—) |
| LFM3-Omni-2B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Phi-4 Mini 3.8B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Mistral 7B Instruct | en attente | 0 | 0 | 0 | 0/0 (—) |
| Nanbeige 4.2 3B | en attente | 0 | 0 | 0 | 0/0 (—) |

A 140-row export records all outcomes; it does not mean 140 calls succeeded. Hardware failures, blocked access and context exclusions are not observed zero-accuracy results.

## Results by library size

| Model | N | Measured responses | Top-1 exact | Top-10 exact | Top-1 precision, non-error outputs | Erreurs/limites |
|---|---:|---:|---:|---:|---:|---|
| ling-3.0-tiny-q4km | 50 | 20 | 0.0% | 15.0% | 0.0% | INVALID_OUTPUT: 16, OK: 4 |
| ling-3.0-tiny-q4km | 100 | 20 | 0.0% | 55.0% | 0.0% | OK: 13, INVALID_OUTPUT: 7 |
| ling-3.0-tiny-q4km | 250 | 20 | 0.0% | 55.0% | 0.0% | OK: 14, INVALID_OUTPUT: 6 |
| ling-3.0-tiny-q4km | 500 | 20 | 0.0% | 35.0% | 0.0% | INVALID_OUTPUT: 12, OK: 8 |
| ling-3.0-tiny-q4km | 1000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| ling-3.0-tiny-q4km | 2000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| ling-3.0-tiny-q4km | 5000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3-0.6b-q8 | 50 | 20 | 70.0% | 80.0% | 82.4% | OK: 17, INVALID_OUTPUT: 3 |
| qwen3-0.6b-q8 | 100 | 20 | 60.0% | 75.0% | 66.7% | OK: 18, INVALID_OUTPUT: 2 |
| qwen3-0.6b-q8 | 250 | 20 | 50.0% | 60.0% | 71.4% | INVALID_OUTPUT: 6, OK: 14 |
| qwen3-0.6b-q8 | 500 | 20 | 65.0% | 70.0% | 76.5% | OK: 17, INVALID_OUTPUT: 3 |
| qwen3-0.6b-q8 | 1000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3-0.6b-q8 | 2000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3-0.6b-q8 | 5000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3-1.7b-q8 | 50 | 20 | 20.0% | 30.0% | 66.7% | INVALID_OUTPUT: 14, OK: 6 |
| qwen3-1.7b-q8 | 100 | 20 | 25.0% | 40.0% | 62.5% | INVALID_OUTPUT: 12, OK: 8 |
| qwen3-1.7b-q8 | 250 | 20 | 30.0% | 35.0% | 75.0% | OK: 8, INVALID_OUTPUT: 12 |
| qwen3-1.7b-q8 | 500 | 6 | 0.0% | 0.0% | — | INVALID_OUTPUT: 6, TRANSPORT_ERROR: 1, GPU_SERVER_NOT_RUNNING: 13 |
| qwen3-1.7b-q8 | 1000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3-1.7b-q8 | 2000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3-1.7b-q8 | 5000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3.5-4b-q4km | 50 | 20 | 0.0% | 10.0% | 0.0% | OK: 14, INVALID_OUTPUT: 6 |
| qwen3.5-4b-q4km | 100 | 20 | 0.0% | 75.0% | 0.0% | OK: 17, INVALID_OUTPUT: 3 |
| qwen3.5-4b-q4km | 250 | 20 | 0.0% | 60.0% | 0.0% | OK: 16, INVALID_OUTPUT: 4 |
| qwen3.5-4b-q4km | 500 | 20 | 0.0% | 75.0% | 0.0% | OK: 17, INVALID_OUTPUT: 3 |
| qwen3.5-4b-q4km | 1000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3.5-4b-q4km | 2000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3.5-4b-q4km | 5000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |

Measured rates use only rows with actual input and output token usage. Precision conditional on non-error outputs uses a different denominator. Invalid or duplicate ranks remain in place; removing them never promotes a correct answer.

## Protocol and interpretation-changing limitations

- Supplied inputs: 6,006 skills, 5,801 distinct names and 20 exact questions. Seven sizes: 50, 100, 250, 500, 1,000, 2,000 and 5.000. No deduplication.
- Each size restarts `random.Random(42).sample(all_skills, N)`. Every question keeps that library, replacing the last item only when its gold ID is absent. Prompts use only the name and first 300 description characters, without full skill bodies.
- **Verified position bias:** the 20 supplied gold IDs are already the first 20 seed-42 samples. They occupy positions 1–20 at every size; no final replacement occurs. The runner does not add a gold prefix. Examined local and Luna/Sol/Terra plans are identical. This series does not test uniformly distributed answer positions. A corrected variant needs a separate version and execution.
- The originally requested and verified server profile uses context 32,768, one slot, transformer/output layers and f16 KV on CUDA, fit=off, no context shift, warmup, RAM cache or checkpoints, and batch 512 / microbatch 128. Later hardware auditing clarifies that default CPU_Mapped input/embedding weights remain; see the final hardware-scope note. Transformer-layer CPU fallback is not allowed.
- Actual tokenization uses `/apply-template` then `/tokenize`, with IDs passed to `/completion`, temperature 0, seed 42 and a 500-token output cap; truncation is rejected. At this context, the estimated v1 gate `44.5*N+200 > 90% of context` excludes sizes 1,000/2,000/5,000, or 60 rows per model. The estimate is not an actual token count.
- Default model thinking is retained. Reasoning can consume the 500-token cap before the expected list; invalid outputs and outputs reaching the cap remain visible in the evidence.
- Available artifacts use different quantizations. Results compare those specific variants, not parameter counts alone.
- The hosted OAuth variant imposes neither temperature 0 nor a 500-token output limit. Identical candidates therefore do not make it strictly equivalent to local inference. The original reference runner was not supplied; exact reproduction of that execution is not claimed.
- Twenty questions are a small, partly lexical sample; duplicate names prevent exact identification of some IDs in a name-only response format.
- Timings describe this Windows environment with 16 GB RAM. Some downloads, installations and preparatory CPU conversions ran alongside model calls; these are not isolated speed measurements. Local service charges are USD 0, excluding electricity.

## Artifacts and identities

| Model | Quantization / provenance | Verified publisher source |
|---|---|---|
| Qwen3-0.6B | Q8_0 | [Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) |
| Qwen3-1.7B | Q8_0 | [Qwen/Qwen3-1.7B-GGUF](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF) |
| Qwen3.5-4B | Q4_K_M; existing Ollama weights, not a publisher-provided Qwen GGUF | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Ling-3.0-tiny | Q4_K_M | [inclusionAI/Ling-3.0-tiny-GGUF](https://huggingface.co/inclusionAI/Ling-3.0-tiny-GGUF) |
| Qwen3.5-9B | Q4_K_M; existing Ollama weights, not a publisher-provided Qwen GGUF | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Llama 3.2 3B | not downloaded: gated access | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Gemma 3 4B | not downloaded: gated access | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) |
| H2O-Lightning-4B | Q4_K_M, official stock ordinary-chat export | [h2oai/h2o-lightning-4b-gguf](https://huggingface.co/h2oai/h2o-lightning-4b-gguf) |
| Gemma 4 E2B | Official QAT Q4_0 | [google/gemma-4-E2B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf) |
| Qwen3-8B | Q4_K_M | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) |
| NeoHorse-1-9B | Q4_K_M | [TokenRhythm/NeoHorse-1-9B-GGUF](https://huggingface.co/TokenRhythm/NeoHorse-1-9B-GGUF) |
| Ornith-1.5-9B | Q4_K_M | [ornith-ai/Ornith-1.5-9B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF) |
| Nemotron 3.5 Bolt E4B | Q4_K_M | [nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB) |
| Ternary Bonsai 2 9B | Official ternary Q2_0 packing, not Q4 | [prism-ml/Ternary-Bonsai-2-9B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf) |
| LFM3-Omni-2B | Q4_K_M | [LiquidAI/LFM3-Omni-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF) |
| Phi-4 Mini 3.8B | Q4_K_M, local conversion of pinned official weights | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) |
| Mistral 7B Instruct | Q4_K_M, local conversion of pinned official weights | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| Nanbeige 4.2 3B | Q4_K_M, local conversion of pinned official weights | [Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) |

Source hashes, commits, sizes and licenses are recorded in the source inventories and weight provenance files. Installed Ling bytes exactly match the official file hash. Phi, Mistral v0.3 and Nanbeige were converted/quantized from pinned official sources. H2O uses its official stock ordinary-chat export, without requesting its typed-decision API. Multimodal models are tested as text only, without a projector.

## Validation and incidents

The 27 synthetic engine tests passed before inference. `review_local_results.py` recalculates metrics from ranks and checks CSV width, unique keys, per-size cardinality, token consistency, SQLite integrity, CSV/SQLite agreement and candidate/prompt plans. Its scope and timestamp are recorded in `evidence/result-review.json`.

Qwen3-1.7B incident: 66 measured responses; CUDA OOM during prefill of request 67 (`500:6`), followed by 13 `GPU_SERVER_NOT_RUNNING` rows and 60 estimated-context exclusions. The original CSV is retained and no call was replayed. These 14 hardware/runtime errors are excluded from measured-output rates; original incident and CUDA logs remain in the private source archive.

Historical launch logs, CUDA proofs, SQLite, CSV, manifests and hashes were retained. No OAuth token was copied into this local series. Benchmark execution itself sent no email or push. Weights, original source data, environments and downloads are excluded from the public repository; integration and report delivery are separate operations.
