# Separate local context and KV profiles — results

Status: completed; verified 2026-11-03T11:32:21.712806+00:00. 200 measured responses, cap 200.

Initial 32K/f16 loads exceeded the 8,192 MiB capacity: 9,097.34 MiB of declared CUDA buffers for Qwen3-8B and 8,230.52 for Mistral v0.3. This supplement measures smaller f16 contexts and a separate Q8_0 cache condition while keeping the same weights.

The historical 32K/f16 series and archives are preserved. These measurements use a different context protocol: actual tokenizer preflight plus 500 tokens, rounded up to a multiple of 256. The 20 questions, candidates, prompts, seed 42, temperature 0, ranking rules and 500-token cap remain identical. Results are not merged with the native-v2 series.

N is the presented library size. Validated inputs contain 6,006 skills and 5,801 distinct names; the 20 original questions are reused. Prompts contain only names and descriptions capped at 300 characters, without full bodies. Sampling and scoring remain unchanged, including duplicate names and gold positions 1–20.

Explicit deviation: the original estimated gate `44.5×N+200 > 90% of n_ctx` is not applied in v3. The control becomes `actual input tokens + 500 ≤ n_ctx`, without an additional 10% margin. Contexts can be close to this exact reserve; each input is retokenized before generation and truncation is rejected. This supplement is therefore not an unchanged execution of the initial protocol.

Hardware: RTX 2060 SUPER, 8,192 MiB. Runtime: llama.cpp b11960 CUDA 12.4, commit `8642541b255bfb5dc39cd351a178be09013a55d6`. Q4_K_M weights are unchanged from the earlier series: [official Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B-GGUF) and a pinned local conversion of [official Mistral v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3). Weight, executable and sealed log-prefix hashes are linked to each launch. Q8_0 here refers to the KV cache, not a different weight quantization.

## Measurements by profile

| Model | Profile | N | n_ctx | KV | Responses | Top-1 | Top-10 | Without errors | Median LLM (s) | Issues |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| qwen3-8b-q4km | f16-adjusted-n50 | 50 | 2816 | f16 | 20 | 10/20 (50.00%) | 10.0/20 (50.00%) | 10 | 9.94 | OK=10, REASONING_NO_FINAL_ANSWER=10 |
| qwen3-8b-q4km | f16-adjusted-n100 | 100 | 5120 | f16 | 20 | 8/20 (40.00%) | 8.0/20 (40.00%) | 8 | 11.94 | OK=8, REASONING_NO_FINAL_ANSWER=12 |
| qwen3-8b-q4km | f16-adjusted-n250 | 250 | 12032 | f16 | 20 | 8/20 (40.00%) | 8.0/20 (40.00%) | 8 | 19.56 | REASONING_NO_FINAL_ANSWER=12, OK=8 |
| qwen3-8b-q4km | f16-adjusted-n500 | 500 | 23552 | f16 | 20 | 8/20 (40.00%) | 9.0/20 (45.00%) | 10 | 34.46 | REASONING_NO_FINAL_ANSWER=10, OK=10 |
| qwen3-8b-q4km | q8-adjusted-n500 | 500 | 23552 | q8_0 | 20 | 10/20 (50.00%) | 10.0/20 (50.00%) | 11 | 42.22 | OK=11, REASONING_NO_FINAL_ANSWER=8, EMPTY_OUTPUT=1 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n50 | 50 | 3328 | f16 | 20 | 16/20 (80.00%) | 20.0/20 (100.00%) | 20 | 5.85 | OK=20 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n100 | 100 | 5888 | f16 | 20 | 9/20 (45.00%) | 15.0/20 (75.00%) | 20 | 11.55 | OK=20 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n250 | 250 | 14080 | f16 | 20 | 8/20 (40.00%) | 12.0/20 (60.00%) | 19 | 20.08 | OK=19, INVALID_OUTPUT=1 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n500 | 500 | 27392 | f16 | 20 | 3/20 (15.00%) | 10.0/20 (50.00%) | 19 | 37.40 | OK=19, INVALID_OUTPUT=1 |
| mistral-7b-instruct-v03-q4km | q8-adjusted-n500 | 500 | 27392 | q8_0 | 20 | 3/20 (15.00%) | 8.0/20 (40.00%) | 18 | 46.02 | OK=18, INVALID_OUTPUT=2 |

## f16 totals across four sizes

| Model | Responses | Top-1 | Top-10 | Outputs without errors |
|---|---:|---:|---:|---:|
| mistral-7b-instruct-v03-q4km | 80 | 36/80 (45.00%) | 57.0/80 (71.25%) | 78 |
| qwen3-8b-q4km | 80 | 34/80 (42.50%) | 35.0/80 (43.75%) | 36 |

These totals combine the four adjusted f16 contexts; Q8_0 results remain in their separate paired comparison.

## Hardware admission

| Model | Profile | n_ctx | KV | Status | Declared CUDA MiB | Admission WDDM MiB | Maximum sampled WDDM MiB |
|---|---|---:|---|---|---:|---:|---:|
| qwen3-8b-q4km | preflight-f16-4096 | 4096 | f16 | completed_no_generation | 5058.34 | 5242.92 | - |
| qwen3-8b-q4km | q8-32768-admission-only | 32768 | q8_0 | completed_no_generation | 7049.42 | 7242.95 | 7242.95 |
| qwen3-8b-q4km | f16-adjusted-n50 | 2816 | f16 | completed | 4878.03 | 5062.91 | 5078.91 |
| qwen3-8b-q4km | f16-adjusted-n100 | 5120 | f16 | completed | 5202.59 | 5386.92 | 5402.92 |
| qwen3-8b-q4km | f16-adjusted-n250 | 12032 | f16 | completed | 6176.28 | 6362.93 | 6378.93 |
| qwen3-8b-q4km | f16-adjusted-n500 | 23552 | f16 | completed | 7799.09 | 7986.95 | 8002.95 |
| qwen3-8b-q4km | q8-adjusted-n500 | 23552 | q8_0 | completed | 6322.67 | 6510.93 | 6532.93 |
| mistral-7b-instruct-v03-q4km | preflight-f16-4096 | 4096 | f16 | completed_no_generation | 4639.52 | 4824.91 | - |
| mistral-7b-instruct-v03-q4km | q8-32768-admission-only | 32768 | q8_0 | completed_no_generation | 6419.60 | 6612.94 | 6612.94 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n50 | 3328 | f16 | completed | 4543.34 | 4726.91 | 4742.91 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n100 | 5888 | f16 | completed | 4863.96 | 5048.91 | 5064.91 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n250 | 14080 | f16 | completed | 5889.96 | 6076.93 | 6092.93 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n500 | 27392 | f16 | completed | 7557.21 | 7746.95 | 7762.95 |
| mistral-7b-instruct-v03-q4km | q8-adjusted-n500 | 27392 | q8_0 | completed | 6040.29 | 6230.94 | 6248.94 |

Admission-only loads and preflights perform no generation. K and V must have the requested types in the logs, Flash Attention must be enabled, and transformer/output layers and KV must remain on CUDA. Declared CUDA buffers and WDDM committed memory are checked before inference. Default CPU_Mapped input/embedding weights are explicitly retained; no transformer-layer fallback is allowed. These counters are distinct from an exhaustive residency trace.

Latencies include prefill and generation, with variable output lengths capped at 500 tokens. Weight loading and hashing are excluded from LLM time. They describe this Windows session with telemetry, without complete machine isolation.

## Paired quality comparison

| Model | N | Shared n_ctx | Paired questions | Top-1 f16 / Q8 | Top-10 f16 / Q8 | Identical raw outputs |
|---|---:|---:|---:|---|---|---:|
| qwen3-8b-q4km | 500 | 23552 | 20 | 8 / 10 | 9 / 10 | 0 |
| mistral-7b-instruct-v03-q4km | 500 | 27392 | 20 | 3 / 3 | 10 / 8 | 11 |

Q8_0 uses rounded quantization: 32 int8 values and an f16 scale, or 8.5 bits/value versus 16 for f16. The [initial llama.cpp experiments](https://github.com/ggml-org/llama.cpp/pull/7412) motivate this one quantized format here without guaranteeing equivalence. The same 20 questions at the same context permit a local comparison; this small set does not replace a general perplexity evaluation.

Outputs with no final answer at the 500-token cap are generation/format failures for this profile. Hardware rejections have no measured accuracy. The seed-42 gold-position bias at positions 1–20 remains. No N≥1000 context, new weights, other weight quantization or hosted inference is tested.

Evidence: `evidence/profile-review.json`, `evidence/runtime-kv-support.json`, `evidence/candidate-comparison.json`, preflights and historical runtime records; transactional SQLite and 45-column CSV files under `results/`. Raw outputs were hash-checked and remain private. The historical baseline-preservation records verify the earlier archives.

For Llama 3.2 and Gemma 3, see [closure of these two access paths](CLOSURE.md). Report generation performs no login, terms acceptance, publication or email delivery.


Final closure: Llama 3.2 and Gemma 3 are not pursued further, as requested by the user. No measured result is assigned to them.
