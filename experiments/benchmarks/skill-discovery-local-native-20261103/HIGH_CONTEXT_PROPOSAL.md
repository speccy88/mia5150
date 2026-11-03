# Proposal for measurements at library sizes 1,000 and above

Prepared at 2026-11-03T09:52:40.072379+00:00. **Proposal only: this document initiates no calls, larger-context loads or new downloads. Execution requires a separate decision.**

The existing 32K series remains unchanged. Its 60 exclusions per model come from the estimated gate, without tokenizing N=1,000/2,000/5,000 inputs. They establish neither the native model limit nor the maximum usable VRAM context.

## Proposed steps

1. In a new directory/version, prepare the same 20 prompts at each size with unchanged candidates, seed and descriptions. Use the actual template and tokenizer of the same runtime to count each input without generation. Tokenization can use an admitted 32K load: tokenizing a string does not require a KV cache of the same length. For models already rejected at 32K loading, consider vocabulary-only tokenization without executing layers; do not force an oversized GPU load.
2. Choose the smallest context within the official window that covers actual inputs plus 500 output tokens. In this separate version, the approximate 44.5*N+200 filter is replaced by the exact reserve gate. A size might fit within 32K or 40,960 after counting; do not assume 64K is required.
3. Load one model at a time. Preserve weight quantization, f16 KV, transformer/output layers on CUDA, default CPU_Mapped input weights, temperature 0, seed 42, a 500-token cap, one slot and fit=off. Reject declared CUDA-buffer totals above 8,192 MiB before generation and leave room for the driver and transient allocations. Also inspect observed dedicated-memory usage and WDDM messages; successful cudaMalloc does not prove residency.
4. Start with N=1,000, then N=2,000 if native and hardware limits allow. N=5,000 is a separate phase after preflight. Save each response and intent transactionally; a hardware error stops the model, and an uncertain request is never replayed automatically.

Indicative context tiers 65,536 / 131,072 / 262,144 correspond to historical estimates 44.700 / 89.200 / 222.700, not actual token counts. LFM3 is restricted here to its 128,000 text configuration; proposed tiers are adjusted to that window.

## Model-by-model review

| Model | Published window / text config / GGUF | Declared CUDA at 32K (MiB) | f16 projection at 64K / 128K / 256K (MiB) | Initial proposal |
|---|---|---:|---|---|
| Qwen3-0.6B | 32,768 advertised; config/GGUF 40,960 | 4202.65 | outside window / outside window / outside window | Tokenize N=1,000; check 32K/40,960. A larger-context extension is separate, not implicit. |
| Qwen3-1.7B | 32,768 advertised; config/GGUF 40,960 | 5348.77 | outside window / outside window / outside window | Tokenize N=1,000; check 32K/40,960. A larger-context extension is separate, not implicit. |
| Qwen3.5-4B | 262144; extension 1010000 separate | 3616.2 | 4640 / 6688 / 10784 >8 GB | Preflight N=1 000, then 2 000; confirm actual allocation before calls. |
| Ling-3.0-tiny | 131072; extension 262144 separate | 4728.97 | 4945 / 5377 / outside window | Preflight N=1 000, then 2 000; confirm actual allocation before calls. |
| Qwen3.5-9B | 262144; extension 1010000 separate | 5920.51 | 6945 / 8993 >8 GB / 13089 >8 GB | Preflight N=1 000; confirm actual allocation before calls. |
| Llama 3.2 3B | not inspected: gated access | - | unavailable | Wait for official acceptance and authenticated access; no substitute. |
| Gemma 3 4B | not inspected: gated access | - | unavailable | Wait for official acceptance and authenticated access; no substitute. |
| H2O-Lightning-4B | 262144 | 3673.62 | 4698 / 6746 / 10842 >8 GB | Preflight N=1 000, then 2 000; confirm actual allocation before calls. |
| Gemma 4 E2B | 131072 | 1580.72 | 1782 / 2184 / outside window | Preflight N=1 000, then 2 000; confirm actual allocation before calls. |
| Qwen3-8B | 40960 | 9097.34 | outside window / outside window / outside window | The 32K f16 profile is already too large; no larger context with this profile. |
| NeoHorse-1-9B | 262144 | 5920.51 | 6945 / 8993 >8 GB / 13089 >8 GB | Preflight N=1 000; confirm actual allocation before calls. |
| Ornith-1.5-9B | 262144 | 5920.51 | 6945 / 8993 >8 GB / 13089 >8 GB | Preflight N=1 000; confirm actual allocation before calls. |
| Nemotron 3.5 Bolt E4B | 65536 | 3604.99 | 4147 / outside window / outside window | Preflight N=1 000, then 2 000; confirm actual allocation before calls. |
| Ternary Bonsai 2 9B | 262144 | 3263.12 | 4287 / 6335 / 10431 >8 GB | Preflight N=1 000, then 2 000; confirm actual allocation before calls. |
| LFM3-Omni-2B | 128000 | 1453.86 | 1843 / outside window / outside window | Preflight N=1 000, then 2 000; confirm actual allocation before calls. |
| Phi-4 Mini 3.8B | 131072 | 6492.44 | 10588 >8 GB / 18780 >8 GB / outside window | Tokenizer preflight at 32K; the f16 projection selects no larger tier. |
| Mistral 7B Instruct | 32768 | 8230.52 | outside window / outside window / outside window | The 32K f16 profile is already too large; no larger context with this profile. |
| Nanbeige 4.2 3B | 262144 | 7839.7 | 13472 >8 GB / 24736 >8 GB / 47264 >8 GB | Tokenizer preflight at 32K; the f16 projection selects no larger tier. |

Projection = observed 32K CUDA allocation + observed KV × (future context / 32,768 − 1). This is a planning approximation, not a measurement, admission guarantee or maximum VRAM estimate. Sliding-window caches, recurrent state, compute graphs and transient allocations can scale differently. No larger context was tested in this series.

## Profiles requiring separate consideration

- KV-cache quantization, a smaller Qwen3-8B/Mistral context, different weight quantization, disabling thinking or raising the generation cap changes the profile. Each requires a new directory and explicitly separate comparison.
- YaRN/RoPE extensions, including Ling beyond 131,072, require the official recipe and separate validation. A large published window does not guarantee that it fits in 8 GB.
- Keeping the sampler preserves the existing comparison but keeps all golds at positions 1–20. Testing distributed positions requires a separate approved protocol, which must not replace or be mixed into this series.

Inspectable sources: `evidence/context-window-audit.json`, `evidence/native-buffer-audit.json`, `evidence/model-artifact-provenance.json` and pinned public-source records under `evidence/sources/`. Text limits take precedence over secondary vision/audio fields.
