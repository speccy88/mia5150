# Native window, configured window and hardware limit

The 60 `CONTEXT_LIMIT_ESTIMATED` rows per model come from the v1 gate against
**n_ctx configured at 32,768**, rather than a measurement of the native limit.

The threshold is 29,491.2 tokens (90% of 32,768). The v1 estimates for N=1000,
2000 and 5000 are 44,700, 89,200 and 222,700. These three sizes are excluded:
3 × 20 questions = 60 rows. Neither tokenization nor generation measures these
excluded sizes. Estimates are not recorded as actual token usage.

| Model | Published window / publisher configuration | Window actually loaded in this study |
|---|---:|---:|
| Qwen3-0.6B | README: 32,768; config/GGUF: 40,960 | 32,768 |
| Qwen3-1.7B | README: 32,768; config/GGUF: 40,960 | 32,768 |
| Qwen3.5-4B | 262,144 native; advertised extension to 1,010,000 | 32,768 |
| Qwen3.5-9B | 262,144 native; advertised extension to 1,010,000 | 32,768 |
| Ling-3.0-tiny | Native config: 131,072; YaRN recipe: 262,144 | 32,768 |

32K is a uniform, conservative choice prepared for the 8 GB machine, **without
searching for each model's hardware maximum**. It is neither a demonstrated VRAM
limit nor evidence that Qwen3.5 or Ling intrinsically cannot handle those 60 rows.
A larger native window also does not prove that it would fit alongside this
study's weights, f16 KV cache and buffers.

Publisher sources, commits, configuration limits, GGUF metadata and n_ctx observed
in launch evidence are distinguished in `context-window-audit.json`. Unretrieved
gated configurations and unloaded servers remain explicitly unobserved.

CSV columns and protocol error codes are unchanged. This audit distinguishes the
configured-profile estimation gate, actual tokenized overflow, runtime/VRAM
errors and published native limits.
