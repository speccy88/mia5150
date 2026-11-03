# Distinct local profile v3
Protocol: v3-local-context-kv-real-token-gate; sizes: [100].
Context 5120; KV f16; weight quantization Q4_K_M.
Same supplied twenty questions, seed42 sampler, name/description[:300], top10, T0, seed42, output500.
The old estimated context gate is replaced only in this new profile by real tokenizer input +500.
No truncation, context shift, automatic fitting or transformer CPU fallback.
Default input embeddings remain CPU_Mapped. Native parser v2 is unchanged.
Invalid final ranks stay in place; unfinished thinking is a generation/format failure.
Raw responses are private, transactional intent precedes each trial, uncertain requests are not replayed.
The original 32K/f16 suite and archives remain unchanged.
