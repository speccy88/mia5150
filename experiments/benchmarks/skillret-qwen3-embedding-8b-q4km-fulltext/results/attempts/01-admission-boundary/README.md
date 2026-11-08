# Admission-boundary feasibility attempt

This is a failed feasibility attempt, before corpus indexing or retrieval. It
used the required Q4_K_M model at 4,096 context, 4,096 dimensions, one input per
request and a 512-token internal microbatch. Two successful requests processed
117 and 1,500 content tokens. The 4,095-content-token probe returned HTTP 400
with truncate:false; no truncated embedding was accepted.

The native GGUF tokenizer adds one EOS; the Ollama admission check reserves two
slots. The primary protocol therefore uses at most 4,094 content tokens. This
changed chunk planning, with no source text removed. The retained initial
protocol and document manifest describe this attempt, not the primary run.

The first genuine unloaded-model request measured 65.854401473 seconds of runtime
loading and 119.252927463 seconds of total request latency. The final feasibility
receipt cites this original cold measurement separately from later warm loads.
No retrieval quality measurement was produced by this attempt.
