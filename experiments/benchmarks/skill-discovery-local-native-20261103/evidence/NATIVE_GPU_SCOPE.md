# Exact hardware scope of the prepared profile

Launcher evidence verifies all transformer and output layers on GPU, the KV cache
on CUDA and the effective context. Verified hybrid-model logs also place recurrent
RS buffers on CUDA. The launcher permits neither auto-fit nor transformer-layer
fallback.

The buffer audit nevertheless shows that upstream llama.cpp b11960 keeps its input
layer on CPU by default. `src/llama-model.cpp` explains this choice and assigns
`dev_input` to the CPU device regardless of the requested GPU layer count. Logs
therefore show a CPU_Mapped model buffer alongside CUDA buffers, for example
545.62 MiB for Qwen3.5-9B.

This profile satisfies the prepared script's operational definition (transformer,
output and KV on CUDA), but it includes CPU participation. CPU_Mapped input/embedding
weights and CPU preparation are explicitly disclosed; the measurements describe
this exact profile.

Observed buffers and sizes are preserved in `native-buffer-audit.json`. Runtime,
parameters and acquired results were not changed during this audit. Forcing input
weights onto CUDA would create a separate profile requiring verification,
particularly of VRAM capacity; it must not be conflated with this study.
