# Admission against physical capacity

The first Qwen3-8B load declared 4,455.34 MiB of CUDA model weights, 4,608 MiB of
f16 KV cache and 34 MiB of compute buffers: 9,097.34 MiB for an 8,192 MiB card.
WDDM accepted the allocation. The 21 acquired responses remain a separate
diagnostic; interrupted intent `100:1` has an uncertain outcome and is not replayed.
No measured rate from this diagnostic enters the canonical comparison.

Before generation, the launcher now rejects declared CUDA budgets exceeding
physical VRAM with `HARDWARE_VRAM_CAPACITY_EXCEEDED`. This does not mean that
cudaMalloc returned OOM. An actually observed OOM retains its own error type.

The first admission-only guard probe had a PowerShell OrderedDictionary summation
bug: it returned zero despite three correct records. It performed no inference.
Its evidence is preserved as `gpu-proof-diagnostic-invalid-sum.json` and explicitly
invalidated in `invalidated-guard-probe-proof.json`. Correct direct-index summation,
rejecting zero or unverified totals, rejected the same model at 9,097.34 MiB.
Rejection logs and JSON are preserved.

Subsequent models pass this check before generation. Completed models are audited
offline from sealed startup prefixes; their declared budgets are below physical
capacity. Allocation remains distinct from actual residency. Dedicated/shared
WDDM counters were collected starting with Ornith, verifying PID, time and
executable. Nonzero shared usage alone cannot locate a paged CUDA buffer.

Observed transformer/output layers and KV/recurrent states are CUDA. The
CPU_Mapped input/embedding buffer is the runtime default, documented separately;
this profile is not described as absolutely GPU-only. The 32K context and other
parameters of this historical suite remain unchanged.
