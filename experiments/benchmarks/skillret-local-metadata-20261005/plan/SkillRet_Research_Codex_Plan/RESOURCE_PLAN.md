# Resource allocation plan — CAD

No account access, GPU, paid API call, fine-tuning run or cloud instance was used or provisioned for this plan. Budget fields in the registry are null and paid operations default to disabled. The figures below are **suggested spending envelopes to approve**, not quotations or predicted bills.

## Allocation options

| Package | Proposed external-spend cap | Scope |
|---|---:|---|
| Initial screening | C$50 | Reproduction preflight and small development pilots; establish actual throughput and token distributions. |
| Balanced study | C$200 total | Strong reference pipeline, two candidate generators, competing reranker families, paired finalist evaluation where the measured budget permits. |
| Extended study | C$500 total | Add one targeted adaptation/distillation study, more independent-provider comparisons and a small transfer/execution panel. |

Do not infer that a cap can complete every proposed run. Reserve roughly 20% of an approved cap for confirmations and failures rather than spending it all selecting configurations. The original proposal's ~10 GPU-hours can be used as an initial **allocation limit**, not an execution-time prediction. Update caps only with explicit user allocation.

## Where the work belongs

| Environment | Proposed role | Boundary |
|---|---|---|
| Codex local/CPU | Audit, data manifests, lexical retrieval, metrics, cached-score fusion, reports | Inspect real RAM, disk, CPU threads and network; do not assume a GPU exists in Codex Cloud. |
| Fred's RTX 2060 Super 8 GB | Small encoders/rerankers; sequential quantized 4B deployment checks | JevK5/Plumb cards quote approximately 9–10 GB for BF16, so that mode is not an 8 GB target. Long-context/batching costs must be measured. |
| RunPod, initially one 24–48 GB GPU | 8B reference models, common-pool reranking, training feasibility | These are planning targets, not confirmed fits or available offers. Prefer 48 GB for adaptation headroom; choose only after a representative memory pilot. |
| Approved APIs | Hosted Jev, hosted reranking, economical and strong LLM comparisons | Verify credentials locally, without printing them; pin provider/model/version/tier and a hard spending limit. |

A GPU changes throughput, not necessarily ranking quality; precision, truncation and software kernels can change outputs. Report the actual deployment configuration, not just parameter count. Keep a higher-precision reference for any quantization study.

## Workload arithmetic

For all 4,392 official queries, a pointwise reranker requires 87,840 query-skill evaluations at K=20; 219,600 at K=50; 439,200 at K=100, before repeats and failures. Joint/listwise systems need different numbers of forward passes, but still process their supplied text. Batch size and concurrency must not be confused with fewer evaluated pairs.

The proposal's 2,000 × 300,000-token full-catalog scenario is 600 million input tokens. That arithmetic does not establish affordability: at a hypothetical **C$1 per million input tokens** it would already cost C$600 in input alone. Real rates differ by provider, context band, cache status and tier. Tokenize the actual catalog and query prompts before allocating a full-catalog run.

## Cost preflight required before paid execution

Use approximately 50 representative development queries covering the short/median/long document cases and multiple candidate lengths. First perform a no-network token/count estimate. Then request approval for a capped live pilot. Record warm/cold p50/p95, request count, all input/output/reasoning tokens, cache reads/writes, provider billing units, failures/retries, GPU-hours including idle/model-loading time, storage and egress. Use a safety margin for upper-tail requests.

Estimated run cost = measured per-query cost × planned query count × repeats + indexing + training + storage/idle/egress. For GPU time, estimate from total measured elapsed time, not a marketing tokens-per-second number from another architecture. Keep a per-stage ledger and stop at the approved cap.

Convert billing currency to CAD with a dated rate/source or a clearly hypothetical exchange-rate assumption. Save the underlying billing currency as evidence; do not silently convert using an old exchange rate from the proposal.

## API candidates and verification

The live official OpenAI catalogue inspected on 2026-10-05 lists `gpt-6-luna`, `gpt-6.1-sol`, and `gpt-6-astra`. Use Luna as an economical challenger, Sol as the initial strong challenger, and Astra only for a bounded marginal-quality test until results justify the premium. These are proposed roles, not claims of SkillRet superiority. Source: https://developers.openai.com/api/docs/models . Pricing must be captured again from https://developers.openai.com/api/docs/pricing at execution time; cached search results encountered during research showed older GPT-5.6 entries.

For TypeSafe Jev resolve a supported explicit version through its current docs/account, and inspect how state length and question count affect billing. For Cohere choose an exact Rerank v4.0 model after checking https://docs.cohere.com/v2/docs/rerank . Jina v3.5 is another documented candidate; its local weights have non-commercial terms. No provider access was verified here.

## RunPod safety and reproducibility

Do not assume prior browser login implies access in this Codex session. Use an authorized connector, API key or SSH configuration supplied through a secret mechanism. Do not scrape browser credentials. Save the selected offer, GPU model/VRAM, hourly cost, region, container image digest, CUDA/runtime versions and persistent-volume location before training.

Use only job-owned paths and resources. Checkpoint and sync evidence before any approved stop action. Do not delete volumes, terminate unrelated pods, change cloud billing settings or overwrite old results. Resource lifecycle permissions must be explicit. No unattended work is implied by this plan.
