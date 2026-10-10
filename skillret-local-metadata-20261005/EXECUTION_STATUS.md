# Execution status

This table describes recorded execution, not only code availability. Read IMPLEMENTATION_MAP.md for methodological boundaries.

| Component | Status | Run receipts |
|---|---|---|
| Original plan/data/split and metric audit | tested | contract-root: completed, training-audit: completed |
| Full4392 public comparison | tested | full-lexical: completed, full-skillret-direct: completed |
| Native R3 sharded FP16 | tested | r3-fp16-cache: completed, r3-fp16-rerank: completed |
| Native Jina joint reranking | tested | jina-source: completed, jina-native: completed |
| Full-body1024 retrieval | blocked-or-incomplete | fullbody-smoke: completed, fullbody-skillret: resource-blocked |
| Six-channel field retrieval | blocked-or-incomplete | field-six: resource-blocked |
| Actual hard-negative LoRA | tested | lora-hard-streamed: completed |
| Hard-negative LoRA transfer | tested | lora-transfer-base: completed, lora-transfer-hard: completed |
| Corrected replay (interrupted checkpoints only) | blocked-or-incomplete | lora-replay-clean-lean: resource-blocked |
| Quantized8B native reranking | blocked-or-incomplete | qwen8-download: resource-blocked, qwen8-rerank: not-run |
| Local4B ID ranking and pairwise | blocked-or-incomplete | llm-candidates: resource-blocked |
| Native multipass option choice | blocked-or-incomplete | large-choice-jev: not-run, large-choice-plumb: not-run |
| 32K full-catalog prompt diagnostic | blocked-or-incomplete | prompt-32768: not-run |
| SRA frozen24-query full-catalog transfer | blocked-or-incomplete | sra-transfer: not-run |
| SkillReason frozen24-query full-catalog transfer | blocked-or-incomplete | skillreason-data: not-run, skillreason-schema: not-run, skillreason-external: not-run |
| Frozen16-query actual escalation latency | blocked-or-incomplete | policy-native: not-run |

Implemented and tested earlier: locked lexical and0.6B retrieval, same-pool rerankers, grouped lexical adaptation, selection/domain/escalation controls,8K prompting and nested catalog scaling. Raw receipts and exact source snapshots remain in runs/.

Deferred: paid APIs/RunPod,8B BF16 reference, full32K dense evidence, large multiseed training and full-query reranking tournament. Conditional SPLADE/late interaction/generated views, learned routing and labelled dependency/activation methods were not promoted or claimed implemented without their required evidence. Certified untouched final confirmation remains unavailable.