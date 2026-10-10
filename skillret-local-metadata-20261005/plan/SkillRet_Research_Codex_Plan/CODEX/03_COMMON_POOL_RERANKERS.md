# J03 — Common-pool reranking tournament

Prerequisite: J01; use its fixed top-20 pools first. If J02 pools later become available, use them in a separately labelled run rather than changing inputs mid-comparison. Check all licenses and approved API budgets.

Implement native scoring for the SkillRet-0.6B reference, the released SkillRouter/R3/SkillReason 0.6B rerankers, Qwen3-Reranker-8B, Jina-reranker-v3.5, JevK5 and Plumb. Add hosted TypeSafe Jev only with authorized access. First compare pointwise relevance on exactly the same documents. JevK5/Plumb need option-letter logits/native runtime, not generic text generation. Jina listwise output needs its native joint-candidate implementation; do not reduce it to arbitrary hidden-state pooling.

Run a second, clearly separated capped-choice JevK5/Plumb experiment. Respect the basic 16-option limit; compare a documented larger-list algorithm as a distinct pipeline. Never aggregate unrelated per-list softmax values as global probabilities. Keep every relevant skill eligible; choosing one option is not the entire multi-skill task.

After valid small-model results, compare an economical and a strong LLM with constrained candidate-ID rankings. Test order permutations and a bounded pairwise variant. Same available text and candidate IDs for the controlled comparison; native length capabilities are another track. Count hidden reasoning/output tokens where billed, malformed outputs, retries and response-cache hits.

Acceptance: per-query rankings, exact readout/prompt/runtime versions, order-sensitivity and quantization checks, latency/cost/validity metrics, paired development comparisons. Promote at most two rerankers while keeping the official reference. No gold-insertion result may be included in the normal leaderboard.

## Shared requirements

Apply MASTER_HANDOFF.md and BENCHMARK_CONTRACT.md. Preserve existing code/results and pin all new inputs. Respect explicit resource approvals; null/absent budget means no paid operations. Save exact commands, versions, input hashes and raw per-query results. Any artifact handoff is one ZIP with START_HERE, including every related patch, test, log and report. No claim of model quality is valid merely because an adapter smoke test passed.
