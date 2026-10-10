# J05 — Targeted adaptation or distillation

Prerequisite: J03 evidence and explicit training resources. Do not assume Plumb is the training target. Select the target from measured failures and desired serving cost; provide the justification and a representative memory pilot first.

Review S01's current released training recipe and S13's forgetting analysis. The SkillRet reranker already uses multi-retriever hard negatives, so that alone is not a new method. Mine negatives from permitted training data with the candidate generators actually used at deployment. Remove every known positive from negatives and in-batch negative collisions; audit duplicate/repository/family grouping before splitting.

Compare no adaptation, hard-negative adaptation and a replay/anchoring variant. For binary relevance use all labelled positives. For listwise losses explicitly support multiple positives. If distilling a stronger teacher, record model/version/cost and check terms; keep teacher labels separate from gold evaluation labels. Do not train on final test queries or qrels.

Use reversible adapters where practical; preserve base weights/checkpoints. Run a tiny optimization smoke test and overfit a toy training batch before spending on full data. Check finite loss, changed adapter parameters, reproducible resume and actual evaluation. Three seeds only for promoted configurations. Evaluate a held-out real/OOD panel as well as in-distribution performance.

Acceptance: training logs and checkpoint hashes, native deployment compatibility, paired gains/cost data against frozen controls, and a transfer-regression report. A lower training loss alone is not acceptance. A failed or unhelpful fine-tune remains a documented result.

Prior-art check: SkillReason [S37] already studies training-time capability-reasoning supervision with query-only deployment. Treat it as a released-model baseline and distinguish any new multi-positive/adaptive/efficiency experiment from merely reproducing that idea.

## Shared requirements

Apply MASTER_HANDOFF.md and BENCHMARK_CONTRACT.md. Preserve existing code/results and pin all new inputs. Respect explicit resource approvals; null/absent budget means no paid operations. Save exact commands, versions, input hashes and raw per-query results. Any artifact handoff is one ZIP with START_HERE, including every related patch, test, log and report. No claim of model quality is valid merely because an adapter smoke test passed.
