# J06 — Locked evaluation and reporting

Prerequisites: model/representation/policy shortlist frozen on declared development data; exposure audit resolved; full evaluation resource caps approved. Include J05 only if it independently passed; do not wait for optional training if simpler systems already meet the goal.

Freeze two to four systems: official reference, quality winner, efficient winner and adaptive/set-aware winner where distinct. Run paired full official SkillRet evaluation as a public-benchmark comparison. Identify a genuinely independent confirmation set or external benchmark; never rename previously consumed test feedback as blind validation. Expensive API runs use the same preregistered approximately 2,000-query sample for all directly compared methods, with strata/weights and confidence intervals recorded.

Run the five proposal method families across 10/100/1,000/6,006 skills on identical nested-library/query manifests and initially three library seeds. Keep random versus hard-overlap distractors separate. Flag trivial @10 coverage in the 10-skill condition and full-prompt overflows. Retain header-only and full-document tracks as distinct.

Report official metrics, MRR@5 and final-set accuracy plus p50/p95, costs, index sizes and failure rates. Use paired/clustered confidence intervals, planned contrasts and a multiplicity policy. Provide effect sizes and a Pareto table, not only ranks or p-values.

Test the frozen finalists on SRA or another approved external benchmark without tuning on it. Downstream execution is optional, isolated and separately scored. Blinded error examples should distinguish missed retrieval, wrong ranking, missed second/third skill, wrong domain, invalid output and correct-retrieval/wrong-use.

Acceptance: an exact supported claim with benchmark/version/sample/precision/hardware, all raw runs and negative findings, reproducible commands and one complete ZIP. Explicitly state what was not run. Do not claim universal superiority or true SOTA from incompatible protocols.

For cross-benchmark confirmation, prioritize frozen finalists on SRA-Bench and SkillReason-Bench (implicit requests), with SkillRouter/R3 as bounded multi-skill/bilingual diagnostics. Inspect each native label/metric protocol and checkpoint exposure before calling results confirmatory. Never average unlike metrics into an unexplained cross-benchmark score.

## Shared requirements

Apply MASTER_HANDOFF.md and BENCHMARK_CONTRACT.md. Preserve existing code/results and pin all new inputs. Respect explicit resource approvals; null/absent budget means no paid operations. Save exact commands, versions, input hashes and raw per-query results. Any artifact handoff is one ZIP with START_HERE, including every related patch, test, log and report. No claim of model quality is valid merely because an adapter smoke test passed.
