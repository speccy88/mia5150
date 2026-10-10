# J04 — Structure, set selection and escalation

Prerequisites: fixed J02/J03 candidates and reranker choices. Review S07/S12 and relevant hierarchy papers in full before implementation.

Compare ordinary relevance top-k, a relevance-gated MMR baseline and query-conditioned diversity (DSR-style). Add compatibility-aware set scoring only with a declared label/annotation source. Evaluate multi-positive completeness and final-set precision/recall/F1/exact match; do not use gold set size as an input. Similar skills can both be necessary, so a diversity penalty is not automatically correct.

Compare the proposal's hard top-one-domain restriction with top-two domains plus global fallback. Freeze a corpus-derived domain map. Measure errors caused by early routing separately from ranking errors, especially cross-domain and overlapping-skill tasks.

Test a small-to-strong escalation policy using development-calibrated confidence/margins/disagreement/constraints. Record escalation rate, average and p95 total cost/latency. Separate local NONE (none of supplied candidates) from genuine no-skill-needed activation. Use counterfactual data or a labelled panel for activation.

Only if dependency errors justify it, add a small typed graph expansion with provenance and fixed graph state. Online graph evolution is not allowed within static official test evaluation. Run component-by-component ablations rather than a single opaque all-features system.

Acceptance: a measurable incremental effect per promoted component, final selected-set validation, a documented deployment policy and a small frozen finalist list. Reject complexity that is not supported by gains.

## Shared requirements

Apply MASTER_HANDOFF.md and BENCHMARK_CONTRACT.md. Preserve existing code/results and pin all new inputs. Respect explicit resource approvals; null/absent budget means no paid operations. Save exact commands, versions, input hashes and raw per-query results. Any artifact handoff is one ZIP with START_HERE, including every related patch, test, log and report. No claim of model quality is valid merely because an adapter smoke test passed.
