# Alignment with the supplied proposal

Source: inputs/proposal_1.pdf, read in full (4 pages). The proposal remains the basis, not something replaced by a generic RAG study.

| Proposal commitment | Keep | Proposed refinement / extension |
|---|---|---|
| Benchmark and empirical study; accuracy/efficiency depend on multiple factors (p.1) | Yes | Search for a measured quality/efficiency frontier, not a predetermined winner. |
| Discovery, rather than activation and execution, is the focus (p.1) | Yes | Execution is an optional external check after discovery results are locked. |
| Prompt, BM25, vector, domain vector, hybrid (p.2) | All five | Treat reranking/field-aware fusion/adaptive escalation as documented extensions to these families. |
| Library sizes 10,100,1,000,6,006 (p.2) | Yes | Pin shared nested-library manifests; distinguish global libraries from query-conditioned interference tests. |
| Skill descriptions used in the five-method definitions (p.2) | Yes, in a controlled header-view track | Add a separate full-body track to compare fairly with published SkillRet baselines. |
| Auto/manual tasks; overlapping and cascading skills (p.2) | Yes | Keep official queries separate from new robustness panels; dependency order requires extra labels. |
| Final selection accuracy, Recall@5, MRR@5, tokens, relative latency (p.3) | All retained | Add official NDCG/Completeness/MAP at 5/10/15, exact-set definition, absolute p50/p95 and cost. |
| Local models and API models; approximately 2,000 API queries (p.3) | Yes, contingent on budgets | Use paired preregistered samples and benchmark-based runtime estimates. |
| Approximately 10 local GPU hours and $20 CAD/API configuration (p.3) | Retain as historical proposal assumptions | Re-estimate before execution. They are not verified costs or promises for new models. |
| Weeks 5–8 literature/harness/experiments, 9+ reporting (p.4) | Compatible | Map phase gates onto those weeks; defer optional graphs/training if time is tight. |

The 20-query BM25 Recall@5=0.542 in the proposal is explicitly a feasibility pilot, not the full-set benchmark anchor (p.3). Never compare it directly to a 4,392-query reference score as a performance improvement.

## Where novelty must be narrowed

Field-Aware Agent Skill Retrieval already examines structured skill fields and growth of skill libraries. Hierarchies, dependency graphs, skill shadowing, and multi-skill routing also have prior work. Therefore do not claim to be the first to combine lexical/vector retrieval or the first to study scaling. [S04–S06,S11,S12,S19]

A defensible contribution is a controlled comparison of **strong task-trained baselines, generic/high-capability rerankers, typed-decision models and structured skill-set selection**, coupled with explicit information budgets, library interference, hardware-normalized reporting and cross-benchmark validation. A targeted improvement is valuable only when ablations show which component caused it.

Suggested revised empirical claim (proposed wording, not an established result):

> We evaluate when skill-aware retrieval, reranking and selective reasoning improve final skill discovery over full-catalog prompting, lexical retrieval and task-trained retrieval baselines, across library sizes and resource budgets. We isolate candidate coverage, ranking quality and multi-skill selection, and test whether gains transfer beyond the development benchmark.

## Responsibilities following the proposal

Anonymous: full-paper methodology cards, version/protocol comparison, related-work and negative-results synthesis. Anonymous and Anonymous: harness/data audit, immutable inputs, baseline reproduction, local/cloud/API adapters, cost and latency measurements. All: freeze hypotheses, development/test exposure policy and final contrasts before the confirmatory runs. This follows the proposed division; it is not a new assignment already accepted by the team.
