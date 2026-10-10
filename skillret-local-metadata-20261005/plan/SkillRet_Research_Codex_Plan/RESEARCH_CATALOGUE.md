# Research catalogue — 54 papers and technical reports

Research cutoff: **2026-10-05**. Scope: retrieval/routing of procedural skills for LLM agents, relevant benchmarks, and transferable tool/IR methods. This is a scoped literature sweep, **not a guarantee of exhaustive coverage**. Robotics motor-skill retrieval and employment-skills search are excluded. Engineering releases are listed separately.

Primary landing pages were checked. Inspection depth is explicit: most entries received abstract/metadata screening, not a complete methodological review. The key SkillRet, field-aware, diversity-aware, SkillReason, and SkillSeek results/method sections were examined more closely. No paper's experimental results were reproduced here. Before implementing a method, Codex/Anonymous must review its full method, official code, data availability, licensing, and evaluation protocol.

Priority: P0 prerequisite; P1 first competitive experiments; P2 gated extension; P3 background/outside the core scope. These are **our project priorities**, not a published ranking of methods.

## S01 — SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents (2026)

**Category:** direct · **Priority:** P0 · **Reviewed:** methods/results inspected.

Primary source: https://arxiv.org/abs/2605.05726

**Experiment implication:** Reproduce the corrected evaluation and task-trained embedding/reranker baselines.

**Boundary:** Published results are not our measurements; pin dataset, code, model and representation.

## S02 — SkillRouter: Skill Routing for LLM Agents at Scale (2026)

**Category:** direct · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2603.22455

**Experiment implication:** Compare retrieve-then-rerank routing and an independent skill benchmark.

**Boundary:** The current title differs from earlier cached descriptions; do not mix benchmark pools.

## S03 — Skill Retrieval Augmentation for Agentic AI (2026)

**Category:** direct · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2604.24594

**Experiment implication:** Use SRA as the first external retrieval/skill-use benchmark.

**Boundary:** Retrieval, incorporation and execution are different outcomes.

## S04 — Graph-of-Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills (2026)

**Category:** direct · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2604.05333

**Experiment implication:** Test dependency-aware expansion only after a strong flat baseline.

**Boundary:** Graph construction and graph-information budgets are extra experimental factors.

## S05 — When Single-Agent with Skills Replace Multi-Agent Systems and When They Fail (2026)

**Category:** direct · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2601.04748

**Experiment implication:** Motivates catalog-size and hierarchical-selection experiments.

**Boundary:** Not a directly comparable SkillRet leaderboard result.

## S06 — More Skills, Worse Agents? Skill Shadowing Degrades Performance When Expanding Skill Libraries (2026)

**Category:** direct · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2605.24050

**Experiment implication:** Measure overlapping-skill confusion and final selection, not only retrieval.

**Boundary:** An overlap stress set is a separate benchmark slice, not the official overall score.

## S07 — Skill Is Not Document: Query-Conditioned Compatibility for LLM Agent Skill Routing (2026)

**Category:** direct · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2606.03565

**Experiment implication:** Test compatibility-aware skill-set selection; inspect R3-Skill for transfer.

**Boundary:** Compatibility and relevance need separate labels and metrics.

## S08 — Right Family, Wrong Skill: Evaluating Risk Exposure in Agent Skill Retrieval (2026)

**Category:** direct · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2606.10388

**Experiment implication:** Add helpful-versus-risky sibling analysis after quality baselines.

**Boundary:** Do not use oracle family labels as an available routing feature.

## S09 — SkillSight: Calibrating Generic Content Bias for Skill Retrieval (2026)

**Category:** direct · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2607.18785

**Experiment implication:** Test correction of generic/shared skill-content bias.

**Boundary:** Training-free does not mean tuning-free; select coefficients on development data.

## S10 — Skill2Query: Exploiting Skill Structure to Generate Pseudo-Queries for Agent Skill Retrieval (2026)

**Category:** direct · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2608.16071

**Experiment implication:** Build corpus-derived pseudo-query views and separately test query rewriting.

**Boundary:** Index augmentation, model training, and online rewriting must be separate ablations.

## S11 — Field-Aware Agent Skill Retrieval (2026)

**Category:** direct · **Priority:** P1 · **Reviewed:** methods/results inspected.

Primary source: https://arxiv.org/abs/2608.02880

**Experiment implication:** Keep name, description and body separate; compare lexical/dense field fusion.

**Boundary:** Reported older SkillRet pool has 6,660 skills and 4,997 queries; not directly comparable to corrected scores.

## S12 — Beyond Top-k Skill Retrieval: Diversity-Aware Skill Routing for LLM Agents (2026)

**Category:** direct · **Priority:** P1 · **Reviewed:** methods/results inspected.

Primary source: https://arxiv.org/abs/2609.05824

**Experiment implication:** Test query-conditioned diversity selection against ordinary top-k and MMR.

**Boundary:** Diversity can remove two genuinely required similar skills; monitor completeness.

## S13 — When Synthetic Data Hurts: On Catastrophic Forgetting in Skill Retrieval for LLM Agents (2026)

**Category:** direct · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2609.10750

**Experiment implication:** Require real/OOD checks and replay or anchoring when adapting models.

**Boundary:** In-distribution synthetic gains alone do not establish better generalization.

## S14 — SkillSeek: Revisiting Agent Skill Retrieval at Marketplace Scale (2026)

**Category:** direct · **Priority:** P1 · **Reviewed:** methods/results inspected.

Primary source: https://arxiv.org/abs/2609.38822

**Experiment implication:** Compare deterministic retrieval with expensive LLM-mediated selection under matched execution conditions.

**Boundary:** Its SkillsBench conclusions are conditional on the tested pool/backbone/harness.

## S15 — SkillFlow: Scalable and Efficient Agent Skill Retrieval System (2025)

**Category:** direct · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2504.06188

**Experiment implication:** Study staged reranking and escalation as a system design.

**Boundary:** The identifier and title were checked at the primary landing page; benchmark-specific outcomes must stay separate.

## S16 — SkillDAG: Self-Evolving Typed Skill Graphs for LLM Skill Selection at Scale (2026)

**Category:** direct · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2606.03056

**Experiment implication:** Compare typed skill relations with a simpler hierarchy and flat global retrieval.

**Boundary:** Freeze graph updates before the fixed-corpus test; online evolution needs another protocol.

## S17 — SE-GoS: Self-Evolving Graph-of-Skills for Skill Library at Scale (2026)

**Category:** direct · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2609.08228

**Experiment implication:** Optional graph-index adaptation experiment.

**Boundary:** Test-time feedback can leak across queries; distinguish frozen and online evaluation.

## S18 — SkillApt: Learning When to Activate Agent Skills from Counterfactual Evidence (2026)

**Category:** direct · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2609.26863

**Experiment implication:** Use matched with-skill/without-skill outcomes for activation research.

**Boundary:** A retrieved relevant skill is not necessarily beneficial or necessary.

## S19 — Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale (2026)

**Category:** direct · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2603.02176

**Experiment implication:** AgentSkillOS motivates capability trees and hierarchical routing.

**Boundary:** Keep orchestration/execution outside the core discovery claim.

## S20 — A Comprehensive Survey on Agent Skills: Taxonomy, Techniques, and Applications (2026)

**Category:** survey · **Priority:** P0 · **Reviewed:** survey sections/reference list inspected.

Primary source: https://arxiv.org/abs/2605.07358

**Experiment implication:** Use as a taxonomy and backward/forward citation-search seed.

**Boundary:** A survey is a discovery source, not experimental evidence that a particular method wins.

## S21 — Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward (2026)

**Category:** survey · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2602.12430

**Experiment implication:** Define the scope of procedural agent skills and their lifecycle.

**Boundary:** Do not import broad ecosystem claims as measured SkillRet outcomes.

## S22 — SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks (2026)

**Category:** benchmark · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2602.12670

**Experiment implication:** Optional downstream execution check for frozen finalists.

**Boundary:** Different papers use different subsets/releases; pin exact tasks, harness and verifier.

## S23 — Agent Skills: A Data-Driven Analysis of Claude Skills for Extending Large Language Model Functionality (2026)

**Category:** background · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2602.08004

**Experiment implication:** Background for real skill-library composition and overlap.

**Boundary:** An ecosystem analysis is not a retrieval method comparison.

## S24 — SWE-Skills-Bench: Do Agent Skills Actually Help in Real-World Software Engineering? (2026)

**Category:** benchmark · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2603.15401

**Experiment implication:** Optional software-engineering transfer benchmark.

**Boundary:** Code execution, repository state and task costs add scope beyond discovery.

## S25 — GraphSkill: Documentation-Guided Hierarchical Retrieval-Augmented Coding for Complex Graph Reasoning (2026)

**Category:** background · **Priority:** P3 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2603.06620

**Experiment implication:** Background on hierarchical retrieval of procedural knowledge.

**Boundary:** Domain-specific reasoning is not a plug-in replacement for SkillRet evaluation.

## S26 — GraSP: Graph-Structured Skill Compositions for LLM Agents (2026)

**Category:** background · **Priority:** P3 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2604.17870

**Experiment implication:** Background on graph-structured skill composition.

**Boundary:** Composition labels and task execution require a separate experiment.

## S27 — Maestro: Reinforcement Learning to Orchestrate Hierarchical Model-Skill Ensembles (2026)

**Category:** background · **Priority:** P3 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2605.22177

**Experiment implication:** Background on learned model/skill orchestration.

**Boundary:** Do not introduce reinforcement learning before cheaper discovery baselines are understood.

## S28 — Skill is Not One-Size-Fits-All: Model-Aware Skill Alignment for LLM Agents (2026)

**Category:** background · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2605.30723

**Experiment implication:** Motivates checking whether the same retrieved skill helps different final LLMs.

**Boundary:** Model-dependent utility is distinct from the fixed relevance labels.

## S29 — Skill or Skip? Learning Selective Skill Invocation in Agentic Tasks via Dual-Granularity Preference Learning (2026)

**Category:** background · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2606.00510

**Experiment implication:** Related selective-invocation approach for an optional abstention study.

**Boundary:** Needs genuine no-skill/utility labels; a candidate miss is not such a label.

## S30 — SkillsInjector: Dynamic Skill Context Construction for LLM Agents (2026)

**Category:** background · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2605.29794

**Experiment implication:** Background for selecting which skill content to expose to the final model.

**Boundary:** Context construction changes the evidence budget; measure it explicitly.

## S31 — SkillNet: Create, Evaluate, and Connect AI Skills (2026)

**Category:** background · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2603.04448

**Experiment implication:** Related infrastructure for connected skill libraries.

**Boundary:** Creating or modifying skills changes the corpus and belongs in a separate track.

## S32 — Memento-Skills: Let Agents Design Agents (2026)

**Category:** background · **Priority:** P3 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2603.18743

**Experiment implication:** Background on evolving skill-based agents.

**Boundary:** Do not allow cross-test-query learning in a static retrieval benchmark.

## S33 — SkillFM: Generating Skills for LLM Agents via Latent Flow Matching (2026)

**Category:** background · **Priority:** P3 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2609.39382

**Experiment implication:** Keep in the literature map as an alternative to retrieving existing skills.

**Boundary:** Generated skills cannot be scored as existing gold IDs without a new benchmark.

## S34 — GraphSkillAA: Attribution-Guided Skill-Graph Updating with Targeted Validation and Rollback (2026)

**Category:** background · **Priority:** P3 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2609.20455

**Experiment implication:** Background for validated graph updates and rollback.

**Boundary:** Current primary title differs from the shorter SkillAA name in secondary indexing.

## S35 — SkillSpec: Consensus-Gated Agent Skill Evolution via Representation Specialization (2026)

**Category:** background · **Priority:** P3 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2610.00704

**Experiment implication:** Recent work on validated skill updates and flat/graph/hybrid representation choices.

**Boundary:** Skill evolution is outside the fixed-corpus leaderboard track.

## S36 — Chaining Skills to Hijack LLM Agents (2026)

**Category:** security · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2610.01564

**Experiment implication:** Motivates treating skill text as untrusted data and isolating downstream execution.

**Boundary:** The core retrieval study should never execute retrieved skill instructions.

## S37 — SkillReason: Reasoning-Enhanced Agent Skill Retrieval for Implicit User Requests (2026)

**Category:** direct · **Priority:** P1 · **Reviewed:** abstract, selected methods/results, benchmark statistics and released interfaces inspected.

Primary source: https://arxiv.org/abs/2608.08640

**Experiment implication:** Test released 0.6B retrieval/reranking models first, optionally promote the 4B pair; use SkillReason-Bench as an independent implicit-query transfer benchmark. Compare training-time capability-reasoning transfer against inference-time query expansion.

**Boundary:** The paper uses the older SkillRet pool (6,660 skills and 4,997 queries), so its claimed SOTA is not directly comparable to the corrected 6,006/4,392 benchmark. Audit training/test exposure per checkpoint.

## T01 — Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models (2025)

**Category:** tool transfer · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2503.01763

**Experiment implication:** ToolRet provides a related external retrieval task.

**Boundary:** API/tool selection and procedural SKILL.md retrieval are related, not identical.

## T02 — Tools are under-documented: Simple Document Expansion Boosts Tool Retrieval (2025)

**Category:** tool transfer · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2510.22670

**Experiment implication:** Tool-REX suggests richer corpus-side capability descriptions.

**Boundary:** Generated capabilities must be supported by the source; no gold-query-conditioned expansion.

## T03 — ToolExpNet: Optimizing Multi-Tool Selection in LLMs with Similarity and Dependency-Aware Experience Networks (2025)

**Category:** tool transfer · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://aclanthology.org/2025.findings-acl.811/

**Experiment implication:** Related dependency-aware multi-tool selection.

**Boundary:** Experience graphs must be trained only on permitted past data.

## T04 — Towards Completeness-Oriented Tool Retrieval for Large Language Models (2024)

**Category:** tool transfer · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2405.16089

**Experiment implication:** COLT motivates all-required-tools/skills completeness rather than first-hit metrics alone.

**Boundary:** Tool co-occurrence may not transfer to unseen skill families.

## T05 — Multi-Field Tool Retrieval (2026)

**Category:** tool transfer · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2602.05366

**Experiment implication:** Related evidence for preserving separate structured fields.

**Boundary:** Reproduce the actual field method before claiming equivalence to an adapted BM25F design.

## I01 — Reciprocal rank fusion outperforms condorcet and individual rank learning methods (2009)

**Category:** IR foundation · **Priority:** P0 · **Reviewed:** abstract/metadata inspected.

Primary source: https://doi.org/10.1145/1571941.1572114

**Experiment implication:** Use rank fusion as the simple lexical/dense combination control.

**Boundary:** Fusion is not guaranteed to improve an already strong specialized retriever.

## I02 — SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval (2021)

**Category:** IR foundation · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2109.10086

**Experiment implication:** Test learned sparse retrieval for complementary lexical/semantic recall.

**Boundary:** Choose and pin an actual checkpoint and index settings; the paper is not the deployment artifact.

## I03 — ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction (2021)

**Category:** IR foundation · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2112.01488

**Experiment implication:** Test token-level multi-vector matching versus one vector per skill.

**Boundary:** Measure index size, document length policy and skill-level aggregation.

## I04 — M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation (2024)

**Category:** IR foundation · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2402.03216

**Experiment implication:** BGE-M3 provides dense, sparse and multi-vector representation controls.

**Boundary:** Do not label a single dense BGE-M3 run as a test of all three modes.

## I05 — Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models (2025)

**Category:** IR foundation · **Priority:** P0 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2506.05176

**Experiment implication:** Use the base family to separate model size from task-specific fine-tuning.

**Boundary:** Use the documented instructions, pooling, and yes/no reranker readout.

## I06 — Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents (2023)

**Category:** IR foundation · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2304.09542

**Experiment implication:** RankGPT motivates listwise LLM reranking and later distillation.

**Boundary:** Control order, output validity, candidate content, and token budget.

## I07 — Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting (2023)

**Category:** IR foundation · **Priority:** P1 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2306.17563

**Experiment implication:** Pairwise comparisons are an alternative for a small ambiguous finalist set.

**Boundary:** All-pairs calls scale quadratically; record actual comparisons and cycles.

## I08 — RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze! (2023)

**Category:** IR foundation · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2312.02724

**Experiment implication:** Related training/distillation and rank_llm implementation reference.

**Boundary:** Not a recommendation to run every older released ranker.

## I09 — Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models (2024)

**Category:** IR foundation · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2409.04701

**Experiment implication:** Preserve context when indexing sections of long skill documents.

**Boundary:** Requires an encoder/readout that supports the method; not every embedding API does.

## I10 — Precise Zero-Shot Dense Retrieval without Relevance Labels (2022)

**Category:** IR foundation · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2212.10496

**Experiment implication:** HyDE is an optional query-side generation control.

**Boundary:** A generated hypothetical document can invent prerequisites; retain original query evidence.

## I11 — jina-reranker-v3: Last but Not Late Interaction for Listwise Document Reranking (2025)

**Category:** IR foundation · **Priority:** P2 · **Reviewed:** abstract/metadata inspected.

Primary source: https://arxiv.org/abs/2509.25085

**Experiment implication:** Background on joint candidate attention without textual ranking generation.

**Boundary:** This is not ColBERT-style separately indexed late interaction.

## I12 — jina-reranker-v3.5: An Efficient Listwise Reranker with Hybrid Attention and Self-Distillation (2026)

**Category:** IR foundation · **Priority:** P1 · **Reviewed:** abstract/metadata and model documentation inspected.

Primary source: https://arxiv.org/abs/2607.18152

**Experiment implication:** Test a small joint listwise reranker against pointwise rerankers and JEV-like decisions.

**Boundary:** Weights carry non-commercial terms; vendor controlled-pool metrics are not end-to-end SkillRet scores.

