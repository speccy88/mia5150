# Experimental roadmap

Prepared 2026-10-05 for **Agent Skill Discovery at Scale**. This is a proposed study, not a record of executed experiments. Paper IDs refer to RESEARCH_CATALOGUE.md and registries/papers.json. Official resources are in registries/engineering_sources.json.

## 1. Decision and research questions

Run a staged tournament rather than a Cartesian product of every model, corpus size, retrieval method and prompt. The leading hypothesis is a **skill-trained retriever plus complementary field-aware lexical evidence, a strong reranker, and selective escalation for ambiguous multi-skill cases**. It is not an established winner.

Retain two deliverables: the best validated retrieval/selection quality, and a Pareto set of quality versus latency/cost/resource demand. A Pareto-efficient configuration is one that no other tested configuration improves in both quality and resource cost.

The original proposal's central question remains: how do discovery method, library size, model choice and skill overlap affect accuracy and efficiency? Add three explicit subquestions: (1) Does a stronger model improve the ranking after candidate coverage is controlled? (2) Does skill structure or set compatibility explain gains beyond flat semantic similarity? (3) Can a small model reproduce the useful decisions of a stronger model at lower deployment cost?

## 2. Evidence changes the starting point

S01's corrected benchmark has task-trained 0.6B/8B embedders and a task-trained 0.6B reranker. The strongest published pipeline in the inspected official resources is SKILLRET-Embedding-8B followed by SKILLRET-Reranker-0.6B on its top 20. See evidence/published_reference_results.json; those numbers are published reference values, not our measurements.

The same paper reports an oracle-insertion diagnostic: supplying missing gold candidates to the same reranker raises NDCG@10 from 87.74 to 88.89. That is only a 1.15-point difference using the rounded endpoints (the source prints +1.14 from unrounded results). It is **not** the theoretical best NDCG and **not** a deployable method. It suggests prioritizing reranker discrimination once this strong retriever is in place. It does not establish that retrieval never matters for smaller or different retrievers. [S01, Table 4]

S14 provides a second warning: deterministic IR can match an LLM-mediated loop in its tested SkillsBench conditions. A more expensive LLM is a hypothesis to test, not a quality upper bound. [S14]

**Additional high-priority evidence:** SkillReason releases 0.6B/4B retrievers and rerankers and an implicit-query benchmark with 61,228 skills. Its paper's SkillRet comparison uses 6,660 skills and 4,997 queries, not the corrected pool. Treat its SOTA statement as scoped to that experiment and rerun the released checkpoints under our pinned protocol. Its external benchmark is especially useful for checking whether a SkillRet improvement transfers to short implicit requests. [S37,E18]

## 3. Benchmark tracks

**A — Official SkillRet comparison.** Pin the corrected corpus, queries, binary qrels, official evaluation code and checkpoints. Reproduce the official full-document representation and baseline limits. Report official NDCG/Recall/Completeness/MAP at 5, 10 and 15, plus the proposal's MRR@5. Public-test scores are comparable only with the same release, corpus, query set, qrels and protocol. Audit any historical development exposure.

**B — Proposal experiment.** Retain prompt, BM25, vector, domain vector and hybrid across 10/100/1,000/6,006 skills. Use a common name+description view for a clean discovery-information comparison, then an explicitly labelled full-body/structured-view extension. Report final selected skill sets as well as ranked candidates. Use the same query and candidate-library manifests across methods within a comparison.

**C — Transfer and robustness.** Freeze a small number of finalists and test on SRA first; add SkillRouter/R3-Skill or a new human-written panel only after access, split integrity and resource estimates are checked. French queries are an optional separate panel, not a renamed version of the English official benchmark. SkillsBench/SWE-Skills-Bench execution is optional and does not replace retrieval evaluation.

A cascade of skills does not follow merely from multiple gold IDs. Dependency order or compatibility needs its own annotation or benchmark.

## 4. Phase 0 — Audit, freeze, and validate (mandatory)

Inspect the existing repository and preserve all working runs, model manifests, scripts and failure logs. Record git status and historical test exposure, including aggregate benchmark results used to make decisions. Distinguish TypeSafe Jev API, JevK5, Plumb, CLM and general LLMs; never merge them under one model label.

Create immutable manifests for dataset/model/code revisions, text views, query partitions, nested skill libraries and candidate pools. Verify that a scorer only sees an opaque query ID and query text—not skill_ids, skill_names, gold k, original source-query keys or relevance-derived categories. Corpus skill names are legitimate inputs; query-side gold names are not.

Use unit tests for multi-positive recall, completeness, cutoff MRR, duplicate IDs, missing queries, invalid outputs and ties. Compare the main evaluator against the pinned official evaluator. No expensive run until these checks pass.

**Split warning:** the released SkillRet models were trained on the public training split. Making a new validation partition inside that split does not retroactively make it unseen by those models. Use it for engineering/training diagnostics with exposure disclosed. For model selection, audit the previous exposed pilot or establish a separately declared development panel. Keep genuine confirmatory queries hidden. If prior full-test feedback already guided experiments, do not claim an untouched 4,192-query remainder; use a new independent panel or external benchmark for confirmatory inference and call the full official score a public-benchmark result.

Gate: reproducible manifest + passing metric/adapter tests + exposure ledger + a token/memory/time/cost pilot. Phase 0 must produce a recommendation, not silently provision cloud resources.

## 5. Phase 1 — Reproduce useful anchors

Screen on a fixed development panel, not the whole public test for every tweak. Start with 20 synthetic/hand-checked smoke cases; use approximately 500 development queries for screening and, only where genuinely available, a larger 1,500-query development set for close decisions. These are proposed sizes, not already-created partitions.

Mandatory anchors: BM25; the existing BGE/CLM baselines when already implemented; Qwen3-Embedding-0.6B; SKILLRET-Embedding-0.6B; Qwen3-Embedding-8B; SKILLRET-Embedding-8B; and the official 8B→0.6B reranker pipeline. Full-catalog prompt and hard domain-routing anchors belong to Track B. The base-versus-task-trained Qwen pairs help separate size effects from task adaptation.

For a 6,006-document corpus, start with exact dense search to avoid confounding algorithm quality with ANN tuning. Cache document embeddings and query embeddings. Report indexing cost separately from per-query serving cost. Benchmark one deployment configuration at a time on the 8 GB local GPU.

Gate: the reproduction matches the pinned official implementation on fixed inputs; larger numerical differences are explained before calling an extension better. Define reproduction tolerances after checking precision/tie effects, before looking at proposed-method results.

## 6. Phase 2 — Improve candidate retrieval only where it adds coverage

Keep the strong SkillRet retriever as the reference. Add released SkillRouter-Embedding-0.6B and SkillReason-embedding-0.6b as specialist challengers before a large generic-model sweep. Promote SkillReason's 4B version only if its small version or transfer results justify the extra cost. [S02,S37,E16,E18] Compare their candidates with BM25, field-aware lexical retrieval, and one complementary generic model such as harrier-oss-v1-0.6b. Test a simple rank-fusion baseline before learned fusion. [I01, S11]

A minimal field experiment has name, description and body scores from both a lexical method and an embedding model. Reproduce S11's actual TF-IDF/dense method first; BM25F and a small learned fusion model are adapted extensions, not exact replicas. Fit field weights only on permitted development data. Contrast field-aware fusion with a flattened representation under equal available evidence. [S11, T05]

Use candidate K={20,50,100}. Measure candidate Recall@K and Completeness@K, overlap between retrieval channels, and the marginal gold coverage contributed by each channel. Preserve a fixed candidate artifact before reranking. An unweighted RRF that drops useful specialized-retriever hits is a failure to record, not a reason to remove that baseline.

Promising follow-ons: SkillSight content-bias correction [S09]; supported corpus-side pseudo-query views inspired by Skill2Query/Tool-REX [S10,T02]; learned sparse SPLADE [I02]; and ColBERT/BGE-M3 multi-vector matching [I03,I04]. Late chunking is conditional on a suitable encoder and a demonstrated long-document failure mode. [I09]

Do not train on a query generated from final test qrels. Index-side expansion using only a skill's text is distinct from training model weights on that test skill; label any such weight adaptation as transductive and report it separately. Query decomposition/rewriting must be an independently costed ablation, never a hidden preprocessing advantage.

Gate: promote at most two candidate generators. Keep a simple specialized dense baseline even when a hybrid wins. If candidate coverage is already high and additional candidates do not improve downstream quality, stop increasing K.

## 7. Phase 3 — Common-pool reranking tournament

Freeze the same query texts, candidate IDs, document views and initial order for all rerankers in each controlled comparison. Run a second native-capability comparison where model-specific length/batching settings are permitted and logged. This separates the quality of a scoring model from advantages caused by seeing more text.

First tournament: SKILLRET-Reranker-0.6B; released SkillRouter, R3 and SkillReason 0.6B rerankers; Qwen3-Reranker-8B; Jina-reranker-v3.5; JevK5; Plumb; TypeSafe Jev when API access is authorized. A hosted reranker such as Cohere Rerank v4.0 is an additional API control. Include the existing CLM implementation as a historical control, not an assumed competitive leader.

For the open JEV-like models, test a pointwise true/false relevance question per query-skill pair on exactly the same 20 candidates. Separately test the native choice readout on an explicitly capped list (up to 16 for the basic JevK5/Plumb comparison). A choice softmax is a distribution over mutually exclusive choices in that list; it is not an independent probability that every skill is relevant. Never concatenate scores from unrelated choice groups as if calibrated globally. Larger-list knockout routing is another algorithm with its own calls and risks.

Use native prompts/readouts. JevK5/Plumb should not be scored using the generic text-generation snippet displayed automatically by the Hub. The dedicated SkillRet/Qwen rerankers also use non-prose yes/no logit readouts, so no text generation is not unique to JEV-like models. [engineering sources E03,E04,E05,E06]

Then test one economical LLM and one strong LLM on a top-20 list. Use a constrained schema containing candidate IDs, not free-form names; validate uniqueness, membership and completeness of the ranking. Test pointwise/listwise formulations and a bounded pairwise variant on ambiguous finalists. Record actual reasoning/output tokens, list permutations, malformed outputs, retries and failures. [I06,I07]

A premium frontier model is a challenger, not an oracle. Promote it only if measured gains justify its cost. Consider an independent provider only after the first frontier comparison is informative.

Gate: promote at most two rerankers, retaining the official task-trained baseline. Report downstream results on real retrieved pools; oracle-inserted candidates are diagnostic only.

## 8. Phase 4 — Structure and adaptive selection

**Set selection:** Compare ordinary top-k, a relevance-gated MMR control, and DSR-style query-conditioned diversity [S12]. Optionally add explicit query-skill compatibility [S07]. Measure multi-skill completeness and final-set precision; do not reward diversity for its own sake. A redundancy label is not the same as the official gold identity.

**Domain routing:** Compare the proposal's hard top-one-domain restriction against top-two domains plus a small global fallback. Count gold lost before ranking and failures on cross-domain queries. Build domains from corpus content, not query-side relevance labels. Use identical domain maps for methods sharing the map. [S05,S19]

**Adaptive escalation:** A small reranker handles easy cases; uncertain/contradictory cases go to Jev or a strong LLM. Choose the escalation rule on development data, using calibrated uncertainty, score margins, channel disagreement or constraint conflicts. Model confidence is not proof of correctness. Log the escalation rate and compute full average and tail cost.

**Graph extension:** Only if cascade/dependency errors remain important, build typed dependency/specialization/conflict relations with provenance. Compare a shallow neighborhood expansion with a full graph method. Freeze graph and skill contents during official evaluation. [S04,S16,S17]

Gate: components must demonstrate an incremental benefit in one-at-a-time ablations. Do not ship an opaque hybrid containing every idea.

## 9. Phase 5 — Targeted adaptation and distillation

Do not commit to Plumb fine-tuning in advance. Adapt the model whose measured failure pattern and inference cost make the clearest case. Candidates: small reranker with hard-negative SFT/BCE, specialized embedder with contrastive loss, JevK5/Plumb LoRA, or a small student distilled from strong-model scores. The released SkillRet reranker already uses multi-retriever hard negatives; merely adding that technique is not a novel contribution. SkillReason already transfers training-time capability reasoning into query-only retrieval, so that broad idea is also prior art, not a new claim for this project. [S01 model documentation,S37]

Mine negatives from the actual candidate generators. Keep every labelled positive out of the negative pool, including positives shared across grouped examples or in-batch negatives. Query/skill-family/repository grouping and duplicate checks must precede split creation. If grouping creates giant connected components, report and handle that explicitly rather than silently allowing leakage.

For pointwise relevance, train all positives as positive with binary loss. For listwise training, use a declared multi-positive loss; do not arbitrarily choose one gold. Uniform soft targets are one possible baseline, not automatically the optimal objective. Unjudged functionally valid alternatives may be false negatives; keep an audit rather than rewriting the official qrels.

Run a no-adaptation control, a hard-negative adaptation, and a replay/anchoring variant. Use three seeds only for finalists to estimate training variation. Evaluate on a separate transfer panel before accepting a synthetic-data win. [S13] Preserve original weights and all intermediate adapters. Check teacher-provider terms before distillation.

Gate: a real quality/efficiency improvement with no hidden split contamination or severe transfer regression. Stop adaptation if simpler readout or representation changes solve the problem.

## 10. Phase 6 — Locked evaluation, scaling and reporting

Freeze two to four systems: official reference, best quality, best efficient, and adaptive/multi-skill finalist when distinct. Run the full official test as a public-benchmark comparison; separately identify genuinely confirmatory data according to the exposure audit. For expensive APIs, use the same preregistered approximately 2,000-query sample for every directly compared system, with the same strata and any required sampling weights.

Run the proposal's five method families and four library sizes on a smaller fixed shared panel before spending on every combination. Use three library seeds initially. Near-duplicate/hard-distractor panels are separately labelled; the library construction is nested within each query/seed and must include the same permitted gold candidates across methods. Never silently replace a full catalog with a retrieved subset in the prompt baseline.

Recommended proposed promotion rule: quality improvement of at least 1 percentage point NDCG@10 with a paired 95% interval above zero; alternatively demonstrate non-inferiority within 0.5 points with at least 25% lower cost or latency. Protect Completeness@10 against a greater-than-1-point regression. These are project decision thresholds, not published standards; set them before the locked comparison.

Use paired query-level bootstrap intervals, with clustered resampling for related query templates/families where appropriate. Use planned contrasts/Holm adjustment rather than selecting one lucky p-value from a large sweep. Include zero/failed outputs in operational scores. Separate inference nondeterminism from training-seed variability. Preserve all tested configurations, including negative results.

Final claims should say exactly which pinned benchmark and reproduced baseline improved. Do not claim universally best skill retrieval, guaranteed better agents, or state of the art across incompatible releases.
