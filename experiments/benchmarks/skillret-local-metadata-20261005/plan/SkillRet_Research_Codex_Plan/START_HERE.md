# START HERE — Skill retrieval research and Codex experiment plan

**Prepared for Fred and the MIA5150 team · 2026-10-05**

## Recommendation

Start with the strongest **task-trained SkillRet baseline**, then test complementary candidate retrieval, common-pool rerankers, structured multi-skill selection, and evidence-driven adaptation. Keep Jev/JevK5/Plumb in the tournament, but do not preselect Plumb for training. A stronger LLM is a challenger, not an oracle. Preserve the proposal's five method families, four library sizes and final-selection focus.

The catalogue contains **54 papers/technical reports** and the engineering registry contains **18 primary documentation/implementation sources**. This is a scoped research sweep, not guaranteed exhaustive coverage. Per-paper inspection depth is recorded; most entries are abstract-screened. Published numbers were checked in primary sources but not reproduced here.

## Read in this order

1. EXPERIMENT_PLAN.md — staged questions, experiment design and promotion gates.
2. PROPOSAL_ALIGNMENT.md — what remains unchanged and which additions are optional.
3. BENCHMARK_CONTRACT.md — metrics, evidence budgets, split/exposure rules and failure handling.
4. RESOURCE_PLAN.md — local/RunPod/API roles and proposed CAD spending envelopes.
5. RESEARCH_CATALOGUE.md — all 54 papers, primary links, relevance and limitations.

For execution, select the existing project repository in Codex, provide this ZIP, and paste **CODEX/MASTER_HANDOFF.md**. Begin with **J00 only**. Allocate actual hardware/API/GPU-hour/CAD limits after its audit and pilot estimate. No paid resource has been authorized by this bundle.

## Package contents

- `inputs/proposal_1.pdf`: the original attached proposal, unchanged.
- `CODEX/MASTER_HANDOFF.md` and `CODEX/00_...` through `06_...`: a master prompt and seven gated work orders.
- `registries/papers.json`: machine-readable 54-paper catalogue.
- `registries/references.bib`: minimal BibTeX title/year/URL records; enrich author/venue metadata from primary exports before submission.
- `registries/engineering_sources.json`: models, provider docs, official benchmark resources and engineering leads.
- `registries/models.json`: 25 model/method roles, with unverified execution-time fields explicitly null/false.
- `registries/experiments.json`: planning dependency graph and default resource guards. **This is a planning manifest, not a working benchmark runner.**
- `evidence/published_reference_results.json`: published SkillRet anchors, labelled as not reproduced.
- `evidence/VERIFICATION_NOTES.md` and `evidence/SEARCH_SCOPE.md`: evidence boundaries and literature-search scope.
- `evidence/PLAN_VALIDATION.json`: actual offline package-consistency test output.
- `tools/validate_plan.py`: rerunnable offline consistency checks; it does not call models or APIs.
- `MANIFEST_SHA256.json`: byte-integrity hashes for packaged files, excluding the manifest itself.

## Validate the planning package

Requires Python 3.10 or newer; the validator uses only the standard library, makes no network requests and creates no cloud resources.

```bash
python tools/validate_plan.py
```

Passing this check establishes that the planning registries, references, work-order dependencies, resource guards and original-proposal hash are internally consistent. It does **not** establish that a model fits on a GPU, a provider account is connected, official metrics have been reproduced, or an experiment will succeed.

## First resources to allocate

Authorize the local repository/data/metric audit first. A suggested next external-spend envelope is **C$50 for screening**, followed by a separately approved **C$200 total balanced study** or **C$500 total extended study** only after measured pilots. These are suggested ceilings, not vendor quotations, completed allocations or guarantees of scope.

## The success condition

A supported improvement over a **reproduced, pinned baseline**, with matched candidates/evidence, explicit public-test exposure, confidence intervals, preserved failures, and lower or justified resource cost. Keep a quality winner and an efficient winner where different. A documented negative result is preferable to an impressive but incomparable score.
