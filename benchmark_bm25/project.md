# MIA5150 Project — Agent Handoff Context

## 1. Purpose of This Document

You are joining an MIA5150 university project titled:

**Agent Skill Discovery at Scale**

Your role is to help the team design, execute, analyze, and document an empirical study of **skill discovery and selection for AI agents**.

This document provides the context required to work on the project without relying on previous conversations.

---

# 2. Project Overview

Modern AI agents can be given access to libraries of reusable **skills**. A skill may contain instructions, workflows, tools, or domain-specific knowledge that an agent can activate when needed.

As the number of available skills grows, an important problem emerges:

> **How should an AI agent efficiently discover and select the correct skill or set of skills from a large skill library?**

Giving an agent the complete skill library in its context may work for small libraries, but this approach becomes increasingly expensive and potentially less effective as the library grows.

The project therefore studies different strategies for **skill discovery/retrieval before skill activation**.

The core comparison involves four discovery approaches:

1. **Listing**
2. **BM25 retrieval**
3. **Vector retrieval**
4. **Domain-vector retrieval**

The objective is to evaluate how these approaches behave as the skill library increases in size.

This is primarily a **benchmark and empirical study**, not a project whose main goal is to train a new machine-learning model.

---

# 3. Central Research Problem

The project investigates whether retrieval-based approaches can improve agent skill discovery compared with simply exposing the agent to a list of available skills.

The broad research question is:

> **How do different skill discovery strategies affect an AI agent's ability to identify the correct skills as the available skill library scales?**

The exact research claim/hypothesis for the final proposal still needs to be finalized.

Do **not** invent a research claim and present it as approved. Suggestions are welcome, but clearly distinguish suggestions from finalized team decisions.

---

# 4. Conceptual Pipeline

The intended architecture can be understood approximately as:

```text
User Request
     |
     v
Skill Library
     |
     v
Discovery / Retrieval Method
     |
     +-----------------------------+
     |            |        |       |
   Listing      BM25     Vector  Domain-Vector
     |            |        |       |
     +------------+--------+-------+
                  |
                  v
         Candidate Skills
                  |
                  v
          Agent / Selector
                  |
                  v
          Selected Skill IDs
                  |
                  v
          Evaluation Layer
```

A key experimental principle is that the **selection boundary should remain comparable across discovery methods**.

The retrieval/discovery mechanism changes, while the downstream selection process should be controlled as much as possible.

---

# 5. Skill Discovery Methods

## 5.1 Listing

The agent is presented with a listing of available skills and must determine which skill or skills apply to the request.

This serves as a direct-selection baseline.

Questions include:

* How does listing perform as the library becomes larger?
* Does context size degrade selection quality?
* What are the token and cost implications?

---

## 5.2 BM25

BM25 is a lexical information-retrieval method.

Instead of exposing the entire skill library to the selector, BM25 retrieves a limited set of candidate skills that appear most relevant to the request.

The selector then chooses among those candidates.

Important variables include:

* skill representation,
* skill name,
* skill description,
* skill body,
* metadata truncation,
* number of retrieved candidates (`k`).

---

## 5.3 Vector Retrieval

Skills and queries are represented using embeddings.

The query embedding is compared with embeddings representing the skills, and the most semantically similar skills are retrieved.

The selector then receives those candidates.

This allows comparison between lexical retrieval and semantic retrieval.

---

## 5.4 Domain-Vector Retrieval

This is intended to introduce an additional routing/retrieval stage.

Conceptually:

```text
Request
   |
   v
Domain Identification / Routing
   |
   v
Domain-Specific Candidate Space
   |
   v
Vector Retrieval
   |
   v
Candidate Skills
   |
   v
Selector
```

The exact implementation of this experimental arm still needs careful definition.

An earlier pilot version used a gating mechanism that may not be sufficiently independent from previous outputs. A future version should use a cleaner, separately defined protocol.

Do not automatically combine historical gated results with results from a redesigned domain-vector method.

---

# 6. Skill Discovery vs. Skill Activation

These are separate concepts.

## Discovery

Determine which skill or skills are relevant.

Example:

```text
Request:
"Analyze this CSV and produce a visualization."

Possible discovery result:
["data-analysis", "visualization"]
```

## Activation

After a skill has been selected, the system actually reads or activates the full skill instructions.

The project should avoid confusing:

```text
retrieval
```

with:

```text
selection
```

and with:

```text
activation
```

These may require different measurements.

---

# 7. Experimental Data

The benchmark uses a library containing skills plus benchmark queries.

Each query should have an annotated reference set of relevant skills.

Conceptually:

```json
{
  "query": "Example user request",
  "gold_skills": [
    "skill_A",
    "skill_B"
  ]
}
```

For a query \(q\):

* \(G_q\) = annotated/gold skill set
* \(C_q\) = retrieved candidate set
* \(S_q\) = final valid selected skill set

These sets allow retrieval and final selection to be evaluated separately.

---

# 8. Main Metrics

The project should distinguish **candidate retrieval quality** from **final selection quality**.

## Candidate Recall

Candidate recall asks whether the retrieval stage successfully placed the correct skills in front of the selector.

$$
\text{Candidate Recall}
=
\frac{|C_q \cap G_q|}{|G_q|}
$$

If the correct skill never enters the candidate set, the downstream selector cannot choose it.

---

## Selection Precision

$$
\text{Precision}
=
\frac{|S_q \cap G_q|}{|S_q|}
$$

This measures how many selected skills were actually relevant.

---

## Selection Recall

$$
\text{Recall}
=
\frac{|S_q \cap G_q|}{|G_q|}
$$

This measures how many of the required skills were successfully selected.

---

## F1

Precision and recall should also be summarized using F1 where appropriate.

---

## Exact-Set Match

A strict success occurs when:

$$
S_q = G_q
$$

This is important because an output containing the correct skill **plus unnecessary skills** should not necessarily count as an exact success.

---

# 9. Scaling Dimension

The project is specifically interested in what happens as the skill library grows.

The experiments therefore use multiple library sizes.

An earlier pilot included sizes up to approximately:

```text
N = 1000 skills
```

The final experimental design should preserve a controlled scaling comparison.

The exact final library sizes must be confirmed from the approved protocol/code before running new experiments.

Do not silently change library sizes.

---

# 10. Additional Measurements

Selection accuracy alone is not sufficient.

The study should also record where possible:

* input tokens,
* output tokens,
* total token usage,
* API/model cost,
* latency,
* failed calls,
* malformed outputs,
* invalid skill IDs,
* provider failures,
* rate limits,
* exclusions,
* attempted cases,
* valid cases,
* planned cases.

It is particularly important not to report accuracy only over successful API calls while hiding failures.

Results should make clear the difference between:

```text
Planned
Attempted
Valid
Successful
```

cases.

---

# 11. Pilot Work Already Completed

A pilot study has already been executed.

It should be treated as **feasibility evidence and exploratory work**, not automatically as the final experiment.

Existing pilot work includes:

* BM25 retrieval experiments,
* vector/BGE retrieval experiments,
* multiple library sizes,
* multiple selector configurations,
* skill integration tests,
* activation checks,
* cost tracking,
* reproducibility tests,
* saved traces,
* existing notebooks,
* automated tests.

The pilot demonstrated that the experimental infrastructure can execute the intended workflow.

It also exposed several methodological and infrastructure issues that should inform the final study.

---

# 12. Important Pilot Findings / Limitations

The earlier work showed several issues that must be considered in the final design.

These include:

### Provider reliability

Some model/provider configurations encountered:

* rate limits,
* overload,
* malformed responses,
* invalid skill IDs,
* incomplete experimental coverage.

### Small denominators

Some model/method combinations had too few valid observations to support strong conclusions.

Therefore:

> Do not derive broad model rankings from the exploratory pilot.

### Retrieval representation matters

BM25 performance changed when the representation of each skill changed.

For example, using:

```text
name + description + body
```

is not directly comparable to using:

```text
capped name + description
```

if the goal is to isolate the retrieval algorithm itself.

The final study should control the document representation when comparing retrieval methods.

### Domain-vector methodology

The earlier domain-vector implementation included a gating mechanism related to earlier outputs.

A cleaner future experiment should make this arm more independent.

### Embedding limitations

Some pilot queries were affected by embedding truncation.

This needs to be monitored in the final study.

---

# 13. Reproducibility

Reproducibility is an important project requirement.

The existing work includes or intends to preserve:

* fixed random seeds,
* source/configuration hashes,
* query reservations,
* exact dependency/source pins,
* sanitized traces,
* failure records,
* notebooks,
* automated tests.

The experimental protocol should be frozen before running the final evaluation.

Do not repeatedly modify the experiment after inspecting results unless the change is explicitly documented as a new protocol/version.

---

# 14. Data Leakage / Experimental Integrity

Gold/reference labels must remain evaluator-only information.

The selector should not receive the correct answers.

Conceptually:

```text
Query
  |
  +----> Discovery Method ----> Selector ----> Prediction
  |
  +------------------------------------------> Evaluator
                                                |
                                                v
                                            Gold Labels
```

Gold labels should be used only after the prediction is generated.

Skill instructions should also not be unintentionally executed during retrieval evaluation.

---

# 15. Query Sets

The exploratory/pilot queries should be treated as consumed if they have already influenced development decisions.

The final experiment should use a **disjoint held-out query set** where possible.

An earlier plan proposed:

```text
20 previously unused queries
seed = 42
```

with additional seeds potentially considered later.

However, this should be treated as a proposed design until confirmed by the team.

Do not run paid experiments simply because this number appears in this handoff.

---

# 16. Cost Controls

The project uses paid API/model calls, so experiments must remain bounded.

The existing harness includes mechanisms such as:

* USD spending limits,
* CAD planning limits,
* request caps,
* per-call cost reservations,
* restricted retries.

A critical rule:

> **Do not initiate new paid model/API calls without explicit authorization from the team.**

Existing results can be analyzed freely.

Any proposed additional experiment should first estimate:

```text
number of calls
estimated tokens
estimated USD cost
estimated CAD cost
```

before execution.

---

# 17. Failure Analysis

The final project should not only report aggregate metrics.

Failures should be categorized.

Useful categories include:

```text
Missing candidate
Wrong category/domain
Correct skill omitted by selector
Extra skill selected
Invalid skill ID
Malformed response
Provider failure
Rate limit
Context/token limitation
Embedding/truncation issue
```

This error taxonomy may help explain why a method succeeds or fails.

---

# 18. Statistical Analysis

Where sample size permits, comparisons should be performed on **paired queries**.

For example, if the same query is evaluated under BM25 and vector retrieval, compare outcomes for that same query rather than treating all observations as independent.

Potential analyses include:

* paired differences,
* bootstrap confidence intervals over queries,
* method-level precision/recall/F1,
* exact-match rates,
* retrieval recall,
* performance by library size,
* cost by method,
* token usage by method,
* latency by method.

Avoid claiming a reliable scaling law unless the evidence actually supports one.

---

# 19. Backup Experimental Plan

If live LLM/provider reliability prevents a complete large experiment, a fallback design can emphasize retrieval evaluation.

Possible backup:

```text
Full-corpus BM25
        vs.
Full-corpus embedding retrieval
```

using matched skill metadata.

This could then be followed by a smaller native agent-selection experiment.

This preserves a meaningful comparison even if large-scale live model calls are unreliable.

---

# 20. Current Proposal Structure

The LaTeX project proposal currently uses the following skeleton:

```text
Abstract

Research claim to test

1. Problem and Related Work

2. Experimental Design
   - Data and Library Construction
   - Four Methods, One Selector
       - Listing
       - BM25
       - Vector
       - Domain-vector
   - Native Integration and Controls
   - Experimental Flow Figure

3. Executed Pilot and Feasibility
   - Retrieval Baselines Table

4. Evaluation and Next-Stage Plan
   - Metrics and Interpretation
   - Sampling and Analysis
   - Resources, Risks and Backup
   - Project Timeline and Responsibilities Table

Reproducibility and Disclosure

References
```

The proposal is intended to remain approximately **2–3 pages excluding references**, subject to the course requirements.

---

# 21. Work That Still Needs to Be Done

Your role as an additional agent is to help complete the project systematically.

The major remaining tasks are:

1. **Review the project methodology**

   * Identify confounds.
   * Check whether the four methods are compared fairly.
   * Verify separation between retrieval, selection, activation, and evaluation.

2. **Help finalize the research question/hypothesis**

   * It must be measurable.
   * It should align with the actual experimental design.
   * Avoid claims broader than the available evidence can test.

3. **Review the benchmark construction**

   * Skill library construction.
   * Distractor selection.
   * Gold/reference labels.
   * Single-positive vs. multi-positive queries.
   * Held-out query integrity.

4. **Review the metrics**

   * Candidate recall.
   * Selection precision.
   * Selection recall.
   * F1.
   * Exact-set match.
   * Costs/tokens/latency.
   * Failure coverage.

5. **Review the scaling experiment**

   * Confirm library sizes.
   * Ensure the same queries are used for paired comparisons.
   * Ensure library construction is controlled.

6. **Improve the domain-vector method**

   * Define the routing stage clearly.
   * Avoid dependence on earlier selector outputs.
   * Ensure it can be compared fairly with other methods.

7. **Review pilot evidence**

   * Separate observed measurements from proposed future work.
   * Do not treat exploratory findings as confirmatory evidence.

8. **Design the final experiment**

   * Specify query count.
   * Specify library sizes.
   * Specify candidate count `k`.
   * Specify model/provider configuration.
   * Specify random seeds.
   * Specify failure/exclusion rules.
   * Specify budget before execution.

9. **Support statistical analysis**

   * Use paired comparisons.
   * Include uncertainty when justified.
   * Avoid rankings unsupported by sample size.

10. **Support final reporting**

    * Proposal.
    * Final report.
    * Tables.
    * Figures.
    * Error analysis.
    * Reproducibility documentation.

---

# 22. Rules for Working on This Project

When helping with this project:

### Do not invent results.

Clearly label information as one of:

```text
Observed
Planned
Proposed
Hypothetical
```

### Do not overwrite pilot evidence.

Historical results must remain traceable.

### Do not silently change the protocol.

If you recommend a change, explain:

```text
Current approach
Problem
Proposed change
Expected benefit
Possible tradeoff
```

### Do not make paid API calls without authorization.

First provide the estimated experiment size and cost.

### Preserve reproducibility.

Whenever changing experimental code or methodology, record:

```text
What changed
Why it changed
Which results it affects
Whether old and new results remain comparable
```

### Be skeptical of apparent improvements.

If one method performs better, first check for:

* unequal metadata,
* different candidate counts,
* different prompts,
* different model configurations,
* unequal failure rates,
* data leakage,
* different denominators,
* different query subsets.

---

# 23. Recommended Immediate Next Step

Before proposing new experiments, inspect the existing project artifacts and reconstruct the **actual current experimental protocol**.

Produce a concise protocol summary covering:

```text
Research question
    ↓
Dataset / skill library
    ↓
Query set
    ↓
Library sizes
    ↓
Discovery methods
    ↓
Candidate count
    ↓
Selector/model
    ↓
Output parsing
    ↓
Evaluation
    ↓
Metrics
    ↓
Failure handling
    ↓
Cost accounting
```

Then identify any differences between:

1. what the existing code actually does,
2. what the pilot did,
3. what the proposal says,
4. what the final experiment should do.

Do not assume these four are currently identical.

---

# 24. Expected Behavior From You

Act as a **technical research collaborator and critical reviewer**, not merely a writing assistant.

Prioritize:

* experimental validity,
* fair comparisons,
* reproducibility,
* clear metrics,
* statistical defensibility,
* cost control,
* traceability between code and reported results.

When you find a methodological problem, flag it explicitly and propose a concrete correction.

When information is missing, state what must be verified rather than guessing.

The immediate objective is to turn the existing pilot and project infrastructure into a **well-controlled, reproducible empirical study of agent skill discovery at scale**.
