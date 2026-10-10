# Codex task — implement AND run the MIA5150 two-day trial

## Outcome

You are working in my local Mac project. Build and execute a small, understandable Python pilot for **Agent Skills discovery** using **LangChain Deep Agents, OpenRouter, and ThakiCloud/SKILLRET**. Do not stop at a plan, generated files, mocked responses, or an import test. Obtain real data, run a real retrieval baseline, and, when local credentials and spending permission are available, run the real API-backed benchmark. End with measured results and a feasibility report that our team can use for the proposal.

This is a two-day feasibility trial, not the final semester experiment. The handbook says to open the data, run the simplest baseline, and print a number before tuning or building abstractions:
https://stellar-custard-c9ea93.netlify.app/#seeds

Read START_HERE.md, AGENTS.md, SOURCE_NOTES.md, and pilot.defaults.json first. Treat the latter as defaults, not as observations. Inspect the existing workspace before changing anything; preserve working code and prior evidence.

## Scope and research question

Main variable: **skill-library size N**. Compare:
1. `listing`: all N skill metadata entries are visible to the same selector.
2. `vector`: global dense retrieval, then the top k metadata entries are visible.
3. `domain_vector`: predict one domain from the query, search within that domain, then expose up to k entries.
4. `bm25`: a lexical retrieval reference, followed by the same selector in the small live trial.

The main three are listing/vector/domain-vector; BM25 is the reference requested by our instructor. Start with BM25, then listing/vector, then the simple domain router. No knowledge graph, Neo4j, multi-agent comparison, agent-skill chains, fine-tuning, reranker, elaborate UI, or custom agent platform. Do not conflate retrieving or selecting skills with successfully executing them.

The hypothesis is NOT that one method must win. Investigate accuracy/cost as N grows and whether domain routing helps or discards relevant skills. Bad performance with a functioning harness is a valid trial result.

## Local setup and dependencies

Use Python 3.12 in project-local `.venv`; prefer `uv` if installed. Otherwise use a normal venv and pip. No sudo, global installs, Homebrew changes, or changes to my shell configuration. If a compatible Python is absent, report the exact blocker and a non-admin installation route; use only the user's allowed network/install mechanisms. Never bypass an execution sandbox or approval requirement.

Candidate dependencies: `deepagents`, `langchain`, `langchain-openrouter`, `langchain-core`, `datasets`, `huggingface-hub`, `sentence-transformers`, `numpy`, `rank-bm25`, `pydantic`, `python-dotenv`, `httpx`, `matplotlib`, `pytest`, and notebook support. Keep them minimal. Resolve a mutually compatible stable set on this Mac, record exact installed versions, and produce a lock file. Do not copy an old version guess from this handoff. Read the installed signatures and official source if live docs differ.

Use ordinary functions, a small CLI, and exactly two thin explanatory notebooks:
- `01_data_and_retrieval.ipynb`: dataset inspection, BM25/vector baselines, examples.
- `02_deepagents_trial.ipynb`: a real agent smoke test, benchmark invocation, results.

Both should call shared functions rather than duplicate the experiment. A few Python files are enough: `pilot.py`, `agent_runner.py`, and the supplied `reference_metrics.py` (or a clearly justified equivalent). Avoid an abstract plugin framework. Add `PLAN.md` with five short checkpoints and update it as you execute them.

## Credentials and spending — hard boundaries

Use OPENROUTER_API_KEY from the process environment or a git-ignored local `.env`. Never ask me to paste it into chat, print it, put it in code or a notebook, dump environment variables, inspect unrelated credential stores, or include secrets in logs/ZIPs. Do not `cat .env`. Check only presence through code. An HTTP client may read the secret to authenticate without exposing it to the model context.

A `setup_local_key.py` helper is supplied. I can run it manually in my own terminal; do not run its interactive key prompt inside the Codex conversation. If the key is absent, complete all data, retrieval, and fixture tests anyway, then report the single local action needed and exact resume command. The notebook must not prompt for a key or display it.

Live generation is disabled unless `ALLOW_PAID_RUNS=1` is explicitly present. Suggested trial guards are CAD 5.00 and a separate native USD 3.00 ceiling; both must hold. The supplied CAD-per-USD factor 1.50 is a labelled planning assumption, NOT a verified exchange rate. Let the user override it. Track native USD and estimated CAD separately. Do not claim a client-side estimate controls bank exchange fees or unrelated account use. Never increase limits or automatically buy credits.

Implement cost controls BEFORE any paid canary:
- Sequential calls only in the pilot; SDK automatic retries off (`max_retries=0`).
- Read current model/endpoint pricing and context/parameter support; save the public metadata snapshot.
- Reserve a conservative upper-bound cost using the full effective prompt, configured maximum output tokens and a safety margin before each request. Reject a request if the remaining budget cannot cover it. Account for routing as well as selection calls.
- Persist actual/reserved cost and request counts so restarts cannot reset the budget. Ambiguously failed requests retain their reserved cost until reconciled.
- Capture actual returned usage/cost; if the adapter drops them, use sanitized raw response handling or generation-ID lookup. Missing cost is unknown, never zero; stop paid expansion until it is reconciled or a conservative reservation is retained.
- Add a request-count cap, timeout, and a small explicit retry policy for transient errors only. Log and debit all actual attempts. Never retry invalid outputs until they happen to be correct.
- Enforce the same guard for every model call, including anything hidden in the framework. No unbudgeted summarizer or subagent calls.

If supported, use a dedicated user-created OpenRouter key with a provider-side credit limit as another protection; do not modify account settings yourself.

## Model preflight

Preferred default candidate: `qwen/qwen3-8b` through `langchain_openrouter.ChatOpenRouter`. This is a candidate, not a guaranteed working endpoint. Respect OPENROUTER_MODEL if I supplied one. Use one model across all arms; do not silently fall back to another model. Before paid runs, verify availability, tool calling, tool choice, prices, effective provider context limit and output limit. Pin one working provider with fallback disabled where supported, and record the actual provider, model identifier and completion IDs. If a provider cannot be pinned, explicitly mark routing variability as a limitation.

Keep temperature at 0 and generation settings fixed. Do not claim temperature 0 makes hosted runs bitwise reproducible. Disable reasoning using the endpoint's documented option where supported; otherwise record the actual behaviour and account for reasoning tokens. Do not merely hide reasoning and assume it was disabled. No web search/tool execution by the model provider. Do not use `openrouter/auto`, moving `:free` routes, or a model leaderboard sweep.

Prefer ChatOpenRouter; an OpenAI-compatible `ChatOpenAI` adapter is a narrowly allowed fallback ONLY if the native integration fails. Keep the same OpenRouter model/provider/settings and demonstrate tool/usage compatibility. Record the adapter change; do not silently replace Deep Agents with bare chat calls.

## Milestone 1 — get real SkillRet data and inspect it

Official sources:
- https://huggingface.co/datasets/ThakiCloud/SKILLRET
- https://huggingface.co/datasets/ThakiCloud/SKILLRET/raw/main/README.md
- https://github.com/ThakiCloud/SKILLRET

Resolve `main` to an immutable Hugging Face commit SHA, save it in a manifest, then use that SHA for every dataset load and cache. Load only the required configurations/splits initially:

```python
skills = load_dataset("ThakiCloud/SKILLRET", "skills", split="test", revision=sha)
queries = load_dataset("ThakiCloud/SKILLRET", "queries", split="test", revision=sha)
qrels = load_dataset("ThakiCloud/SKILLRET", "qrels", split="test", revision=sha)
```

Documentation inspected for this handoff describes `skills.id`, `name`, `description`, `skill_md`, `major`, `sub`, and a separate `domain` field; `queries.id`, `query`, `skill_ids`, `skill_names`, `k`; qrels `query_id`, `skill_id`, `relevance`. Inspect ACTUAL schemas; fail helpfully on drift. Do not manufacture missing fields or assume there is one gold skill per query. `major` is the six-way taxonomy; it is NOT the separate `domain` field.

Current documented v1.1 test counts are 6,006 skills, 4,392 queries and 7,187 qrels. Compute observed counts and report any difference instead of forcing these numbers. Check referential integrity, nonempty gold sets, duplicates, missing text, labels, lengths and licenses. Compare query-side gold IDs with qrels as an audit, outside agent state. Save a few readable examples and a dataset summary with source/revision/hash/provenance. The full dataset need not be copied into the final ZIP if too large; include download commands and exact manifests/checksums.

**Leakage boundary:** ONLY `query` is available to the router/selector. Never pass `skill_ids`, `skill_names`, `k`, `original_query`, source-generation metadata, qrels, answer counts, or evaluation-only annotations into a prompt. Gold labels live in the evaluator only. Do not ask a model to choose the number given by `k`.

Use a deterministic 200-query sample for the first real retrieval baseline. Reserve its IDs as `pilot_query_ids.json`; any test query inspected or used for prompts is pilot-consumed, not an untouched final holdout. Select 20 of these IDs for the live pilot (include single- and multiple-positive cases when available). Keep a disjoint final-query reservation file. Do not tune prompts against that reserved set. SkillRet's inputs are public skills and synthetic requests; disclose this rather than call them real user production logs.

## Milestone 2 — first baseline number before spending money

Clone/read the official SkillRet evaluation code at a pinned Git commit. Locate the actual BM25 implementation and evaluator; do NOT assume a script/function path or execute the repository's multi-GPU training scripts.

First run BM25 over the entire pinned test corpus for the deterministic 200-query sample. Match the official document representation (`name | description | skill_md`), preprocessing, parameters, and metrics where possible. Prefer reusing or faithfully porting the small evaluator rather than installing training-only CUDA dependencies on Mac. Record source file paths/commit and any changes.

If exact baseline code/settings cannot be recovered quickly, use `rank_bm25.BM25Okapi` with explicitly recorded tokenization and parameters; call it `local_bm25_baseline`, NOT a published-score reproduction. Print a real Recall@5 or Recall@10 with query count and corpus size. An official-style method on a query subset is a partial-protocol check, not exact reproduction of a full-test aggregate. A full official comparison is optional and only comparable when revision, corpus, all queries, representation, preprocessing and metric definitions match.

The web copies inspected while preparing this handoff contain different published BM25 numbers. Do not hardcode a supposed reference score or tune until yours matches. Resolve version provenance or state that numeric reproduction remains unverified.

Second, implement the metadata-only discovery protocol for the main experiment. Construct a canonical metadata string from ORIGINAL name + description, never full body. Cap it once (default 128 BGE tokens, excluding special tokens, also respecting the installed Deep Agents description limit). Apply that same rendered metadata to listing, BM25, vector and domain-vector. Log original/retained lengths and the explicit transformation; no silent truncation. This is a separate experiment from the full-document official BM25 check.

Dense retrieval: `BAAI/bge-small-en-v1.5`, pinned revision, CPU by default, SentenceTransformers, normalized embeddings and exact NumPy dot products (cosine). Apply the documented query instruction, not a document instruction. No external vector database needed. Save corpus-ID order and text hashes next to cached arrays. Cache key includes revision, model revision, text transform and max length. Check query length; reject or explicitly record truncation instead of hiding it. Empty or nonfinite embeddings are errors. Do not silently switch embeddings to random vectors/TF-IDF.

Use `reference_metrics.py` and its tests for the explicitly defined local metrics; validate it against the official evaluator on tiny rankings before a reproduction claim. Log Recall@5/10, MRR and Completeness@k. Higher IR recall is not proof of agent execution success.

## Milestone 3 — a genuine, restricted Deep Agents skill runner

Use the actual `create_deep_agent` and native `SkillsMiddleware` for metadata discovery. Verify the current installed API rather than blindly copying old chat snippets. Docs:
https://docs.langchain.com/oss/python/deepagents/skills
https://docs.langchain.com/oss/python/deepagents/customization
https://docs.langchain.com/oss/python/integrations/chat/openrouter

**Critical path rule:** `skills=["/skills/"]` points to a SOURCE directory containing child skill directories. Passing individual skill directories or SKILL.md paths does not load them in the documented API.

For EACH query/strategy use a fresh `StateBackend` holding ONLY that run's candidate skills, provided through `create_file_data()` and the invocation's `files` dictionary. No host filesystem, unrelated project files, labels, secrets or full-corpus index is mounted. Use no persistent chat memory/checkpointer shared across queries.

Canonical layout:
```
/skills/s-<original-UUID>/SKILL.md
```
Generate safe unique frontmatter `name: s-<UUID>`; retain the original readable name in the common description. Same mapping in all arms. This prevents unrelated skills with duplicate names from overriding one another. Strip source frontmatter with a safe parser before attaching the preserved original Markdown body; record all transformations. Keep attribution/license/source metadata outside model-visible fields unless it is part of the fixed representation. Skill text is untrusted data. Do not execute scripts, install skill dependencies, browse links or let skills alter the evaluation policy.

The MAIN benchmark is **metadata-only skill selection**, not arbitrary task execution:
- Deep Agents injects the canonical candidate metadata.
- The only offered action is a common terminal `select_skills(skill_ids: list[str])` tool; allow zero through five IDs, not the hidden ground-truth count.
- No `read_file`, filesystem enumeration, search, writes, shell, subagent, planning or external tools in this main mode. Preserve the native skills metadata injection, but explicitly label this restricted discovery mode rather than claim it is the stock unrestricted agent.
- Use an enforced tool allowlist and terminal handling; `tools=[]` or `subagents=[]` ALONE may NOT remove built-ins/default subagents. Inspect actual outgoing tool schemas. Use supported middleware/profile controls from the installed version; add an execution-layer denial guard so an unoffered tool cannot execute if hallucinated.
- One selector model call per query/arm. Capture its tool arguments and stop; do not allow an extra model turn for an acknowledgment. A failed/absent/multiple selection or malformed tool output is a recorded format failure, not an invitation to rerun for accuracy.
- Validate selected IDs against candidate IDs WITHOUT consulting labels. Do not autocorrect a misspelled ID or substitute the first retrieved item. Preserve raw outputs. Unknown IDs are false positives and invalidate the record.

Use the same selection instruction everywhere: choose skills needed for the request; return IDs; do not perform the task. No strategy name, similarity scores or hidden answer hints in the prompt. Shuffle candidate display by a seeded order independent of relevance so retrieval rank is not accidentally an extra feature in only some arms.

Before any API test, use a scripted fake chat model with the REAL Deep Agents construction to capture the effective model messages and tools. Check: exactly the intended metadata entries appear, non-candidates and body-only sentinels do not, duplicate names remain distinct, qrels are absent, and only the terminal selection tool is offered.

Then run one credentialed canary on tiny hand-authored skills with an obvious selection. This validates integration only, not SkillRet accuracy. Log provider usage and actual task wiring. If it fails, debug the adapter/tool contract before batch calls.

A SEPARATE small `activation_smoke` mode must also demonstrate native progressive disclosure: allow `read_file` plus `select_skills`, require at least one selected candidate SKILL.md to be read, log the read trace and selected IDs, and stop within four model calls and two bounded reads. Use the same isolated StateBackend. No shell/writes/subagents/resources. Run this on two real SkillRet queries and three small hand-authored fixtures. Missing linked resources are expected; do not fetch them. Body read truncation must be explicit. Activation results must NOT be mixed into the metadata-only main table or called execution success.

If native skill loading cannot be made to work in the stable installed release, stop that portion with a traceable blocker. A custom prompt-listing fallback may be a diagnostic but must not masquerade as a Deep Agents result.

## Milestone 4 — bounded live scaling trial

Library sizes [50, 250, 1000]; shortlist k=5; 20 fixed real queries; one sampling/order seed (42) initially. Twenty queries × three sizes × four strategies = 240 planned selector decisions, not 240 guaranteed API calls: preflight exclusions, budget stops and the separate router/smoke requests must be accounted for. Start with five queries at N=50 for all arms, then expand; reuse already completed compatible cases instead of charging twice.

Construct **per-query nested candidate libraries** using the gold set ONLY in the experimental sampler: retain all labelled relevant skills and add randomly sampled distractors as N grows. Use the supplied tested helper. Save exact memberships and display order. Never feed the sampler's reasons/labels to the agent. Describe this as a gold-preserving, conditional distractor-scaling study, NOT the official full-corpus SkillRet benchmark or a shared global production library. Do not silently change to top-ranked hard-negative sampling. A hard-negative stress test is future work.

Method details:
- listing exposes all N canonical metadata entries.
- BM25/vector select min(k,N) from that N-sized corpus, using only canonical metadata.
- domain-vector predicts one `major` category from the query and a fixed six-category description list. Use a small separate call to the SAME OpenRouter model, with a fixed categorical output contract. Never derive the predicted category from gold IDs or query labels. Restrict exact vector search to that predicted category within the same N-sized corpus. If empty, return an empty set, record a routing failure, and do not secretly fall back to global search. Log routing retention for diagnosis using labels AFTER the decision.
- Cache routing once per identical query/config for physical execution efficiency. Distinguish actual billed cost from standalone logical pipeline cost: an independent domain-vector workflow still needs that routing decision. Do not make its per-query token cost look artificially free because the pilot reused a router response across N.

Keep all other settings fixed. Record model/provider changes as new run configurations, never mixed results. Evaluate single-/multi-positive queries separately as descriptive diagnostics, not additional tuning axes.

Inspect the exact effective messages AFTER middleware, including tool schemas and framework text. Estimate input token length using the corresponding tokenizer where available; report estimates as such and use provider counts as actual usage. Preflight context and cost. If full listing will not fit, record `context_limit` with N and estimated size; do not truncate the catalogue, drop the correct skill, auto-summarize or enable context compression. Disable summarization or assert it never occurs. Larger context limits do not establish good long-context performance.

Log latency as wall-clock time per logical workflow, with retrieval, routing and selector components; do not call output_tokens / wall_time pure model generation throughput. Record cached tokens/usage details when provided. Token accounting includes input/output for every stage and repeated inputs, not just the first system prompt. No LangSmith/cloud tracing required; keep local sanitized traces.

## Metrics and evidence

Persist rows immediately to JSONL, with a stable case key from dataset/representation/model/provider/config/query/N/strategy/seed hashes. Resume only exact matching configurations. Append attempts rather than overwrite failures. Finished cases must not trigger paid calls on a second run. Never purge previous results as a cleanup step.

Each row includes query ID, run/config IDs, dataset revision, N, k, candidate IDs, returned IDs, raw sanitized response, parsed selection, expected gold set in an EVALUATOR-ONLY result field, status, first-stage metrics, selection metrics, input/output/cached/reasoning tokens where available, request IDs, actual USD cost or unknown/reserved status, CAD conversion assumption, wall-time components, error category and attempt count. Agent-visible objects must never contain the evaluator-only fields.

Distinguish:
- candidate Recall@k / Completeness@k: retrieval quality (listing coverage is trivially 1 by construction).
- final precision/recall/F1 and exact-set match against SkillRet relevance labels.
- malformed/unknown/duplicate-ID outputs and tool/transport failures.
- conditional semantic accuracy on valid completions, service reliability on attempted cases, and planned-case coverage including budget/context skips. Do not hide failures by reporting only successes; also don't label an API outage as pure LLM reasoning failure.

Use the supplied tiny metric examples as known-answer controls. Empty predictions with nonempty truth score 0; retrieval rankings cannot gain points from repeated IDs; no substring name matching; unknown IDs remain wrong. Compare a few records by hand.

## Milestone 5 — proposal-ready report and handoff

Generate `reports/TRIAL_REPORT.md` and CSV tables FROM persisted result rows, not memory. Include dataset inspection, exact protocol, observed metric and denominators, official-baseline reproduction status, observed costs, versions, failures, and a practical GO / CONDITIONAL GO / NO-GO assessment. State whether the evidence supports feasibility, not a general claim of superiority. No p-values or sweeping scale conclusions from 20 exploratory queries. Do not invent expected success rates, citations, performance or resource use.

Produce three separate matplotlib figures (default colours): selection exact-match versus N, total logical LLM tokens versus N, and candidate recall versus N. Label omissions/context limits; don't join fabricated missing points. Separate activation smoke outputs.

Add `reports/PROPOSAL_NOTES.md`: candidate claim, benchmark category, known dataset limitations, the actual trial-run number with protocol/revision, measured call-budget arithmetic for a larger study, no-training clarification, team-role placeholders, and smaller backup (BM25 vs vector + a smaller listing study). Keep final-holdout and external-validity limitations explicit. Related-work entries must be verified; unreviewed suggested papers stay marked unreviewed. Do not write a fake completed NeurIPS report.

Run `python -m pytest -q`; show collection count and failures. Do NOT execute pytest files directly and treat exit 0 as test execution. Execute both notebooks; when API access is blocked, their API cells must clearly record SKIPPED/BLOCKED and their real-data baseline cells should still run if downloads work. A separate opt-in flag must control paid notebook execution, so reopening a notebook does not silently rerun charged calls.

Required documented CLI flow (adapt arguments if needed; keep this simplicity):
```
python pilot.py doctor
python pilot.py download
python pilot.py inspect
python pilot.py baseline --queries 200
python pilot.py smoke --live
python pilot.py run --config pilot.defaults.json --resume --live
python pilot.py activation-smoke --live
python pilot.py report
python -m pytest -q
python pilot.py package
```

For each command provide its expected output files, not fabricated numeric values. Include one smaller `--quick` mode for five queries and N=50. `--live` alone does not override missing key/ALLOW_PAID_RUNS/budget.

Package everything related into ONE ZIP with START_HERE.md: source, prompt/spec, instructions, two notebooks, pinned dependencies, tests, configs, sanitized traces, results, charts, manifests, reports, and verified source notes. Exclude `.env`, secrets, virtual environments, model weights and unnecessary bulk caches. Use an explicit export allowlist and secret scan; never blindly zip the workspace. Include a dataset downloader and manifest when raw corpus redistribution/size is inappropriate. State clearly that rerunning needs dependencies, downloads, a key and API credits; do not call the result an offline self-contained benchmark.

## Acceptance and stop rules

Successful pilot means: real data inspected; a real BM25 number printed; a real OpenRouter response with usage; native SkillsMiddleware exposure verified; at least a small set of real SkillRet selector decisions across listing/BM25/vector; domain-vector attempted only after that works; native SKILL.md read demonstrated separately; tests pass; traces/costs/results retained; report generated. The entire 240-decision sweep is optional when time, context or budget prevents it.

A blocked pilot is also an honest deliverable: complete everything feasible, preserve the concrete exception/request/config, explain the blocker, and provide one exact resume command. Never substitute random retrieval, canned API answers or synthetic fixtures for real experiment evidence. Do not claim an unrun component works.

Proceed autonomously on routine reversible work within this project and the stated limits. Ask only for a genuinely missing secret setup action, permissions, model choice if the requested model cannot work, or budget approval beyond the default guard. Stop at boundaries; do not work around account or host protections.

At the end tell me: what you ran; the first real number and denominator; whether the live agent and activation smoke worked; actual USD and estimated CAD spend; where the report/ZIP are; unresolved issues; and whether the team has enough evidence to write a feasibility-grounded proposal.
