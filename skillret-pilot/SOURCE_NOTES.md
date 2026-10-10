# Source checks and implementation cautions

Inspected 2026-09-26 using web retrieval. These notes identify documented interfaces and design consequences; they do not certify the future installation or any benchmark performance. Resolve immutable versions on the target Mac.

## 1. Course trial

https://stellar-custard-c9ea93.netlify.app/#seeds

The two-day trial section asks students to obtain/inspect data, run the simplest baseline and report one number. It explicitly discourages tuning and elaborate abstractions at this stage. Record failures and changes if the trial is blocked. The supplied project brief likewise values a controlled empirical comparison, including negative results. The task therefore stages a simple CPU result before agent development.

## 2. Deep Agents skills

https://docs.langchain.com/oss/python/deepagents/skills

Documented API details: sources contain child skill folders, each with SKILL.md; source paths are not individual skill-file paths. Metadata is discovered through SkillsMiddleware. Full instructions are read later through read_file. StateBackend is in-memory and accepts file data made with create_file_data. Later duplicate skill names override earlier ones. Only configured sources are loaded. These facts motivate one isolated candidate-only source root and unique canonical names.

This corrects an earlier illustrative suggestion in the conversation to pass a list of individual retrieved skill paths: do not copy that into the implementation without checking source-directory semantics.

## 3. Deep Agents defaults and customization

https://docs.langchain.com/oss/python/deepagents/customization
https://raw.githubusercontent.com/langchain-ai/deepagents/main/libs/deepagents/deepagents/graph.py
https://raw.githubusercontent.com/langchain-ai/deepagents/main/libs/deepagents/deepagents/middleware/skills.py
https://docs.langchain.com/oss/python/langchain/middleware/built-in

The graph source inspected declares default filesystem, execution and subagent tools, with backend-dependent execution. Supplied tools are additive; an empty list is not a reliable removal mechanism. The current source documents profile/middleware controls and a default general-purpose subagent. The implementation must inspect its INSTALLED release and verify outgoing tools. Main selection and activation smoke use different, explicitly documented allowlists. Guard every real model call; do not permit implicit extra agents or summarization to change the experiment.

## 4. LangChain OpenRouter integration

https://docs.langchain.com/oss/python/integrations/chat/openrouter

Documents `langchain_openrouter.ChatOpenRouter`, native tool binding, token usage on returned messages, response metadata and provider-routing controls. Inspect which cost fields survive the installed adapter rather than assuming a particular dictionary nesting. Native supported parameters vary with the selected provider/model.

## 5. OpenRouter usage, routing and generation metadata

https://openrouter.ai/docs/cookbook/administration/usage-accounting
https://openrouter.ai/docs/guides/routing/provider-selection
https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation
https://openrouter.ai/docs/api/api-reference/api-keys/get-current-api-key

Usage accounting documents token and cost fields. Generation metadata is available through `GET https://openrouter.ai/api/v1/generation?id=...` and includes model/provider/cost information. Provider routing allows parameter-support requirements and restrictions/fallback controls. These support observability and consistency, but do not prove client-side estimates exactly bound billing in every failure mode. Keep conservative reservations for unresolved requests and prefer a user-configured provider-side key cap.

## 6. SkillRet schema and protocol

https://huggingface.co/datasets/ThakiCloud/SKILLRET
https://huggingface.co/datasets/ThakiCloud/SKILLRET/raw/main/README.md
https://github.com/ThakiCloud/SKILLRET

The inspected card calls the current data v1.1 and documents test counts of 6,006 skills / 4,392 queries / 7,187 qrels. It has separate `skills`, `queries`, `qrels` configurations; the main task uses test corpus/queries/qrels. The official document representation is name + description + skill_md. Queries include their own answer IDs/names/count: never put these in a model prompt. Taxonomy `major` (six categories) differs from the separate `domain` field. Skill documents come from public repositories, while requests are synthetic. The metadata/query licence is Apache-2.0; source skills retain MIT/Apache-2.0 licences according to the card. Preserve provenance and verify actual records.

The v1.1 card says its evaluation split was filtered and is not directly comparable with v1.0. Web-retrieved GitHub HTML and raw README copies displayed different BM25 scores during this check. Therefore this handoff deliberately supplies NO hardcoded expected benchmark score. Codex must pin the data and baseline source, match the protocol, and only then compare compatible reported/reproduced numbers.

The repository README advertises GPU-heavy embedding/training scripts; running those is not required for a small CPU BM25 reproduction. The exact BM25 source entry point was not verified in this authoring session. Locate it locally or clearly label an independent implementation.

## 7. Small embedding model

https://huggingface.co/BAAI/bge-small-en-v1.5

The card documents a 384-dimensional representation, 512-token sequence limit and the query prefix `Represent this sentence for searching relevant passages: `. This is a practical pilot candidate, not a measured winner. Apply a common explicit metadata cap and log input truncation. Do not compare this capped-metadata experiment numerically with a full-skill-document reference as though inputs were identical.

## 8. Suggested selector candidate

https://openrouter.ai/qwen/qwen3-8b/api

The inspected page lists tool calling for `qwen/qwen3-8b`. Do not infer that every provider supports every desired option or that advertised maximum context is the active endpoint's context. Retrieve current metadata, perform a canary and pin the effective endpoint. This handoff intentionally does not freeze a current price or declare live compatibility tested.

## 9. Codex task organization

https://developers.openai.com/blog/run-long-horizon-tasks-with-codex

The official example uses a persistent task/spec document with goals, boundaries, deliverables and acceptance checks, together with a milestone plan. This handoff uses that pattern without requiring additional plugins or a particular Codex version.

## Scope boundary

The previous team discussion proposed Graph of Skills, Li's scaling paper, Skill Shadowing and SRA as related work. Reviewing and verifying those is relevant to the proposal but is not a prerequisite to getting the first real baseline number. This handoff does not certify their novelty claims or reproduce their experiments.

## Local execution verification (2026-09-26)

- Dataset actually downloaded at immutable HF SHA `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`; observed counts 6006/4392/7187. The recorded README/source revision is authoritative over any mutable-head note above.
- Official Git source pinned to `0cc508e6bfcfff1fe6352d90ff5957ba38a64bba`. Located `skillret/eval.py:build_skill_text`, `eval_bm25`, `trec_eval`. Ported only CPU BM25 logic; no training/GPU scripts executed. Upstream bm25s lock is 0.3.3; local 0.3.11 is recorded as a protocol difference. Three known-answer controls agree for recall/precision/NDCG/completeness. No published-score reproduction claim.
- Installed versions: Deep Agents 0.7.19, LangChain 1.4.2, langchain-openrouter 0.2.9, OpenRouter SDK 0.11.46. Exact dependency freeze and installed source hashes are in the handoff.
- Real native metadata loading and native `read_file` were exercised. Profile controls disable the default subagent and summarization. `FilesystemMiddleware` requires read_file scaffolding; supported tool exclusion removes it from the metadata-only model. Both effective tool schemas and execution denial are tested.
- Public endpoint snapshot for qwen/qwen3-8b advertises Alibaba, 131072 context, 98304 maximum prompt, 8192 output. Auto tool choice is supported; required/function choice is not. Public prices in the snapshot are USD 0.000000117/input token and 0.000000455/output token; they are not measured charges and are refreshed before live runs. [Provider controls](https://openrouter.ai/docs/guides/routing/provider-selection) and [reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) were reviewed.
- Native adapter wire tests uncovered SDK serialization differences: provider max_price must use decimal strings; SDK reasoning schema supports effort=none rather than enabled=false; raw HTTP capture recovers the provider field dropped by its response schema. No ChatOpenAI fallback was used. Fixture HTTP tests do not prove live availability.
- Timeout is milliseconds at this SDK boundary; 90000 ms is verified as a 90-second transport timeout. SDK retry_config=None and per-call retries=None supplement max_retries=0; a mocked 503 produces one attempt.
- BGE revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` and Qwen tokenizer revision `b968826d9c46dd6066d109eabc6255188de91218` are pinned. Context estimates use the tokenizer with a framing margin; cost reserves use a larger UTF-8 byte bound. Token estimates are not provider usage.
- No local key/paid permission was available. OpenRouter canary, semantic selector comparison, domain router and live activation remain unrun. Actual paid spend is zero. Consult saved results and current report, not these interface notes, for metrics and denominators.

## User-authorized Ollama follow-up

The user explicitly requested the two locally installed Ollama models instead of OpenRouter for testing. This authorizes a separate local-inference protocol; it does not assert that the original OpenRouter run happened. The paid runner and its guards remain intact.

Read-only localhost discovery found Ollama 0.34.4 with qwen3.5:4b-mlx (4.5B) and qwen3.5:9b-mlx (9.4B), both nvfp4/safetensors with advertised tools and thinking support. Exact digests and API metadata are saved per run. No models were pulled or modified.

The native ChatOllama integration is langchain-ollama 1.1.0 with ollama 0.6.2. Official [chat API](https://docs.ollama.com/api/chat), [tool calling](https://docs.ollama.com/capabilities/tool-calling), and [ChatOllama docs](https://docs.langchain.com/oss/python/integrations/chat/ollama) were checked against installed source. Output limits belong in the adapter's options/num_predict, not a top-level Client.chat argument; the first canary attempt failed locally before HTTP and was retained. Transport fixtures verify think=false, context/output bounds, model name, tool schemas, usage and no retries. Fixture checks are not model results.

Local tests use only http://127.0.0.1:11434, with environment proxies disabled, sequential generations, no automatic fallback, no cloud model tags, no paid API, and no downloaded skill execution. A separate persistent local request cap bounds resource use. Temperature=0/seed=42 do not guarantee bitwise reproducibility. Model/engine differences prevent treating local results as OpenRouter reproduction. Read `reports/OLLAMA_TRIAL_REPORT.md` for measured outcomes.

Both initial real model canaries chose the correct fixture but returned the native s- prefix. Strict validation rejected them; no output correction occurred. A single fixed local contract revision adds a UUID-format clarification and schema pattern for both models and every method. The first clarification preflight exposed native system content as blocks rather than a string (TypeError before HTTP); block preservation is now tested. These failures remain in evidence/ollama and results/ollama. Prompt revision 2 results must not be mixed with original canaries.

Local trace review: failed graph summaries can omit completed reads. Read-only reporting now deduplicates actual native ToolMessage outputs by tool-call ID, cross-checks structured read traces when available, and excludes rejected read requests. The raw failed rows remain untouched. Three real candidate reads were observed for 9B before later pagination/out-of-candidate denials. The local run hash covers inference/retrieval code before the report function; changes only to reporting and trace recovery do not rerun inference. Exact replay added zero requests.

## Five-model OpenRouter follow-up (2026-09-26)

User-requested model IDs were verified through public `/api/v1/models/<model>/endpoints` snapshots: qwen/qwen3.8-27b, qwen/qwen3.8-flash, deepseek/deepseek-v4.1-flash, xiaomi/mimo-v2.6-pro, z-ai/glm-5.3-flash. Exact snapshots and provider pins are saved under evidence/model_metadata and manifests/openrouter. Each model retains one provider across its methods; no fallback is permitted.

[OpenRouter prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching) distinguishes cache reads/writes. Qwen Flash advertises cache-write pricing. The guard reserves every input byte token bound at prompt PLUS cache-write price, as a conservative upper bound even when billing substitutes cache-write price; no explicit cache creation is requested. Other positive surcharges are rejected. This is a planning reserve, not a measured bill.

Actual GLM endpoint behavior overrode an inference from parameter metadata: reasoning=none produced a pre-generation HTTP 400 stating reasoning is mandatory. The revised GLM configuration requests effort=low and keeps bounded output. The other four models request none; actual returned reasoning tokens are retained. This difference must be disclosed in latency/cost comparisons.

Qwen Flash and DeepSeek initial canaries returned s-prefixed names. A single common UUID clarification from the already-tested local protocol is applied to all five models before SkillRet benchmark selection. It changes only input text/schema, preserves native SkillsMiddleware, and never repairs returned IDs. Initial failures and their charges are retained.

Darkbloom returned upstream shared-pool HTTP 429 for Qwen 27B listing calls while smaller canary/retrieval prompts succeeded. Failed cases remain recorded. No provider switch or automatic retry was made. Known validation/rate-limit rejections are reconciled only when no generation ID exists and provider key cumulative usage agrees with settled ledger charges. A first reconciliation attempt saw stale cumulative usage and failed closed; a later read matched. Ambiguous errors retain reservations.

## User-requested scaling benchmark expansion

The same native selector/settings and provider pins were extended to N=50/250/1000 on three fixed existing pilot queries. Scope was chosen before new observations: 180 planned cases, 53 existing N=50 cases reusable, at most 132 additional calls including missing routers. The full five-model/20-query sweep would require1,200 selectors plus routers, beyond the remaining 153-request allowance. No ceilings were increased and no final-holdout text was consumed.

Canonical nested library construction was verified to preserve N=50 membership/display order when adding larger sizes. N=50 records, including invalid decisions and failures, retain explicit source case/run IDs. Query-only domain routes are reused across N; logical token/cost metrics include routing per workflow while physical charges count each request once. Native canaries remain the integration prerequisite; wrong skill IDs remain benchmark outcomes. Per-query domain gating remains unchanged.

One new Qwen27B/Darkbloom N=250 listing request returned upstream429. It was reconciled as uncharged using the raw pre-generation error and cumulative key usage; remaining new Qwen cases are paused in this batch without provider fallback. The other four models continue through all sizes. Reports distinguish reused evidence, live attempts, exclusions and conditional metrics.

MiMo's late HTTP 429 engine_overloaded variant was also verified as a pre-generation shared-pool rejection. Its preceding successful listing charge took time to appear in cumulative key usage; the guard held all further paid work during that lag. After usage caught up, a US$0.0000000115 difference remained between cumulative usage and summed returned costs. Reconciliation now allows at most US$0.0000001 (0.1 microdollar) of rounding only after explicit no-generation HTTP 400/429 validation, logging the difference and tolerance. Regression controls reject a one-microdollar discrepancy, the observed US$0.057 lag, and a larger unexplained charge. No spending ceiling or request cap changed.
