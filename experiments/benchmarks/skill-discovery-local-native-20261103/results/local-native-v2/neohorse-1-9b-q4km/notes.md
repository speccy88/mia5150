# Run notes

Protocol: v1-native-final-answer-parser-v2
Model: neohorse-1-9b-q4km-native-v2; backend: llama; context: 32768.
Quantization: Q4_K_M.
Hardware: NVIDIA RTX 2060 SUPER, 8192 MiB; observed driver 610.88 (preparation, 2026-11-03).
Actual local startup proof is bound to the server process, model file hash and log prefix.

140 trials = seven sizes x supplied 20 queries. Candidate source order retained.
Fresh random.Random(42).sample per size; identical base library for all queries at that size.
Absent gold ID replaces the last skill. Descriptions are sliced to 300 Python characters.
This preserves v1 placement bias. Confirm RNG reset scope against the original hosted runner
before claiming byte-identical reproduction; that runner was not provided.

The 6006 IDs have 5801 distinct names. Nothing is deduplicated. Scores resolve names,
not IDs: duplicate-name identity cannot be recovered from the required output format.
Candidate IDs, prompt hashes and per-response ambiguous-name audits remain local.
Some queries contain name hints, and 20 queries provide a small, partly lexical sample.

The v1 context estimate (44.5*N+200 > 90% window) creates explicit blocked rows;
actual tokenized overflow is separate. Estimated counts are NEVER token usage.
Local /apply-template + /tokenize uses the loaded model's real tokenizer, then passes
the resulting token IDs to native /completion (temperature=0, n_predict=500, seed=42).
GPU-only means all transformer/output layers and KV on CUDA; CPU tokenization/control
is expected. Partial offload is rejected. No context shifting or automatic memory fit.
500 output tokens are reserved before local inference. No truncation is accepted.
No thinking-mode override is applied; generation can exhaust its 500-token allowance.
Native parser v2 retains a prefilled <think> marker as parsing state only. It is
not added to the model prompt a second time. Delimited reasoning is excluded from
rank scoring; unfinished reasoning has REASONING_NO_FINAL_ANSWER with real usage.
Invalid and repeated slots in the final answer retain their positions. Raw model
responses are kept in the private response directory, excluded from Git export.

Compatible API: system/user chat messages, temperature=0, max_tokens=500. Unsupported
parameters are errors, with no silent retry or replacement model.
OAuth variant: same system text in instructions, user input array, streaming Responses,
store=false. temperature and max_output_tokens MUST be omitted on this route.
It is a separate variant, without a guaranteed 500-token output cap or fixed temperature.
OAuth context comes from the authenticated account catalog, not the larger public API
maximum. Actual input/output token counts come from response.completed.usage; the
44.5 tokens/skill gate remains an estimate, not a claimed tokenizer measurement.
Reasoning effort is not sent; the actual server value is kept in each SQLite audit.
If an error stops a model, CSV placeholders use NOT_RUN_AFTER_MODEL_STOP. These rows
were never attempted and must not be interpreted as observed zero-accuracy results.

Ranking preserves invalid/duplicate slots; bad rank 1 is never promoted away.
Markdown/quotes/description suffixes are normalized. Case-only matching is a recorded
fallback (fuzzy_match); strict exact/retrieval metrics require exact name spelling.
P@K denominator is always K; one gold: recall hit, NDCG=1/log2(rank+1), MRR=AP=1/rank.
candidate_completeness=1 because the gold ID is assured in every candidate library;
it is not response-list length. rank_of_gold=0 means absent, never rank 0 relevance.

Token counts come from server usage/tokenizer only; unavailable fields remain blank.
Unknown hosted monetary cost remains blank. Local service cost=0 excludes electricity.
Latency: retrieval=0 (listing method); llm measures inference HTTP roundtrip; e2e includes
preflight/template/tokenization/parsing, excluding run setup and SQLite/CSV persistence.
Estimated context skips have zero latency/metrics. Other errors retain elapsed time.
No automatic inference retries. In-flight interrupted calls require explicit resolution,
so a crash cannot silently repeat a possibly charged request. SQLite is authoritative;
CSV is rebuilt atomically after each transaction. Use unique output directories.

No inference ran during script preparation. Tests use synthetic data/mock HTTP only.
No full prompts, API keys or OAuth tokens are printed to console or stored in run outputs.
CSV contains supplied queries/names and invalid output lines: review before sharing.
OAuth credentials use Windows DPAPI CurrentUser in this package's private/. No Hermes
or Codex credential files are read, copied, or modified. Do not publish private/ or data/.
