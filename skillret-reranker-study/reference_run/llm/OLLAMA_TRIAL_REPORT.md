# Local Ollama follow-up — real Deep Agents inference

The user authorized local Ollama testing instead of OpenRouter. These results are a separate provider/model protocol; they do not complete the original OpenRouter acceptance milestone.

**Assessment: conditional go for a local feasibility study.** Real native metadata discovery, local inference and strict tool boundaries are exercised. Saved format and activation failures remain limitations for unattended expansion; this is not evidence of successful skill execution or general method superiority.

Both installed model digests, server version, quantization and settings are pinned in `manifests/ollama/`. The native Deep Agents graph, metadata and tool isolation are retained. The local protocol adds one fixed UUID-format clarification to the system instruction and tool schema, uniformly across both models and all methods. Initial canaries returned the native s- prefix and were rejected; their failures remain saved. No model output is repaired. There are five fixed SkillRet pilot queries, N=50, top-five shortlists, gold-preserving random libraries and the same display order. No new final-holdout queries are consumed. No model downloads or paid API calls occur.

| Local model | Method | Valid/planned | Exact matches/valid | Candidate recall (queries) | Mean attempted workflow seconds |
|---|---|---:|---:|---:|---:|
| qwen3.5:4b-mlx | listing | 4/5 | 1/4 | 1.000 (5) | 36.615 |
| qwen3.5:4b-mlx | bm25 | 4/5 | 2/4 | 0.900 (5) | 9.517 |
| qwen3.5:4b-mlx | vector | 4/5 | 2/4 | 0.800 (5) | 8.466 |
| qwen3.5:4b-mlx | domain_vector | 3/5 | 1/3 | 0.667 (3) | 11.505 |
| qwen3.5:9b-mlx | listing | 0/5 | — (0 valid) | 1.000 (5) | 76.811 |
| qwen3.5:9b-mlx | bm25 | 5/5 | 1/5 | 0.900 (5) | 20.396 |
| qwen3.5:9b-mlx | vector | 5/5 | 2/5 | 0.800 (5) | 17.311 |
| qwen3.5:9b-mlx | domain_vector | 0/5 | — (0 valid) | — (0) | — |

Conditional accuracy excludes invalid completions; valid/planned exposes missing/failed cases. Detailed recall/F1, token counts and failures are in OLLAMA_RESULTS.csv and append-only `results/ollama/`. Logical domain-vector usage includes its query-only router. Cold model loading and prompt processing are included in wall time; these are not pure generation-speed measurements. Five queries are too few for a general model ranking.

Latency and token means include invalid completed attempts as well as valid ones; gated cases without inference are excluded. Physical totals include all canaries, comparison/routing calls and activation attempts, including failures.

The domain arm is attempted for a query only when listing, BM25 and vector selectors returned valid IDs. This gate and format failures yield different query subsets, so the conditional averages are diagnostics, not paired method comparisons. No failed output is repaired or sampled again for accuracy.

Post-hoc router retention against labels:

- qwen3.5:4b-mlx: 3 routes; mean gold retention 0.667.
- qwen3.5:9b-mlx: no eligible router calls.

Labels are evaluated after routing and never enter router state. Single/multiple-positive denominators are in `OLLAMA_SINGLE_MULTI.csv`.

Separate activation results:

| Model | Recorded/planned | Valid fixtures | Valid SkillRet cases | Native reads |
|---|---:|---:|---:|---:|
| qwen3.5:4b-mlx | 5/5 | 3/3 | 0/2 | 3 |
| qwen3.5:9b-mlx | 5/5 | 3/3 | 0/2 | 8 |

- qwen3.5:4b-mlx, q-04792: Blocked: parallel_reads_forbidden.
- qwen3.5:4b-mlx, q-04981: Blocked: parallel_reads_forbidden.
- qwen3.5:9b-mlx, q-04792: Blocked: read_window_limit.
- qwen3.5:9b-mlx, q-04981: Blocked: read_outside_candidates.

Actual native read_file calls demonstrate disclosure, not execution of skill instructions. Each model plans three hand-authored fixtures and two real SkillRet queries. One fixture canary precedes each model's comparison.

Native reads count only actual ToolMessage results, deduplicated by tool-call ID. A graph that fails after a read can return no structured read list; its already-saved model-message history retains the completed native tool results. `evidence/ollama/derived_native_reads.json` recovers those events without changing raw rows. Requested reads rejected by a guard are never counted as reads. Observed real SkillRet native reads: **3**; valid real activation workflows: **0/4 planned**. A subsequent denied pagination or out-of-candidate request still makes that activation workflow a failure.

Recorded comparison failures:

- qwen3.5:4b-mlx: 3 unknown_ids cases.
- qwen3.5:4b-mlx: 2 selector_prerequisite_failed cases.
- qwen3.5:9b-mlx: 5 unknown_ids cases.
- qwen3.5:9b-mlx: 5 selector_prerequisite_failed cases.

Canary history, failure query IDs and exact run IDs are in `ollama_summary.json`; failed canaries block comparison for that configuration.

Total local generation attempts including preserved failures: 62; actual HTTP requests: 61; completed responses with usage: 61. The initial adapter argument error occurred before HTTP. Local API charge: USD 0 / CAD 0; electricity and hardware cost were not measured. Original OpenRouter cost ledger remains separate.

Settings: temperature=0, seed=42, think=false, 32768 context, max output 512 (router 96), sequential calls, 90-second timeout, no automatic retries, persistent 100-request cap. The request uses a conservative full-message/tool UTF-8 byte bound before inference; no context is silently truncated. Returned usage and thinking behavior are checked. Local inference requires no OpenRouter key or paid opt-in; the original paid route retains all guards.

Request UUIDs are client-assigned trace identifiers; the local response does not supply a provider completion ID. Model identity is recorded by installed digest and returned model tag.

Input/output token counts come from returned usage. Ollama did not expose separate cached-token or reasoning-token counters here; zeros in the shared per-case accumulator are placeholders for unavailable detail, not measured cache/reasoning totals. The summary marks those counters null. Thinking was explicitly disabled in the wire request and no thinking content was returned.

Run `python pilot.py ollama-smoke`, `python pilot.py ollama-run`, then `python pilot.py ollama-report` in the project venv. Add `--model qwen3.5:4b-mlx` or `--model qwen3.5:9b-mlx` to restrict a run. Exact matching cases are reused; failure traces remain.

API references: [Ollama chat](https://docs.ollama.com/api/chat), [tool calling](https://docs.ollama.com/capabilities/tool-calling), [ChatOllama](https://docs.langchain.com/oss/python/integrations/chat/ollama). Installed interfaces and raw request/response hooks were used to verify actual behavior.
