# Primary-source verification: MIA5150 proposal
Checked 2026-10-24 UTC. Read-only audit; no inference, API spending, repository edits, or new labels.

## Bottom line

The named DeepSeek models and BGE-M3 price are not invented. The material correction is to make pricing/provider/model identity and the experiment's actual call/token volume explicit. The original proposal's input-only multiplication is mathematically correct but is not a defensible complete study budget.

| Claim | Verdict | Evidence and corrective wording |
|---|---|---|
| DeepSeek V4 Flash 0731 exists | Supported | OpenRouter lists exact slug `deepseek/deepseek-v4-flash-0731`, release 2026-07-31 and 1M context. Preserve the exact slug, endpoint/provider, execution date and returned model identity. [Model page](https://openrouter.ai/deepseek/deepseek-v4-flash-0731) |
| DeepSeek V4.1 Flash exists | Supported | OpenRouter lists `deepseek/deepseek-v4.1-flash`, release 2026-09-10 and 1M context. It is a different model, not an interchangeable spelling of V4 Flash 0731. [Model page](https://openrouter.ai/deepseek/deepseek-v4.1-flash) |
| The proposal's $0.0215/M input rate | Plausible displayed rate, not a frozen provider quote | OpenRouter Discover displays $0.0215 input/$0.30 output for V4 Flash 0731; its model page presently displays $0.0422/$0.1763. This inconsistency means neither overview alone establishes the rate of a pinned execution. Archive the endpoint-specific quote and actual returned usage/cost before each paid study. [Discover](https://openrouter.ai/discover), [model page](https://openrouter.ai/deepseek/deepseek-v4-flash-0731) |
| Old direct DeepSeek model names preserve model identity | False for current direct API | Official docs say `deepseek-flash` currently serves DeepSeek-V4.1-Flash-1017; legacy `deepseek-v4-flash` is accepted but routed to that newer model. Current direct uncached input is $0.15 off-peak/$0.30 peak per million; cached input and output have separate rates. A legacy string alone is not a model pin. [Official pricing](https://api-docs.deepseek.com/quick_start/pricing/), [changelog](https://api-docs.deepseek.com/updates/) |
| BGE-M3 costs $0.01/M tokens | Supported as an OpenRouter hosted offer, not inherent model cost | OpenRouter lists this rate for Parasail and DeepInfra and an approximately 8K input limit. Name the hosted route if using it; local embeddings incur no per-token API bill but have hardware/time costs. Do not call API availability an execution guarantee. [OpenRouter BGE-M3](https://openrouter.ai/baai/bge-m3) |
| SkillRet contains 6,006 skills, 4,392 queries, 7,187 qrels | Supported for v1.1 test/evaluation split, not entire dataset | Current official dataset card distinguishes these test counts from training and out-of-split records, labels test queries synthetic, and says v1.0 and v1.1 scores are not directly comparable. Use “the pinned v1.1 test split” instead of implying this is the whole dataset. [Dataset card](https://huggingface.co/datasets/ThakiCloud/SKILLRET) |
| All SkillRet content is simply Apache-2.0 | Needs qualification | Official card assigns Apache-2.0 to benchmark metadata, queries and taxonomy while underlying skills retain source MIT/Apache-2.0 licenses. Preserve notices/provenance; a benchmark metadata license is not blanket relicensing of source skills. [Dataset card](https://huggingface.co/datasets/ThakiCloud/SKILLRET) |
| Dataset is approximately 335 MB total | Snapshot/scope dependent, not confirmed as current total | Current official card reports 714 MB total. This does not disprove a 335 MB historical test-only/local download, but “total” needs revision/split/files/compression qualification. [Dataset card](https://huggingface.co/datasets/ThakiCloud/SKILLRET) |
| Exact revision `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12` verified | Not independently verified by this web pass | The revision-specific README and commit URLs could not be opened through the web tool. Current main card corroborates counts, not byte identity of that revision. The implementing agent should check fetched revision and file SHA-256 manifests; do not present this audit as verifying immutable bytes. |
| A skill description is limited to 1,024 characters | Supported | Agent Skills specification limits the YAML `description` field to 1–1,024 characters. This is not a token limit and does not cap full `SKILL.md` or name-plus-description combined. The revised proposal's combined 1,024-character/128-BGE-token preprocessing is an experimental choice that must be documented separately. [Specification](https://agentskills.io/specification) |

## Budget correction and acceptance criteria

1. Preserve the original arithmetic: 2,000 × 300,000 = 600,000,000 input tokens; at an assumed $0.0215/M that is $12.90 input-only. Neither the 300,000-token workload nor the complete-study price has been established by that multiplication.
2. State whether “2,000 queries” means 2,000 unique requests per cell or 2,000 total logical cases. At 2,000 per method-size cell, five methods × four sizes means 40,000 selector cases per model before routing, canaries, repeats or retries. Listing and shortlist methods have very different token volumes; budget by planned case, not one universal prompt size.
3. Count input, output/reasoning, routing, retries, canaries, activation checks and embedding/index preparation separately. Distinguish logical route allocation from physical cached calls. Record provider-returned cost separately from reservations and estimates.
4. Save model/provider/rate/context snapshots; verify exact endpoint availability with a bounded canary. Disable unrecorded provider fallback when it would invalidate paired model comparisons.
5. Keep USD primary. Label CAD as an estimate using an explicit dated FX rate/planning factor; the proposal implies approximately 1.426 CAD/USD while the revised proposal uses 1.50, so they are not the same convention.
6. Tokenize the assembled prompt with the actual tokenizer, including schemas/system content and output reserve. Character limits and nominal context capacity do not establish that every full-listing prompt fits.
7. For local 8 GB work, use measured peak RAM/VRAM, batch size, truncation and elapsed time. “No hosted API fee” is not “no compute cost,” and the proposed ten GPU hours is an estimate, not evidence.

## Suggested conservative prose

“Hosted cost is a planning estimate for the frozen case matrix, not a provider quote or measured bill. We will pin endpoint/provider and record actual token usage and returned costs. Public model names can be aliases; exact returned versions and execution dates will be retained. The SkillRet v1.1 test split contains 6,006 skills, 4,392 synthetic queries and 7,187 positive qrels. Source-skill licenses and dataset revision/file hashes will be preserved.”

## Limits of this verification

Current public primary pages were checked, not historical invoices, API credentials, private provider contracts, archived experiment request bytes or source-skill license files individually. No hosted capability or price was tested through paid inference. Revision-specific web retrieval failed; that is an access/verification limitation, not proof that the revision is invalid.

