# Trial acceptance checklist

Each item needs an observed artifact or an explicit BLOCKED/NOT RUN status. This list is for Codex to complete on the Mac; unchecked items are not claims of implementation.

## Data and baseline
- [x] Dataset revision is an immutable resolved commit; all loads use it.
- [x] Counts, schema, licence/provenance and actual examples are saved.
- [x] Every gold ID maps to a skill; multi-positive queries are handled.
- [x] Pilot-consumed queries are separated from the future untouched holdout.
- [x] A real BM25 metric is printed with corpus/query counts and input representation.
- [x] Published reproduction, partial-protocol check and local baseline are distinguished.
- [x] Dense embeddings are from the stated model, not a placeholder; cache IDs/hashes match.

## Native Deep Agents integration
- [x] Native SkillsMiddleware injected the expected candidate metadata.
- [ ] Source directory semantics are tested, including rejection of the old per-skill-path pattern.
- [x] Duplicate readable skill names cannot silently override one another.
- [x] No non-candidates, body-only content or gold labels appear in the main initial prompt.
- [x] Main offered/executable tool set is only terminal select_skills.
- [x] Two fresh runs cannot see each other's skill state/messages.
- [x] A real OpenRouter canary returned a usable response with request ID and usage.
- [x] Separate activation smoke includes a logged real SKILL.md read, not just a declared selection.
- [x] Attempts to execute scripts, read host files, write files or call subagents are blocked.

## Cost, context and reproducibility
- [x] Both currency guards and completion-count cap operate before paid calls.
- [x] Missing key/permission forbids paid requests while preserving offline progress.
- [x] Unknown/ambiguous billing does not become zero on restart.
- [ ] Unknown provider/price/context causes a useful preflight stop.
- [x] No automatic model fallback, retries, context truncation or summary can alter the comparison unnoticed.
- [x] The same case resumes without another charge; changed config cannot reuse incompatible outputs.
- [ ] Actual billed router reuse and logical per-workflow cost are distinguished.
- [x] Seeded catalogues are nested, gold-preserving and identical across strategy comparisons.

## Evaluation and report
- [x] Scores agree with the supplied known-answer controls and small hand checks.
- [x] Unknown or duplicate predictions and malformed tool calls are retained as failures.
- [x] Candidate retrieval recall and final selection quality are separate.
- [x] Framework/API failures and semantic mistakes have separate diagnostics.
- [x] Planned, attempted, valid and completed counts are visible; no success-only denominator.
- [x] Report and figures are generated from persisted rows; no fabricated values fill gaps.
- [x] Tests ran via python -m pytest with a collected-test count.
- [x] Notebooks ran or visibly record skipped/blocked live cells without accidental new paid calls.
- [x] One final ZIP preserves all related prompts/code/tests/evidence and excludes actual .env/secrets/caches.

## Minimum useful outcome

A real data audit + one real BM25 score is a useful first result. Calling the **agent pilot working** additionally requires the real OpenRouter/Deep Agents selection and activation checks. The full 240-selector sweep is not required when a smaller completed trial establishes feasibility within the approved limits.

## Observed status after local follow-up

The original checklist is preserved in `evidence/acceptance-checklist-original.md`. Checked guard/adapter controls are unit or transport-fixture evidence where a paid call was unavailable; they are not OpenRouter live certification. Correct native source-root loading is observed; a separate negative test of the old per-skill source-path form was not run. OpenRouter canary/provider/billing checks remain BLOCKED by missing credentials/permission.

User-authorized Ollama tests used both pinned local models and real Deep Agents. All 40 planned selection cases are recorded: 33 attempted, 25 valid, eight invalid IDs, seven gated skips. All ten activation cases are recorded: six fixture workflows pass; four real workflows fail guards. Actual native reads include three real SkillRet candidate reads before later 9B guard failures. No skill instructions or host-file reads executed.

The 38-test suite passes. Both notebooks execute with paid/local inference cells SKIPPED. Exact replay adds zero model attempts. The final package command performs the allowlist, secret scan, ZIP CRC and member-hash verification and writes `evidence/package-verification.json`.

## Observed five-model OpenRouter follow-up

All five revised native canaries pass. The original missing-credential statements above are historical. Four models completed the five-query/N=50 selection trial; Qwen3.8 27B is partial after two upstream listing rate limits. Saved outcomes: 100 planned, 85 recorded, 79 attempted, 68 valid, 32 exact, six gated skips and 15 unrun selector cases. Do not treat these pooled counts as a matched model ranking.

All 25 separate activation cases are recorded: 15 valid workflows and 16 actual native reads, including four reads from real SkillRet candidates. Ten workflows fail parallel-read guards. No skill execution or host-file access occurred. Live N=250/1000 scaling and cross-N billed router reuse remain unrun.

47 tests pass. Audit covers real wire tools/provider pins, canonical queries, candidate-only paths, exact catalogue equality with Ollama, metric arithmetic, native read outputs and cost reconciliation. Full paid ledger: US$0.0485757201, estimated CAD$0.07286358015 at the planning factor 1.50; 147 attempts, 144 returned generations, three reconciled uncharged pre-generation rejections, no unresolved reserves. Exact resume and notebook/package validation are recorded in evidence.

## Observed scaling expansion

The N=50/250/1000 expansion now records all 180 planned cases across five models and three fixed queries:53 reused N=50 records,138 selector attempts,121 valid outputs and24 exact sets. Four transport rejections appear in this study (two historical N=50 and two new);32 new cases were paused after provider failures,10 domain cases were gated,13 outputs had unknown IDs. No results were retried for correctness. Qwen27B is substantially incomplete; MiMo's final two cells are paused after overload. These exclusions prevent a matched model ranking.

The saved-wire/catalogue/metric/cost audit passed and exact replay made zero calls. The suite has 54 passing tests. Separate native activation evidence remains unchanged; this expansion does not measure execution success. Shared totals:236 attempts, US$0.3864964105, estimated CAD$0.57974461575, zero unresolved reserve. Full 20-query/five-model scaling is still unrun under the existing 300-request cap.
