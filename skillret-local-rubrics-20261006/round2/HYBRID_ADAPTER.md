# Reusable SkillRet + Jev adapter

`hybrid_retrieval.py` is a stdlib-only, label-free core. It combines an injected retriever, candidate header dictionary, and Jev adapter. The tested configuration uses exactly20 distinct candidate IDs, version-pinned Jev Score, and equal-weight RRF60 with stable dense-order ties.

- `CachedRetriever` and `CachedJev` replay existing outputs without network or credential access.
- `JevScoreAdapter` accepts an explicit transport and active `SharedBudget`; it never discovers credentials, changes providers, retries, or loads labels.
- `SharedBudget` locks the existing shared ledger, counts external commitments, reserves before transmission, persists attempt IDs, refuses duplicates/overruns/reused run IDs, and never refunds automatically. It does not by itself approve a new experiment or verify provider balance/rates; those remain runner preflight requirements.
- `HybridRetriever.search()` returns separate baseline/reranked/fused/served/strict-benchmark outputs. Invalid or failed reranking serves the dense baseline, but leaves strict benchmark rerank output empty and explicitly marks `baseline_fallback`. This prevents production availability from hiding benchmark failures.

## Working cached demos

```bash
cd /home/user/code/mia5150-local/round2
python3 hybrid_demo_cli.py --panel confirmation --query q-04581
python3 hybrid_demo_cli.py --panel confirmation --query q-04581 --simulate-failure
```

The first command replays the observed improvement. The second demonstrates baseline fallback with an empty strict rerank output. Neither can make paid calls. Open `hybrid-demo.html` for the standalone browser viewer:28 cases across three panels, baseline/Jev/fusion top10, labeled relevance for explanation, task and skill-header details, and a clearly marked simulated failure toggle. Labels are used only by the demo builder and never passed into the reusable pipeline.

The Python replay verified24 successful cached reranks and matched the saved fixed-fusion predictions where present; four selective no-call cases demonstrate absent rerank output. The core has unit tests for deterministic ties, candidate consistency, missing/nonfinite answers, label exclusion, failures, budget limits, pre-transmission persistence, duplicate calls and no implicit refunds.

Browser-render verification was attempted with the installed Windows Edge in isolated hidden/headless profiles, but the host returned no DOM output. Therefore browser runtime/layout QA is **not confirmed**; the CLI replay and Python tests are verified. No browser profile, credential store or environment file belongs in the portable deliverables.

## Distinct next experiment: query representation

Cached diagnostics across all4392 public queries find1264 relevant query-skill pairs missing from dense top20. Dense top100 contains644 of those; adding BM25 top10 contains206. The prior20-query panels were not representative of this broader coverage problem.

Gold-skill domain micro Recall@20 is lowest in ai_agents (0.6973) and developer_tools (0.7356). These diagnostic categories overlap across queries and use positive labels; they must not become inference-time routing features.

Prepare one **request-clause lexical view** experiment: deterministically extract request/requirement clauses from public query text, preserve the original query as a retrieval branch, retrieve BM25 with the new view, and fuse candidate lists with fixed RRF60. No generated text, paid model, or embedding rerun is needed. Freeze a fresh eight-query panel and rule before evaluating labels; compare original BM25 and original SkillRet to the new representation with candidate Recall@20/oracle ceilings and actual ranking metrics kept separate. This tests query representation, not another Jev reranking panel or a parameter sweep. A resource-guarded local lexical run is future work; no inference or charges were incurred by this preparation.

All domain analysis is diagnostic on the already-exposed public dataset. A genuinely untouched final evaluation still requires independent data. At overnight completion, shared commitments are **$1.77/$4**, and the reusable budget component refuses new reservations while the ledger's paid-tests pause flag remains set. The proposed clause experiment subsequently ran and failed its coverage objective; see `CLAUSE_AND_COVERAGE_REPORT.md` rather than treating it as untested current advice.
