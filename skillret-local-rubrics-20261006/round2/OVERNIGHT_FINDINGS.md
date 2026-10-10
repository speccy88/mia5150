# MIA5150 overnight findings

**Most useful lead:** fixed SkillRet + Jev Score RRF60. A separately frozen8-query panel improved NDCG@10 from0.750300 to0.871070 (4wins/3ties/1loss). This follows a posthoc gain on12 earlier queries, but remains small-sample evidence from historically exposed public data. No globally-best or blind-test claim is supported.

**Useful negative and mixed findings:**

| Test | Result | Decision |
|---|---|---|
| Selective Jev gate, separate8 | NDCG0.850288→0.833791; gate fired on all8 | Do not adopt this gate |
| Request-clause lexical view, separate8 | Recall20 0.7500→0.6875; baseline reproduced8/8 | Do not adopt this rule |
| Fixed30-budget candidate mix, all4392 | Macro Recall30 0.873330→0.874962;183 pairs gained/130 lost; oracle nearly flat | Complementary, but not a clear universal replacement |
| OpenRouter listwise | Four successes, one HTTP error, seven unattempted | Preserve incomplete result; no blind retry |

**Coverage explanation:** dense top20 misses1264 positive query–skill pairs across1120 queries, out of7187 pairs/4392 queries. Dense top100 recovers644 and BM25 top10 recovers206, with158 shared:692 distinct pairs recovered, not850. Domains are diagnostic gold-skill metadata, not deployable query labels.

**Working implementation:** reusable label-free retrieval/Jev adapter; explicit locked budget reservations and attempt recording; deterministic ties; no retries; separate dense production fallback and empty strict benchmark output on failure. Cached demo has28 cases and24 successful score replays. CLI success/failure demonstrations verified. Browser viewer delivered, but headless runtime/layout QA remains unverified; no browser/CUA tool was available to replace the failed attempt.

**Validation and resources:**52 offline tests pass. CPU-light cached experiments ran under existing RAM/paging guards and one-thread settings; no resource settings were relaxed. All raw prior negative/incomplete evidence remains. No live RunPod resources or unresolved owned monitor processes remain. No GPU work occurred. The final authorized two-case request-order check succeeded; one case kept identical rankings, the other changed3 Jev pair orderings and2 fused pair orderings. Both retained fused top1 and top5 membership. This is operational evidence, not a quality gain or proof of invariance.

**Budget:**$1.77/$4 conservatively committed, including$1 external,$0.10 initial Jev,$0.50 stopped OpenRouter,$0.05 selective Jev,$0.10 fixed-fusion confirmation,$0.02 order check. Known successful API usage$0.006159237 is not a reconciled invoice; unknown failed-call charges/reservations remain. No top-up occurred. The shared ledger is explicitly paused for new paid tests pending next-day review.

**Next-day recommendation:** review the negative/mixed findings, use the cached class demo, and prioritize an independently authored/labeled confirmation set before stronger accuracy claims. Preserve canonical candidate order and dense fallback in deployment experiments. Diagnose the unknown OpenRouter HTTP failure from provider-side records before any retry; resolve RunPod OAuth separately if still desired. No further computation or paid test is queued merely to consume the night or budget.

Evidence: `ROUND2_CLASS_DEMO.md`, `FUSION_CONFIRMATION_REPORT.md`, `CLAUSE_AND_COVERAGE_REPORT.md`, `CANDIDATE_BUDGET_REPORT.md`, `ORDER_SENSITIVITY_REPORT.md`, `HYBRID_ADAPTER.md`, `FINAL_BUDGET_STATUS.json`, and adjacent JSON manifests/results. Portable cached demo: `MIA5150_Hybrid_Cached_Demo.zip` (original13-file snapshot; newest reports are separate).
