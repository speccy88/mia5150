# Cached rank fusion: exploratory finding

One fixed equal-weight reciprocal-rank fusion (RRF, constant60, dense-order tie breaking) combined the existing SkillRet and Jev Score top20 lists. No weights, thresholds, or constants were swept. No API or model calls were made.

| Method | NDCG@10, all12 |
|---|---:|
| SkillRet | 0.716236 |
| Jev Score | 0.758343 |
| Equal-weight RRF60 | 0.838193 |
| Label-based candidate-pool oracle | 0.980447 |

Fusion vs SkillRet: +0.121957, six wins/five ties/one loss. Descriptive paired bootstrap95 interval [0.017108,0.248998]. This interval does not correct for posthoc method choice, previous experiments, or small-sample uncertainty. This is **not a held-out/generalization claim**.

Improvements: q-03802 (+0.2184), q-03154 (+0.1296), q-01972 (+0.1145), q-02255 (+0.0782), q-02590 (+0.6845), q-00702 (+0.3691). Only regression: q-04419 (-0.1309), where the relevant skill moved from dense rank2 to fusion rank3. Fusion restored dense's correct top1 on q-04772, q-01223, and q-03923, where Jev alone had moved the relevant skill to rank2.

The top20 pool contains all relevant skills for11/12 queries; mean Recall@20 is0.972222. q-01972 misses one of its three relevant skills, limiting that query's oracle NDCG@10 to0.765361. The aggregate oracle is0.980447. Most observed headroom is therefore ranking-related, though candidate discovery remains a real limitation for that query. Oracle labels are diagnostic only and never used in inference.

## One next probe, prepared only

Selective escalation: on a frozen eight-query panel, call Jev only when cached SkillRet and BM25 top5 lists share at most two IDs, capped at four calls in the frozen cohort order. Escalated results use this fixed RRF60; other queries keep dense ranking. Compare all8 against cached dense with matched candidate coverage; preserve failures and cost accounting. The gate is fixed before new cohort labels/results, not tuned on them.

The panel was selected by deterministic salted SHA256 of query IDs before this analysis loaded labels, excluding all128 prior scorer queries. IDs: q-03502, q-01302, q-00844, q-02230, q-00719, q-00940, q-03891, q-04890. No labels were inspected for these8 during preparation. They are unused in this paid round, **not historically untouched**: all4392 public queries were previously included in dense benchmark evaluation. A genuinely blind final test still needs new independent data.

Artifacts: cached-fusion-evaluation.json; next-confirmation-8.json; selective-probe-plan.json. Reproduction: existing project .venv/bin/python round2/cached_fusion.py, then system python3 round2/prepare_selective_probe.py. No new paid work, OpenRouter retries, installation, or OAuth changes occurred. Existing $1.60 commitments remain intact.
