# Fixed fusion gains on a separately frozen public panel

**Finding:** fixed equal-weight RRF60 combining SkillRet and Jev Score improved NDCG@10 on a new eight-query panel, from **0.750300 to0.871070**. This supports keeping fixed fusion as the leading candidate from these small API experiments. It does not establish a globally best method or a blind-test improvement.

| Metric | SkillRet | Fixed RRF60 |
|---|---:|---:|
| NDCG@10 | 0.750300 | 0.871070 |
| Completeness@5 | 0.875 | 0.875 |
| Recall@20 | 1.000 | 1.000 |

Delta NDCG@10: **+0.120770**; descriptive paired bootstrap95 interval **[+0.010467,+0.240920]**. Four wins, three ties, one loss. Do not treat this tiny-sample interval as proof of broad generalization.

Improvements: q-04581 (+0.369070), q-00136 (+0.369070), q-02424 (+0.082008), q-03156 (+0.226294). Regression: q-03583 (-0.080279). q-00091, q-04959, and q-03450 tied.

All eight candidates pools contained every labeled relevant skill: oracle NDCG@10=1.0. This panel tests ordering, not recovery of missing candidates. Candidate sets were identical for baseline and fusion.

## Why this experiment

Before choosing this probe, an offline diagnostic added the cached BM25 top10 to dense top20 on the previously reviewed12+8 queries. It recovered **zero additional relevant candidates** on both panels; oracle NDCG was unchanged. That offered no evidence for paying to rerank larger unions yet. We therefore tested the already-fixed RRF60 method on new queries rather than tuning its weights or repeating an earlier panel.

The protocol, primary metric, parameters, query IDs and candidate pools were frozen before inference and label inspection. Eight IDs were selected by salted SHA256, excluding the128 historical scorer queries and the previous selective eight-query panel. No gate, quota, parameter sweep, retries or provider fallback was used. All8 predictions were saved before scoring.

**Exposure limitation:** these are new to the current reranking round, but belong to the historically evaluated public4392-query benchmark. They are not a genuinely untouched final test. Previous posthoc12-query results remain labeled posthoc, and the negative selective-policy result is preserved separately.

## Cost, latency and reproduction

- Eight successful calls, no failures,29,210 input tokens.
- Calculated usage **$0.001226820** at the published $0.042/input-million, output free. [TypeSafe model documentation](https://docs.typesafe.ai/models), checked2026-10-06.
- Median API wall time **0.250889s**; total **2.000601s**. Cached retrieval/fusion computation is excluded, so this is not end-to-end pipeline latency.
- Added reservation **$0.10**, retained pending reconciliation. Total round commitments **$1.75/$4**, including the user's $1 external reservation.
- Known successful usage across round2 is **$0.005850915**; the unknown failed OpenRouter charge and final billing remain unreconciled. No top-up, RunPod use, other provider call, GPU job or host-settings change occurred.

```bash
cd /home/user/code/mia5150-local
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python round2/coverage_diagnostic.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python round2/score_fusion_confirmation.py
cd round2
python3 -m unittest -q
```

These commands only evaluate saved data or test dummy responses. Live generation was one-shot and is not part of reproduction. Raw outputs: `round2/fusion-confirmation-8-v1/results.json`. Evaluation: `evaluation.json` in the same directory. Protocol SHA256: `e8133175358110f7e96a8cbd52dc6c83378f844d6e83c5f1248641b206e1c261`. Result SHA256: `e3202d5f5525ed5122da6620031a063466fd3320962a1f63016514da4569d79b`.
