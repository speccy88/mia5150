# SkillRet: small experiments, mixed results

**Conclusion:** fixed SkillRet+Jev RRF60 is the leading candidate from these small API experiments: the initial posthoc gain was followed by a gain on a separately frozen eight-query panel. The selective gate did not improve NDCG. Preserve the baseline and do not claim a globally best method or blind-test improvement.

## Results that can be compared within each panel

| Experiment and panel | Baseline NDCG@10 | Candidate NDCG@10 | Interpretation |
|---|---:|---:|---|
| Jev Noul, initial12 | 0.716236 | 0.740272 | Small exploratory gain; interval includes zero |
| Jev Score, initial12 | 0.716236 | 0.758343 | Small exploratory gain; interval includes zero |
| Fixed RRF60, same initial12 | 0.716236 | 0.838193 | Posthoc gain; needs confirmation |
| Selective Jev + RRF60, separate8 | 0.850288 | 0.833791 | No NDCG improvement; one win, one loss, six ties |
| Fixed RRF60, second separate8 | 0.750300 | 0.871070 | Frozen-protocol gain; four wins, three ties, one loss |
| Request-clause BM25 fusion, third separate8 | 0.579248 | 0.548357 | Negative; primary Recall@20 fell0.7500 to0.6875 |
| OpenRouter listwise, initial12 | 0.716236 | 0.266057 | Incomplete run: four successes, one error, seven unattempted; not a completed quality comparison |

The12 and8 panels differ; do not compare their absolute scores as model gains. Both are historically exposed public development data. Neither is a genuine blind final test. Historical Jev200, LoRA validation, and routing pilots are separate experiments and are not pooled here.

## The new selective-policy result

The protocol and eight IDs were frozen before new predictions and label inspection. Dense/BM25 top5 overlap of at most2 triggered escalation; at most four calls were allowed in pre-frozen order. All eight queries triggered, leaving four unchanged because of the quota. This gate did not distinguish which cases deserved escalation in this sample.

- NDCG@10 delta: **-0.016496**; descriptive paired bootstrap95 interval **[-0.114965, +0.065476]**.
- Completeness@5: **0.625 → 0.750**. Recall@20 stayed **0.895833**, as the candidate pools were matched.
- q-01302 improved by0.174603; q-03502 regressed by0.306574. The other six tied, including the four no-call cases.
- All four calls succeeded: actual call fraction **50%**, no retries, no error fallback used.
- Median API call latency **0.250s**; total API wall time **0.991s**, or **0.124s/query** across all8. These exclude cached retrieval and lexical-gate computation and are not an end-to-end hardware comparison.
- Input usage **14,893 tokens**, calculated charge **$0.000625506**, with **$0.05** conservatively reserved pending billing reconciliation.

## Five-minute class walkthrough

1. State the hypothesis: small typed relevance models may complement dense retrieval cheaply.
2. Show the initial12 results, then the fixed rank-fusion example. Explain that this improvement was observed posthoc.
3. Show the candidate-pool oracle:0.980447 on the initial12, with one query missing a relevant skill. Reranking cannot recover absent candidates.
4. Show both subsequent frozen panels: the selective policy failed to improve NDCG, while unchanged unconditional RRF60 gained0.120770 on a second separate8. Preserve both outcomes. Candidate-pool oracle on this latest panel was1.0, so this gain concerns ordering.
5. End with the reproducibility and cost controls: immutable raw results, pinned model/dataset/protocol, same candidate pools, failures retained, capped calls, no automatic retries, and no tuning on a blind final set.

Additional offline example: the deterministic request-clause view failed its primary coverage objective despite preserving the original query as a retrieval branch. All8 baseline top100 lists reproduced. Full cached coverage also shows1264 missing query–skill pairs across1120 queries; dense100/BM25top10 recover692 distinct pairs after subtracting158 overlapping recoveries. See `CLAUSE_AND_COVERAGE_REPORT.md` for denominators and diagnostic domain caveats.

## Costs and stop state

Known successful usage across these API experiments totals **$0.006159237**. This excludes any unknown charge for the failed OpenRouter attempt and is not a reconciled invoice. Keep **$1.77/$4 committed**: $1 external reservation, $0.10 initial Jev, $0.50 stopped OpenRouter, $0.05 selective Jev, $0.10 frozen fusion confirmation, and $0.02 order check. No top-up occurred. No remaining budget needs to be spent.

The resumed overnight work is complete, including two final candidate-order reversal calls. Mapped scores changed; one case retained identical ranking, the other had small rank changes. Both kept the same fused top1 and top5 membership. This is operational sensitivity evidence, not a quality gain or proof of invariance. **Paid testing is now explicitly paused in the shared ledger; no further experiment is running or queued.** OpenRouter's failure remains unresolved; it was not retried. RunPod remains unavailable through the tested v2 key endpoint; no pod was launched or permission changed. OAuth/plugin setup is a separate pending task.

## Local evidence and reproduction

Project: `/home/user/code/mia5150-local`.

```bash
cd /home/user/code/mia5150-local
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python round2/score_jev.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python round2/cached_fusion.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python round2/score_selective.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python round2/score_openrouter.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python round2/score_fusion_confirmation.py
cd round2
python3 -m unittest -q
```

These commands evaluate saved outputs or test with dummy responses; they make no paid calls. Raw results and evaluations remain in `jev-typed-12-v1/`, `openrouter-listwise-12-v1/`, `selective-jev-8-v1/`, and `fusion-confirmation-8-v1/`. Frozen manifests and request hashes are alongside them. Credentials remain outside the project and all deliverables. The original local-results ZIP is unchanged. See `FUSION_CONFIRMATION_REPORT.md` for the latest cohort, coverage diagnostic, latency and uncertainty details.

Published Jev price checked2026-10-06: [TypeSafe models](https://docs.typesafe.ai/models), $0.042/input-million, output free. Frozen selective protocol SHA256: `429b77b7c086b7bfc9166fe5be0855d00950d68aad421494f10bce3784abeef6`. Selective raw-result SHA256: `a95463c6f67c7830e58e50cf3c283f7f0f7f2818791c31272e9ac9b5533d4b69`.
