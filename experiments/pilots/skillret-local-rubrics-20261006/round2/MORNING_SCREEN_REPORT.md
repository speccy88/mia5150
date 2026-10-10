# MIA5150 morning research handoff — 6 October 2026

**The original hosted Jev rubric improved BM25 Recall@5 on the six-case screen; none of the four new rubrics improved that primary result.** Local JevK5 and Qwen comparisons completed on two cases with unchanged recall and differing ranking quality. The small exposed panels support diagnostics only, not a new full SkillRet benchmark claim.

## Hosted TypeSafe Jev: completed, then stopped as planned

BM25 top20 -> original Jev improved true macro Recall@5 from **0.527778 to 0.583333**, the candidate-pool ceiling. Role Choice and constraint Score tied original; action Noul and anchored Score each fell to 0.527778. All five reached Recall@10=0.583333. The original top10 arm stayed at Recall@5/10=0.527778 because its smaller candidate pool lacked the extra positive.

The frozen promotion gate required a >=0.05 Recall@5 gain over original, at least two query wins, more wins than losses and all development calls valid. No new rubric qualified. **No confirmation calls were made and confirmation labels remain unused.** Secondary NDCG changes were retained without switching the primary objective or selecting a posthoc winner.

Exact accounting: six queries x four API batches = **24 successful calls**. Each query had original+action on20 candidates (40 questions), anchored+role on20 (40), constraint on20 (20), and original on10 (10): **110 typed questions/query, 660 total**. This is 600 questions for five top20 rubrics plus60 for the separate depth10 control. Zero failed calls, zero retries.

The independent audit recalculated true Recall@5/10, completeness and candidate ceilings for **72 query/rubric/mode rows** (six queries x six rubric/depth arms x direct/fusion). All matched saved metrics. Recall divides by all positive qrels, not just positives found in the pool. See `VARIED_SCREEN_PER_QUERY.csv` and `VARIED_SCREEN_AUDIT.json`.

## Local comparison: separate two-query diagnostic

Local models used the same first two development queries, BM25 top20 pools and frozen original/constraint text. Existing native finite-score adapter and prior model-specific temperatures were retained; no tuning on these cases. Each completed model made80 local pointwise evaluations, all with required option letters present. Local JevK5 is not hosted TypeSafe Jev.

| Model / direct ranking | Recall@5 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| BM25 baseline | 0.750000 | 0.750000 | 0.521912 |
| jevk5 / original_direct | 0.750000 | 0.750000 | 0.521912 |
| jevk5 / constraint_direct | 0.750000 | 0.750000 | 0.622038 |
| qwen35 / original_direct | 0.750000 | 0.750000 | 0.368625 |
| qwen35 / constraint_direct | 0.750000 | 0.750000 | 0.556574 |
| Pool oracle | 0.750000 | 0.750000 | 0.806574 |

Recall could not exceed0.75 within these pools. JevK5 constraint scoring increased NDCG@10 by0.100127 over BM25. Qwen original scoring decreased it by0.153287; constraint scoring increased it by0.034662. All local fixed-RRF60 arms reached NDCG@10=0.556574, except the same value is also the JevK5 original fusion; none improved recall. These are two-query observations, not reliable model rankings.

Plumb was stopped during loading when available Windows RAM crossed below the unchanged2GiB reserve; no prediction and no retry. Qwen initially remained blocked, then passed the explicitly requested fresh assessment after file-specific clean-cache hints: host5.011GiB available, GPU idle. One bounded Qwen attempt completed normally. Earlier reports marking Qwen unattempted remain historical snapshots; this report supersedes that status. No limits were relaxed and no model was downloaded.

| Completed model | Median seconds / 20 candidates | Min host GiB | Min guest GiB | Min free GPU MiB | Max GPU C |
|---|---:|---:|---:|---:|---:|
| jevk5 | 8.303 | 2.073 | 3.854 | 4824 | 64 |
| qwen35 | 7.893 | 2.687 | 3.724 | 5142 | 64 |

No active swap was observed. Local timing covers20 sequential pointwise decisions and excludes cached retrieval; hosted latency covers variable-size batches. No direct speed ratio is claimed. The GPU was verified idle after shutdown; host monitors were stopped. WSL stays6GiB RAM,2processors,2GiB swap.

## Portable class demo and failure semantics

Open `hybrid-demo-v2.html` locally. It preserves the three historical panels and adds five BM25 top20 rubric panels plus the original top10 control: **nine panels,64 displayed cases**. These are repeated views of distinct evidence cohorts, not64 independent new test cases. It exposes top5/top10 results, true recall, completeness, pool ceilings, task text and skill evidence. The introduction states the original-versus-new-rubric null finding.

The failure toggle shows **strict benchmark empty rerank/zero metrics** separately from **served retrieval fallback**. Failed or unattempted cases retain their denominator. Fallback is never counted as a successful Jev prediction; simulations do not edit historical measurements. All256 interaction states passed an offline Node DOM-contract simulation. Browser visual rendering is not verified. The self-contained HTML has no external assets or network calls.

## Cost, validation and remaining scope

67 offline tests pass. Hosted usage was168,117 input tokens / **USD0.007060914** at the pinned rate. The USD0.10 reservation remains committed. Shared commitments are **USD1.87 / USD4**; cumulative known usage isUSD0.013220151, excluding the prior uncertain OpenRouter call. Metered usage is not verified cash billing. No top-up or additional hosted call was made. Other paid work remains paused.

Plumb quality remains unmeasured in this diagnostic because of host memory pressure. No new blind-test or full-benchmark gain has been established. No further confirmation was justified by the frozen hosted gate. Keep these six- and two-query panels separate from the earlier200-query Jev experiment, LoRA validation, eight-query fusion confirmation and unrelated routing pilot.

All original raw results and previous ZIPs are preserved. The new release is a project addendum, with reports, public cached predictions, exact protocols, code, tests and offline demo. No publication, upload or Git push occurred. Secret scanning checks only release contents and never reads the saved keys.

Offline reproduction: `.venv/bin/python round2/audit_varied_screen.py`; `.venv/bin/python round2/build_hybrid_demo_v2.py`; `.venv/bin/python round2/morning_screen_report.py`; `.venv/bin/python -m unittest discover -s round2 -p "test_*.py"`. Run in `/home/user/code/mia5150-local`. Browser-free demo contract check: `node round2/qa_hybrid_demo_v2.js round2/hybrid-demo-v2.html`. Do not rerun one-shot inference workers.
