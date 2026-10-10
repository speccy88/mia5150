# Morning handoff: varied Jev/BM25 screen

The screen is implemented and offline-tested, but **not run against Jev**. Both approval-review refusals happened before process launch. There have been zero new API calls and zero new reservations. The ledger remains USD0.77 reserved plus USD1 external, total **USD1.77 / USD4**. No further paid execution attempt is planned without direct approval in this existing task.

## Validation completed

The previous preflight snapshot correctly records 59 passing tests. Six additional offline tests now bring the total to **65 passing tests**, run with the project's isolated Python environment. The PowerShell launcher was syntax-checked only; neither it nor its API worker was executed.

The frozen mapping remains unchanged:

| Arm | Finite answer | Ranking value |
|---|---|---|
| Original rubric | Score, four ordered levels | Returned score in [0,3] |
| Task-action fit | Noul, yes/no | P(yes), multiplied by 3 for display; identical ordering |
| Anchored relevance | Score, four ordered levels | Returned score in [0,3] |
| Direct/support role | Choice: irrelevant, topical, support, direct | Expected ordinal index 0,1,2,3 from probabilities |
| Explicit constraint fit | Score, four ordered levels | Returned score in [0,3] |

Choice probabilities must contain exactly the four predefined options, each finite and within [0,1], with total within 0.02 of 1. The fixed parser normalizes that total before calculating the expected index. This is an ordinal scoring assumption, not a calibrated probability of benchmark relevance. It intentionally uses the complete distribution rather than only the most probable class. No learned weights or post-result prompt changes are used. Invalid/missing answers, wrong model/type, nonfinite values and unverified token usage stop the run; no retries.

Mock checks confirm answer-map insertion order and Choice probability-map insertion order do not affect results. Reversing candidate order with correctly remapped, distinct mock scores preserves skill-ID ranking. Ties deliberately follow original retrieval order. RRF60 must retain the original BM25 order as its retrieval input. These tests validate mapping logic only; they do not establish live model order invariance or reproduce an API result.

## Cached development ceilings

| Six-query development panel | True Recall@5 | True Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| BM25 ranked top10/top20 | 0.527778 | 0.527778 | 0.457957 |
| Oracle within top10 | 0.527778 | 0.527778 | 0.563085 |
| Oracle within top20 | 0.583333 | 0.583333 | 0.641298 |

True recall divides retrieved positives by **all** positive qrels for each query, then takes the macro average. Completeness@5 is a separate 0.333333 for these arms. The oracle uses development labels only and cannot be deployed. On this panel, top10 reranking cannot improve recall above the existing baseline; top20 provides only 0.055556 absolute recall headroom. This is a small, historically exposed public panel; it does not characterize full SkillRet. Confirmation labels remain unused. The frozen cohort and promotion rule were not changed after inspecting these ceilings.

## Approval and one-command path

The automatic reviewer considered the relayed later authorization insufficient against the original local-only delegation. **Direct user approval in this existing task must resolve that block first.** The command below is a prepared user-run entry point after resolution, not a way to bypass review. Its switch documents the prerequisite and does not itself grant authorization.

Approval must cover the saved WSL Jev key and TypeSafe endpoint `https://api.typesafe.ai/v1/systemone`, sending only public SkillRet query text and candidate names/descriptions. The model is pinned to `jev-1.13.0`. Maximum **30 POST calls, 1.92M input tokens, USD0.08064 metered usage**, with a **USD0.10 reservation** within the unchanged USD4 shared cap. No top-up, other provider or automatic retry. Fred reports USD5 signup credits; no balance verification was initiated, and metered usage is not a verified cash charge.

After direct approval is recorded, run in Windows PowerShell:

```powershell
& 'C:\Users\user\Documents\Codex\2026-10-05\task-2\round2\Start-VariedScreen.ps1' -AfterDirectTaskApproval
```

The launcher starts the existing Windows telemetry monitor, rejects unavailable/stale telemetry, runs the existing WSL resource guard, and stops its monitor in `finally`. WSL memory/CPU limits remain unchanged. The worker takes the shared budget lock, reserves USD0.10 before transmission, rejects duplicate runs, and keeps other paid work paused. The exact existing project/environment is required; the addendum is not a standalone environment or a credential bundle.

The frozen experiment has 24 development calls (five top20 rubric/type arms and separate original top10 control), followed by at most six calls for one qualifying variant on the separate confirmation panel. Selection requires all development calls valid, mean Recall@5 delta >=0.05 against original direct Jev, at least two wins and more wins than losses. Direct reranking, fixed RRF60 fusion and retrieval coverage remain separate metrics.

Protocol SHA256: `d832c146580a51e466d8e887c6a9901979bbefce844c553f18205189822c407e`.

Offline test command: `.venv/bin/python -m unittest discover -s round2 -p "test_*.py"` from `/home/user/code/mia5150-local`.

All earlier evidence ZIPs are preserved unchanged. No keys, account responses or secret-store files are included in the addendum.
