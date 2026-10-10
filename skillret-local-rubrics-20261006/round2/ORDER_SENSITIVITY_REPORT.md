# Two-case operational order check

Selection and permutation were frozen: the first two cases in the already-frozen confirmation cohort, q-00091 and q-03583, each with exactly one reversal of all20 candidate questions. Model, task and rubric were unchanged. Returned scores were mapped back to stable skill IDs; ranking ties still used original dense order to avoid introducing mechanical tie changes.

| Case | Scores changed /20 | Maximum score change | Jev inverted pairs /190 | RRF inverted pairs /190 |
|---|---:|---:|---:|---:|
| q-00091 | 17 | 0.10 | 0 | 0 |
| q-03583 | 10 | 0.07 | 3 | 2 |

Both cases kept the same fused top1 and the same top5 membership (5/5 overlap); the second case did not keep the full ordering. Score-change epsilon was1e-9. No relevance labels were read and no quality gain was calculated.

**Interpretation:** the two requests show some score/rank sensitivity but stable leading membership in these cases. This is not proof of invariance. Backend or temporal nondeterminism may explain changes as well as permutation; without repeating the original order, causation cannot be isolated. No repeat or permutation search was performed.

Two calls succeeded,7341 input tokens, known usage **$0.000308322**. Conservative additional reservation **$0.02** remains. Total commitments **$1.77/$4**. Published input-rate bound for this test was128000×$0.042/million=$0.005376; output is free. No top-up occurred.

**New paid testing is stopped.** The shared ledger now marks paid tests paused, and the reusable budget component refuses new reservations while that flag is set. Existing reservations remain until billing reconciliation. All old one-shot runs and negative/incomplete outputs are preserved.

Protocol SHA256: `2a38f670f91c94ab1fbc29d392b4b400561a87ccd5bf4d55934eacb2ddb30e34`. Raw result SHA256: `53f6155b28df6ad989b7054c0cf25e8b22992774db460962c5d4be4f60627d6d`. Evidence directory: `round2/order-sensitivity-2-v1/`; comparison is reproducible with `python3 round2/compare_order_sensitivity.py`, which makes no network calls.
