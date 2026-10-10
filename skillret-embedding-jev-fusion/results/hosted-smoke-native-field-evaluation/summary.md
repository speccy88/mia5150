# Smoke hosted snapshot

10/10 queries complete; all 10 selected queries remain in metric denominators.

| Metric | A matched dense | B Jev | C RRF60 |
|---|---:|---:|---:|
| NDCG@5 | 0.917330 | 0.841622 | 0.946928 |
| Recall@5 | 0.933333 | 0.933333 | 0.933333 |
| Completeness@5 | 0.900000 | 0.900000 | 0.900000 |
| MAP@5 | 0.891667 | 0.794444 | 0.933333 |
| Precision@5 | 0.240000 | 0.240000 | 0.240000 |
| NDCG@10 | 0.949688 | 0.855749 | 0.963644 |
| Recall@10 | 1.000000 | 0.966667 | 0.966667 |
| Completeness@10 | 1.000000 | 0.900000 | 0.900000 |
| MAP@10 | 0.917063 | 0.801852 | 0.944444 |
| Precision@10 | 0.140000 | 0.130000 | 0.130000 |
| NDCG@15 | 0.949688 | 0.855749 | 0.976326 |
| Recall@15 | 1.000000 | 0.966667 | 1.000000 |
| Completeness@15 | 1.000000 | 0.900000 | 1.000000 |
| MAP@15 | 0.917063 | 0.801852 | 0.952778 |
| Precision@15 | 0.093333 | 0.086667 | 0.093333 |
| MRR@5 | 0.950000 | 0.833333 | 1.000000 |
| MRR@10 | 0.950000 | 0.833333 | 1.000000 |

| Comparison / metric | Delta | 95% paired bootstrap CI |
|---|---:|---|
| B_minus_A / NDCG@10 | -0.093940 | [-0.196381, +0.000000] |
| B_minus_A / Recall@5 | +0.000000 | [+0.000000, +0.000000] |
| B_minus_A / Recall@10 | -0.033333 | [-0.100000, +0.000000] |
| B_minus_A / Completeness@5 | +0.000000 | [+0.000000, +0.000000] |
| B_minus_A / MRR@5 | -0.116667 | [-0.250000, +0.000000] |
| C_minus_A / NDCG@10 | +0.013956 | [+0.000000, +0.038512] |
| C_minus_A / Recall@5 | +0.000000 | [+0.000000, +0.000000] |
| C_minus_A / Recall@10 | -0.033333 | [-0.100000, +0.000000] |
| C_minus_A / Completeness@5 | +0.000000 | [+0.000000, +0.000000] |
| C_minus_A / MRR@5 | +0.050000 | [+0.000000, +0.150000] |

Truncation subsets, upstream coverage@20/50, unique ledger usage, overhead-inclusive timing and descriptive projections are in [evaluation.json](evaluation.json). Scope usage overlaps other scopes; use the experiment ledger total for cumulative spend. These projections do not authorize expansion.

Experiment unique settlements: 40; unresolved attempts: 0; settled USD 0.020591466; recorded CAD with fee/buffer 0.04027845185595000 (excluding any separate fixed fee).
Scope invocation wall time: 32.086s; experiment elapsed including gaps: 1523.254s.
