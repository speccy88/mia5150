# Pilot hosted snapshot

200/200 queries complete; all 200 selected queries remain in metric denominators.

| Metric | A matched dense | B Jev | C RRF60 |
|---|---:|---:|---:|
| NDCG@5 | 0.810919 | 0.648771 | 0.793868 |
| Recall@5 | 0.843333 | 0.760000 | 0.850833 |
| Completeness@5 | 0.725000 | 0.645000 | 0.745000 |
| MAP@5 | 0.763833 | 0.576681 | 0.742639 |
| Precision@5 | 0.259000 | 0.232000 | 0.264000 |
| NDCG@10 | 0.830145 | 0.693412 | 0.814578 |
| Recall@10 | 0.890000 | 0.875000 | 0.903333 |
| Completeness@10 | 0.795000 | 0.790000 | 0.815000 |
| MAP@10 | 0.776222 | 0.601270 | 0.754974 |
| Precision@10 | 0.140000 | 0.137000 | 0.142000 |
| NDCG@15 | 0.836555 | 0.701727 | 0.821250 |
| Recall@15 | 0.910833 | 0.900833 | 0.922500 |
| Completeness@15 | 0.830000 | 0.820000 | 0.855000 |
| MAP@15 | 0.779064 | 0.604620 | 0.758514 |
| Precision@15 | 0.095667 | 0.094667 | 0.097667 |
| MRR@5 | 0.868667 | 0.673750 | 0.829833 |
| MRR@10 | 0.870750 | 0.684494 | 0.834060 |

| Comparison / metric | Delta | 95% paired bootstrap CI |
|---|---:|---|
| B_minus_A / NDCG@10 | -0.136732 | [-0.174832, -0.098768] |
| B_minus_A / Recall@5 | -0.083333 | [-0.130000, -0.035833] |
| B_minus_A / Recall@10 | -0.015000 | [-0.045000, +0.015833] |
| B_minus_A / Completeness@5 | -0.080000 | [-0.140000, -0.020000] |
| B_minus_A / MRR@5 | -0.194917 | [-0.251250, -0.139581] |
| C_minus_A / NDCG@10 | -0.015567 | [-0.037712, +0.006697] |
| C_minus_A / Recall@5 | +0.007500 | [-0.015000, +0.030833] |
| C_minus_A / Recall@10 | +0.013333 | [-0.005000, +0.033333] |
| C_minus_A / Completeness@5 | +0.020000 | [-0.015000, +0.060000] |
| C_minus_A / MRR@5 | -0.038833 | [-0.075669, -0.002000] |

Truncation subsets, upstream coverage@20/50, unique ledger usage, overhead-inclusive timing and descriptive projections are in [evaluation.json](evaluation.json). Scope usage overlaps other scopes; use the experiment ledger total for cumulative spend. These projections do not authorize expansion.

Experiment unique settlements: 779; unresolved attempts: 1; settled USD 0.403215288; recorded CAD with fee/buffer 0.78871934447460000 (excluding any separate fixed fee).
Held unresolved allowance: USD 0.002752512; CAD 0.00538411991040000 with recorded fee/buffer. This is not a known charge.
Scope invocation wall time bounds: 709.615s to 1511.962s; 1 invocation(s) lack final receipts. Experiment elapsed including gaps: 3640.503s.
