# Actual hosted study findings

Dense retrieval has the highest pilot NDCG@10 point estimate. Jev reduced NDCG@10 by 0.136732; RRF60 changed it by -0.015567 (95% paired bootstrap CI [-0.037712, 0.006697]). RRF60 did not demonstrate an NDCG improvement, although its Recall@10 increased from 0.890000 to 0.903333. These conclusions apply to the fixed 200-query pilot.


The full local SkillRet baseline passed its reproduction gate on all 4,392 queries and 6,006 skills. NDCG@5/10/15: 0.790523 / 0.810990 / 0.818848; maximum absolute official-target difference 0.000323 (gate 0.005).

Only the supplied immutable hosted evaluations are reported below. No missing smoke, pilot or full study result is inferred. Empty or failed selected queries keep zero scores and remain in each fixed denominator.

## Smoke

10/10 queries have complete hosted rankings. Actual model: `jev-1.13.0`. A, B and C use the same twenty candidates.

| Metric | A matched dense | B Jev | C RRF60 |
|---|---:|---:|---:|
| NDCG@5 | 0.917330 | 0.841622 | 0.946928 |
| NDCG@10 | 0.949688 | 0.855749 | 0.963644 |
| NDCG@15 | 0.949688 | 0.855749 | 0.976326 |
| Recall@5 | 0.933333 | 0.933333 | 0.933333 |
| Recall@10 | 1.000000 | 0.966667 | 0.966667 |
| Recall@15 | 1.000000 | 0.966667 | 1.000000 |
| Completeness@5 | 0.900000 | 0.900000 | 0.900000 |
| Completeness@10 | 1.000000 | 0.900000 | 0.900000 |
| Completeness@15 | 1.000000 | 0.900000 | 1.000000 |
| MAP@5 | 0.891667 | 0.794444 | 0.933333 |
| MAP@10 | 0.917063 | 0.801852 | 0.944444 |
| MAP@15 | 0.917063 | 0.801852 | 0.952778 |
| Precision@5 | 0.240000 | 0.240000 | 0.240000 |
| Precision@10 | 0.140000 | 0.130000 | 0.130000 |
| Precision@15 | 0.093333 | 0.086667 | 0.093333 |
| MRR@5 | 0.950000 | 0.833333 | 1.000000 |
| MRR@10 | 0.950000 | 0.833333 | 1.000000 |

| Paired comparison | Metric | Delta | 95% interval |
|---|---|---:|---|
| B_minus_A | NDCG@10 | -0.093940 | [-0.196381, +0.000000] |
| B_minus_A | Recall@5 | +0.000000 | [+0.000000, +0.000000] |
| B_minus_A | Recall@10 | -0.033333 | [-0.100000, +0.000000] |
| B_minus_A | Completeness@5 | +0.000000 | [+0.000000, +0.000000] |
| B_minus_A | MRR@5 | -0.116667 | [-0.250000, +0.000000] |
| C_minus_A | NDCG@10 | +0.013956 | [+0.000000, +0.038512] |
| C_minus_A | Recall@5 | +0.000000 | [+0.000000, +0.000000] |
| C_minus_A | Recall@10 | -0.033333 | [-0.100000, +0.000000] |
| C_minus_A | Completeness@5 | +0.000000 | [+0.000000, +0.000000] |
| C_minus_A | MRR@5 | +0.050000 | [+0.000000, +0.150000] |

B_minus_A: 10,000 paired query resamples, seed 20261006; fixed denominator 10.
C_minus_A: 10,000 paired query resamples, seed 20261006; fixed denominator 10.

| Dense upstream coverage | Macro recall | Relevant pairs | Pair recall |
|---|---:|---:|---:|
| @20 | 1.000000 | 14/14 | 1.000000 |
| @50 | 1.000000 | 14/14 | 1.000000 |

### any candidate skill md truncated

Queries: 4.

| Metric | A matched dense | B Jev | C RRF60 |
|---|---:|---:|---:|
| NDCG@5 | 0.824020 | 0.716392 | 0.867320 |
| NDCG@10 | 0.904917 | 0.751709 | 0.909110 |
| NDCG@15 | 0.904917 | 0.751709 | 0.940814 |
| Recall@5 | 0.833333 | 0.833333 | 0.833333 |
| Recall@10 | 1.000000 | 0.916667 | 0.916667 |
| Recall@15 | 1.000000 | 0.916667 | 1.000000 |
| Completeness@5 | 0.750000 | 0.750000 | 0.750000 |
| Completeness@10 | 1.000000 | 0.750000 | 0.750000 |
| Completeness@15 | 1.000000 | 0.750000 | 1.000000 |
| MAP@5 | 0.791667 | 0.652778 | 0.833333 |
| MAP@10 | 0.855159 | 0.671296 | 0.861111 |
| MAP@15 | 0.855159 | 0.671296 | 0.881944 |
| Precision@5 | 0.250000 | 0.250000 | 0.250000 |
| Precision@10 | 0.175000 | 0.150000 | 0.150000 |
| Precision@15 | 0.116667 | 0.100000 | 0.116667 |
| MRR@5 | 0.875000 | 0.708333 | 1.000000 |
| MRR@10 | 0.875000 | 0.708333 | 1.000000 |

| Paired comparison | Metric | Delta | 95% interval |
|---|---|---:|---|
| B_minus_A | NDCG@10 | -0.153208 | [-0.306416, +0.000000] |
| B_minus_A | Recall@5 | +0.000000 | [+0.000000, +0.000000] |
| B_minus_A | Recall@10 | -0.083333 | [-0.250000, +0.000000] |
| B_minus_A | Completeness@5 | +0.000000 | [+0.000000, +0.000000] |
| B_minus_A | MRR@5 | -0.166667 | [-0.375000, +0.000000] |
| C_minus_A | NDCG@10 | +0.004193 | [+0.000000, +0.012578] |
| C_minus_A | Recall@5 | +0.000000 | [+0.000000, +0.000000] |
| C_minus_A | Recall@10 | -0.083333 | [-0.250000, +0.000000] |
| C_minus_A | Completeness@5 | +0.000000 | [+0.000000, +0.000000] |
| C_minus_A | MRR@5 | +0.125000 | [+0.000000, +0.375000] |

B_minus_A: 10,000 paired query resamples, seed 20261006; fixed denominator 4.
C_minus_A: 10,000 paired query resamples, seed 20261006; fixed denominator 4.

### all twenty candidates full text

Queries: 6.

| Metric | A matched dense | B Jev | C RRF60 |
|---|---:|---:|---:|
| NDCG@5 | 0.979536 | 0.925108 | 1.000000 |
| NDCG@10 | 0.979536 | 0.925108 | 1.000000 |
| NDCG@15 | 0.979536 | 0.925108 | 1.000000 |
| Recall@5 | 1.000000 | 1.000000 | 1.000000 |
| Recall@10 | 1.000000 | 1.000000 | 1.000000 |
| Recall@15 | 1.000000 | 1.000000 | 1.000000 |
| Completeness@5 | 1.000000 | 1.000000 | 1.000000 |
| Completeness@10 | 1.000000 | 1.000000 | 1.000000 |
| Completeness@15 | 1.000000 | 1.000000 | 1.000000 |
| MAP@5 | 0.958333 | 0.888889 | 1.000000 |
| MAP@10 | 0.958333 | 0.888889 | 1.000000 |
| MAP@15 | 0.958333 | 0.888889 | 1.000000 |
| Precision@5 | 0.233333 | 0.233333 | 0.233333 |
| Precision@10 | 0.116667 | 0.116667 | 0.116667 |
| Precision@15 | 0.077778 | 0.077778 | 0.077778 |
| MRR@5 | 1.000000 | 0.916667 | 1.000000 |
| MRR@10 | 1.000000 | 0.916667 | 1.000000 |

| Paired comparison | Metric | Delta | 95% interval |
|---|---|---:|---|
| B_minus_A | NDCG@10 | -0.054427 | [-0.184535, +0.021253] |
| B_minus_A | Recall@5 | +0.000000 | [+0.000000, +0.000000] |
| B_minus_A | Recall@10 | +0.000000 | [+0.000000, +0.000000] |
| B_minus_A | Completeness@5 | +0.000000 | [+0.000000, +0.000000] |
| B_minus_A | MRR@5 | -0.083333 | [-0.250000, +0.000000] |
| C_minus_A | NDCG@10 | +0.020464 | [+0.000000, +0.061392] |
| C_minus_A | Recall@5 | +0.000000 | [+0.000000, +0.000000] |
| C_minus_A | Recall@10 | +0.000000 | [+0.000000, +0.000000] |
| C_minus_A | Completeness@5 | +0.000000 | [+0.000000, +0.000000] |
| C_minus_A | MRR@5 | +0.000000 | [+0.000000, +0.000000] |

B_minus_A: 10,000 paired query resamples, seed 20261006; fixed denominator 6.
C_minus_A: 10,000 paired query resamples, seed 20261006; fixed denominator 6.

These are descriptive query groups: a query is in the truncated group if any of its twenty candidates was shortened. They are not randomized groups or a causal estimate of truncation harm.

### Response validation and confidence sensitivity

Native provider score and confidence are preserved in the primary ranking. Structural/range/provenance validation and any arithmetic diagnostics are reported as recorded; arithmetic consistency is not silently inferred from an accepted response.

| Recorded validation mode | Pair records |
|---|---:|
| cent | 190 |
| inconsistent | 1 |
| strict | 9 |

| Per-query mode set | Queries |
|---|---:|
| cent | 5 |
| cent;inconsistent;strict | 1 |
| cent;strict | 4 |

Counts cover score records retained in the immutable snapshot, normally complete queries; paid responses in an incomplete query may not yet appear here. No missing mode is inferred.

| Recorded diagnostic flag | True | False |
|---|---:|---:|
| arithmetic_diagnostics_only | 200 | 0 |
| checks.all_values_on_cent_grid | 200 | 0 |
| checks.confidence_within_strict_tolerance | 14 | 186 |
| checks.joint_nearest_cent_feasible | 199 | 1 |
| checks.probability_sum_within_strict_tolerance | 198 | 2 |
| checks.score_within_strict_tolerance | 97 | 103 |
| checks.strict_consistent | 9 | 191 |
| provider_values_preserved | 200 | 0 |
| rounding_is_observation_not_provider_guarantee | 200 | 0 |

Native score descending; exact native-score ties use original dense rank, omitting confidence. Exploratory; primary ranking remains unchanged.
Changed B query orders: 2; changed C query orders: 0. The denominator remains 10.

| Metric | Alternative B minus primary B | Alternative C minus primary C |
|---|---:|---:|
| NDCG@5 | +0.000000 | +0.000000 |
| NDCG@10 | +0.000677 | +0.000000 |
| NDCG@15 | +0.000677 | +0.000000 |
| Recall@5 | +0.000000 | +0.000000 |
| Recall@10 | +0.000000 | +0.000000 |
| Recall@15 | +0.000000 | +0.000000 |
| Completeness@5 | +0.000000 | +0.000000 |
| Completeness@10 | +0.000000 | +0.000000 |
| Completeness@15 | +0.000000 | +0.000000 |
| MAP@5 | +0.000000 | +0.000000 |
| MAP@10 | +0.000926 | +0.000000 |
| MAP@15 | +0.000926 | +0.000000 |
| Precision@5 | +0.000000 | +0.000000 |
| Precision@10 | +0.000000 | +0.000000 |
| Precision@15 | +0.000000 | +0.000000 |
| MRR@5 | +0.000000 | +0.000000 |
| MRR@10 | +0.000000 | +0.000000 |

### Timing and projection

Recorded scope invocation wall-time bounds: 32.086 to 32.086 s. Known API time in those invocations: 16.377 s. Wall time minus known API time: 15.709 to 15.709 s; this remainder can include unobserved API time and is not necessarily pure local overhead. Experiment elapsed including gaps: 1523.254 s.

Runner invocation start through immutable final receipt filesystem mtime (end proxy after ingestion), including local validation/checkpoint/cache/ingestion overhead. Invocation sums exclude gaps; experiment elapsed includes gaps. Parent command work before recorded runner start is not measured. Original API duration of reused responses is excluded from new API time.

Linear projection to 4392 queries: additional wall-time bounds 14060.3 to 14060.3 s; total experiment bounds including elapsed gaps 15583.5 to 15583.5 s. An upper planning bound is not an observed runtime or a six-hour guarantee.
Projected recorded CAD usage/held commitment: 17.69029605513324000, excluding separately configured fixed fees.
Projected additional USD usage 9.0231804012; USD usage plus held commitments at the same snapshot 9.0437718672.
Rate extrapolation uses settled observations. Any unknown historical reservation is a planning allowance, not measured token usage, and is added once rather than multiplied across future queries. Smoke/pilot overlap is not summed. This descriptive projection does not authorize expansion or resolve billing uncertainty.

### Largest query-level NDCG@10 changes

| Comparison | Direction | Query ID | Delta | A | B | C |
|---|---|---|---:|---:|---:|---:|
| B_minus_A | rise | `q-02554` | +0.042505 | 0.877215 | 0.919721 | 1.000000 |
| B_minus_A | fall | `q-00211` | -0.369070 | 1.000000 | 0.630930 | 1.000000 |
| B_minus_A | fall | `q-03606` | -0.369070 | 1.000000 | 0.630930 | 1.000000 |
| B_minus_A | fall | `q-00324` | -0.243762 | 0.619669 | 0.375906 | 0.636439 |
| C_minus_A | rise | `q-02554` | +0.122785 | 0.877215 | 0.919721 | 1.000000 |
| C_minus_A | rise | `q-00324` | +0.016771 | 0.619669 | 0.375906 | 0.636439 |

## Pilot

200/200 queries have complete hosted rankings. Actual model: `jev-1.13.0`. A, B and C use the same twenty candidates.

| Metric | A matched dense | B Jev | C RRF60 |
|---|---:|---:|---:|
| NDCG@5 | 0.810919 | 0.648771 | 0.793868 |
| NDCG@10 | 0.830145 | 0.693412 | 0.814578 |
| NDCG@15 | 0.836555 | 0.701727 | 0.821250 |
| Recall@5 | 0.843333 | 0.760000 | 0.850833 |
| Recall@10 | 0.890000 | 0.875000 | 0.903333 |
| Recall@15 | 0.910833 | 0.900833 | 0.922500 |
| Completeness@5 | 0.725000 | 0.645000 | 0.745000 |
| Completeness@10 | 0.795000 | 0.790000 | 0.815000 |
| Completeness@15 | 0.830000 | 0.820000 | 0.855000 |
| MAP@5 | 0.763833 | 0.576681 | 0.742639 |
| MAP@10 | 0.776222 | 0.601270 | 0.754974 |
| MAP@15 | 0.779064 | 0.604620 | 0.758514 |
| Precision@5 | 0.259000 | 0.232000 | 0.264000 |
| Precision@10 | 0.140000 | 0.137000 | 0.142000 |
| Precision@15 | 0.095667 | 0.094667 | 0.097667 |
| MRR@5 | 0.868667 | 0.673750 | 0.829833 |
| MRR@10 | 0.870750 | 0.684494 | 0.834060 |

| Paired comparison | Metric | Delta | 95% interval |
|---|---|---:|---|
| B_minus_A | NDCG@10 | -0.136732 | [-0.174832, -0.098768] |
| B_minus_A | Recall@5 | -0.083333 | [-0.130000, -0.035833] |
| B_minus_A | Recall@10 | -0.015000 | [-0.045000, +0.015833] |
| B_minus_A | Completeness@5 | -0.080000 | [-0.140000, -0.020000] |
| B_minus_A | MRR@5 | -0.194917 | [-0.251250, -0.139581] |
| C_minus_A | NDCG@10 | -0.015567 | [-0.037712, +0.006697] |
| C_minus_A | Recall@5 | +0.007500 | [-0.015000, +0.030833] |
| C_minus_A | Recall@10 | +0.013333 | [-0.005000, +0.033333] |
| C_minus_A | Completeness@5 | +0.020000 | [-0.015000, +0.060000] |
| C_minus_A | MRR@5 | -0.038833 | [-0.075669, -0.002000] |

B_minus_A: 10,000 paired query resamples, seed 20261006; fixed denominator 200.
C_minus_A: 10,000 paired query resamples, seed 20261006; fixed denominator 200.

| Dense upstream coverage | Macro recall | Relevant pairs | Pair recall |
|---|---:|---:|---:|
| @20 | 0.927500 | 295/323 | 0.913313 |
| @50 | 0.950833 | 304/323 | 0.941176 |

### any candidate skill md truncated

Queries: 96.

| Metric | A matched dense | B Jev | C RRF60 |
|---|---:|---:|---:|
| NDCG@5 | 0.816734 | 0.634499 | 0.795352 |
| NDCG@10 | 0.831648 | 0.684516 | 0.813846 |
| NDCG@15 | 0.836260 | 0.694739 | 0.827747 |
| Recall@5 | 0.857639 | 0.736111 | 0.842014 |
| Recall@10 | 0.890625 | 0.864583 | 0.885417 |
| Recall@15 | 0.904514 | 0.895833 | 0.925347 |
| Completeness@5 | 0.718750 | 0.593750 | 0.687500 |
| Completeness@10 | 0.770833 | 0.750000 | 0.760417 |
| Completeness@15 | 0.802083 | 0.791667 | 0.843750 |
| MAP@5 | 0.762876 | 0.559462 | 0.738773 |
| MAP@10 | 0.773413 | 0.587003 | 0.751319 |
| MAP@15 | 0.775769 | 0.591232 | 0.758693 |
| Precision@5 | 0.277083 | 0.235417 | 0.272917 |
| Precision@10 | 0.147917 | 0.142708 | 0.146875 |
| Precision@15 | 0.100694 | 0.099306 | 0.104167 |
| MRR@5 | 0.876215 | 0.667882 | 0.851389 |
| MRR@10 | 0.877517 | 0.678609 | 0.852877 |

| Paired comparison | Metric | Delta | 95% interval |
|---|---|---:|---|
| B_minus_A | NDCG@10 | -0.147132 | [-0.200799, -0.093303] |
| B_minus_A | Recall@5 | -0.121528 | [-0.185764, -0.060764] |
| B_minus_A | Recall@10 | -0.026042 | [-0.069444, +0.015625] |
| B_minus_A | Completeness@5 | -0.125000 | [-0.208333, -0.041667] |
| B_minus_A | MRR@5 | -0.208333 | [-0.289931, -0.127947] |
| C_minus_A | NDCG@10 | -0.017802 | [-0.048070, +0.012335] |
| C_minus_A | Recall@5 | -0.015625 | [-0.043403, +0.010417] |
| C_minus_A | Recall@10 | -0.005208 | [-0.029514, +0.019097] |
| C_minus_A | Completeness@5 | -0.031250 | [-0.083333, +0.010417] |
| C_minus_A | MRR@5 | -0.024826 | [-0.077604, +0.027431] |

B_minus_A: 10,000 paired query resamples, seed 20261006; fixed denominator 96.
C_minus_A: 10,000 paired query resamples, seed 20261006; fixed denominator 96.

### all twenty candidates full text

Queries: 104.

| Metric | A matched dense | B Jev | C RRF60 |
|---|---:|---:|---:|
| NDCG@5 | 0.805550 | 0.661945 | 0.792499 |
| NDCG@10 | 0.828757 | 0.701625 | 0.815253 |
| NDCG@15 | 0.836828 | 0.708177 | 0.815253 |
| Recall@5 | 0.830128 | 0.782051 | 0.858974 |
| Recall@10 | 0.889423 | 0.884615 | 0.919872 |
| Recall@15 | 0.916667 | 0.905449 | 0.919872 |
| Completeness@5 | 0.730769 | 0.692308 | 0.798077 |
| Completeness@10 | 0.817308 | 0.826923 | 0.865385 |
| Completeness@15 | 0.855769 | 0.846154 | 0.865385 |
| MAP@5 | 0.764717 | 0.592575 | 0.746207 |
| MAP@10 | 0.778816 | 0.614438 | 0.758349 |
| MAP@15 | 0.782105 | 0.616978 | 0.758349 |
| Precision@5 | 0.242308 | 0.228846 | 0.255769 |
| Precision@10 | 0.132692 | 0.131731 | 0.137500 |
| Precision@15 | 0.091026 | 0.090385 | 0.091667 |
| MRR@5 | 0.861699 | 0.679167 | 0.809936 |
| MRR@10 | 0.864503 | 0.689927 | 0.816690 |

| Paired comparison | Metric | Delta | 95% interval |
|---|---|---:|---|
| B_minus_A | NDCG@10 | -0.127132 | [-0.182390, -0.074262] |
| B_minus_A | Recall@5 | -0.048077 | [-0.116987, +0.019231] |
| B_minus_A | Recall@10 | -0.004808 | [-0.048077, +0.038462] |
| B_minus_A | Completeness@5 | -0.038462 | [-0.134615, +0.057692] |
| B_minus_A | MRR@5 | -0.182532 | [-0.258494, -0.107849] |
| C_minus_A | NDCG@10 | -0.013504 | [-0.046369, +0.018307] |
| C_minus_A | Recall@5 | +0.028846 | [-0.006410, +0.065705] |
| C_minus_A | Recall@10 | +0.030449 | [+0.004808, +0.062500] |
| C_minus_A | Completeness@5 | +0.067308 | [+0.019231, +0.125000] |
| C_minus_A | MRR@5 | -0.051763 | [-0.103526, -0.001603] |

B_minus_A: 10,000 paired query resamples, seed 20261006; fixed denominator 104.
C_minus_A: 10,000 paired query resamples, seed 20261006; fixed denominator 104.

These are descriptive query groups: a query is in the truncated group if any of its twenty candidates was shortened. They are not randomized groups or a causal estimate of truncation harm.

### Response validation and confidence sensitivity

Native provider score and confidence are preserved in the primary ranking. Structural/range/provenance validation and any arithmetic diagnostics are reported as recorded; arithmetic consistency is not silently inferred from an accepted response.

| Recorded validation mode | Pair records |
|---|---:|
| cent | 3659 |
| inconsistent | 33 |
| strict | 308 |

| Per-query mode set | Queries |
|---|---:|
| cent | 42 |
| cent;inconsistent | 8 |
| cent;inconsistent;strict | 23 |
| cent;strict | 127 |

Counts cover score records retained in the immutable snapshot, normally complete queries; paid responses in an incomplete query may not yet appear here. No missing mode is inferred.

| Recorded diagnostic flag | True | False |
|---|---:|---:|
| arithmetic_diagnostics_only | 4000 | 0 |
| checks.all_values_on_cent_grid | 4000 | 0 |
| checks.confidence_within_strict_tolerance | 457 | 3543 |
| checks.joint_nearest_cent_feasible | 3967 | 33 |
| checks.probability_sum_within_strict_tolerance | 3930 | 70 |
| checks.score_within_strict_tolerance | 1843 | 2157 |
| checks.strict_consistent | 308 | 3692 |
| provider_values_preserved | 4000 | 0 |
| rounding_is_observation_not_provider_guarantee | 4000 | 0 |

Native score descending; exact native-score ties use original dense rank, omitting confidence. Exploratory; primary ranking remains unchanged.
Changed B query orders: 58; changed C query orders: 38. The denominator remains 200.

| Metric | Alternative B minus primary B | Alternative C minus primary C |
|---|---:|---:|
| NDCG@5 | +0.003424 | -0.000433 |
| NDCG@10 | +0.001721 | -0.000418 |
| NDCG@15 | +0.001721 | -0.000418 |
| Recall@5 | +0.005000 | +0.000000 |
| Recall@10 | +0.000000 | +0.000000 |
| Recall@15 | +0.000000 | +0.000000 |
| Completeness@5 | +0.005000 | +0.000000 |
| Completeness@10 | +0.000000 | +0.000000 |
| Completeness@15 | +0.000000 | +0.000000 |
| MAP@5 | +0.003472 | -0.001000 |
| MAP@10 | +0.002755 | -0.000970 |
| MAP@15 | +0.002755 | -0.000970 |
| Precision@5 | +0.001000 | +0.000000 |
| Precision@10 | +0.000000 | +0.000000 |
| Precision@15 | +0.000000 | +0.000000 |
| MRR@5 | +0.001833 | +0.000250 |
| MRR@10 | +0.001000 | +0.000250 |

### Timing and projection

Recorded scope invocation wall-time bounds: 709.615 to 1511.962 s. Known API time in those invocations: 261.205 s. Wall time minus known API time: 448.410 to 1250.757 s; this remainder can include unobserved API time and is not necessarily pure local overhead. Experiment elapsed including gaps: 3640.503 s.

Final receipts use original filesystem mtime as an end proxy after ingestion. Missing-final invocations use immutable checkpoint activity as a lower bound and an explicit reviewed recovery window when available; a reboot timestamp is only an upper bound, never an exact exit. No success or exit is inferred for missing finals. Invocation sums exclude gaps; experiment elapsed includes gaps and the six-hour deadline remains anchored at the first ledger reservation. Parent work before runner start is unmeasured; reused API duration is excluded from new API time. Wall-minus-recorded-settled-API residuals can include unknown in-flight API time; genuine non-API lower bounds are therefore zero when missing-final or unresolved-attempt evidence remains.

Invocations without a final receipt: 1. Their last checkpoints and reviewed restart evidence supply bounds only. No final or exit receipt is invented, and an exact duration is not asserted.

Linear projection to 4392 queries: additional wall-time bounds 14873.5 to 31690.7 s; total experiment bounds including elapsed gaps 18514.0 to 35331.2 s. An upper planning bound is not an observed runtime or a six-hour guarantee.
Projected settled-rate CAD valuation: 17.32027680466221600; upper planning commitment including the historical unknown held allowance once: 17.32566092457261600. These are the evaluator's named planning bounds, not statistical cost intervals or known charges.
Corresponding USD settled-rate projection: 8.85460772448; upper planning commitment: 8.85736023648. Held unknown amount: USD 0.002752512 / CAD 0.00538411991040000.
Rate extrapolation uses settled observations. Any unknown historical reservation is a planning allowance, not measured token usage, and is added once rather than multiplied across future queries. Smoke/pilot overlap is not summed. This descriptive projection does not authorize expansion or resolve billing uncertainty.

### Largest query-level NDCG@10 changes

| Comparison | Direction | Query ID | Delta | A | B | C |
|---|---|---|---:|---:|---:|---:|
| B_minus_A | rise | `q-02216` | +0.819427 | 0.148041 | 0.967468 | 0.444123 |
| B_minus_A | rise | `q-03179` | +0.500000 | 0.500000 | 1.000000 | 1.000000 |
| B_minus_A | rise | `q-03716` | +0.500000 | 0.500000 | 1.000000 | 0.630930 |
| B_minus_A | rise | `q-04270` | +0.430677 | 0.000000 | 0.430677 | 0.356207 |
| B_minus_A | rise | `q-04524` | +0.430677 | 0.000000 | 0.430677 | 0.315465 |
| B_minus_A | fall | `q-04164` | -1.000000 | 1.000000 | 0.000000 | 0.500000 |
| B_minus_A | fall | `q-04251` | -1.000000 | 1.000000 | 0.000000 | 0.315465 |
| B_minus_A | fall | `q-03670` | -0.762802 | 1.000000 | 0.237198 | 0.877215 |
| B_minus_A | fall | `q-04907` | -0.682523 | 0.919721 | 0.237198 | 0.624051 |
| B_minus_A | fall | `q-01482` | -0.672833 | 0.877215 | 0.204382 | 0.613147 |
| C_minus_A | rise | `q-03645` | +0.569323 | 0.430677 | 0.630930 | 1.000000 |
| C_minus_A | rise | `q-03179` | +0.500000 | 0.500000 | 1.000000 | 1.000000 |
| C_minus_A | rise | `q-01332` | +0.389454 | 0.578014 | 0.967468 | 0.967468 |
| C_minus_A | rise | `q-01882` | +0.369070 | 0.630930 | 0.630930 | 1.000000 |
| C_minus_A | rise | `q-03218` | +0.369070 | 0.630930 | 1.000000 | 1.000000 |
| C_minus_A | fall | `q-04251` | -0.684535 | 1.000000 | 0.000000 | 0.315465 |
| C_minus_A | fall | `q-00394` | -0.500000 | 1.000000 | 0.386853 | 0.500000 |
| C_minus_A | fall | `q-04164` | -0.500000 | 1.000000 | 0.000000 | 0.500000 |
| C_minus_A | fall | `q-04503` | -0.500000 | 1.000000 | 0.356207 | 0.500000 |
| C_minus_A | fall | `q-04597` | -0.500000 | 1.000000 | 0.333333 | 0.500000 |

## Cumulative unique experiment accounting

Verified ledger head: `49e2214f29134e404010826d6cf36014ae724885f75b1d4efeda15d66933ee28`. Physical attempt reservations: 780; attempts with a preserved HTTP response: 779; settled responses: 779; unresolved reservations: 1; verified alias cache-reuse events: 40.

A reserved physical attempt is not proof of provider receipt or execution. A preserved response is evidence of an observed HTTP outcome. A reservation without a response can still have executed or been billed.

Observed settled USD credit usage: 0.403215288. Unresolved reserved allowance: USD 0.002752512, which is neither observed spend nor proof of no charge. Recorded CAD valuation of settled usage with configured fee/contingency/buffer: 0.78871934447460000; separate unknown held CAD allowance: 0.00538411991040000. CAD figures exclude any separately configured fixed experiment fee and are not asserted new card charges. No hypothetical per-call fee is added.

Observed settled input tokens: 9600364; fully reported output token total for settled responses: 61116. Lost/unobserved response usage is unknown and is not fabricated as zero tokens. Verified cache reuse adds no charge, but an explicitly permitted physical replacement request can duplicate an original charge. Smoke and pilot logical usage overlaps; only the unique ledger total is cumulative.

## Reviewed restart-recovery protocol exception

The reviewed receipt `d7b8c2ddc6c4f619fa43a7294cb065e737bf373e1524ff75c0fe9ade8ff15835` binds original attempt `f340d90743f01aac474aed7d9135f15599da30e73aa1f6bb08fed1cf6eb746dd-1` and the one permitted identical-request replacement `f340d90743f01aac474aed7d9135f15599da30e73aa1f6bb08fed1cf6eb746dd-2`. Coordinator-reviewed single identical-request exception within existing CAD10/USD5 limits; not a new human approval or evidence of no original charge.

Original preserved-response evidence: False; original reservation still unresolved: True. Its held unknown allowance is USD 0.002752512 / CAD 0.00538411991040000. Replacement reserved: True; replacement response preserved: True; replacement usage settled: True.

One possible duplicate execution and charge is explicitly disclosed if the replacement was sent. No original response is preserved, so its execution or billing outcome cannot be inferred from absent local response bytes. The exception retains the original reservation and separately accounts for the replacement; it does not erase or relabel the original attempt. Request bytes, candidates and selected query IDs remain fixed. The specified selection rule uses the first replacement response regardless of its scores; no choice between observed outputs is authorized.

The interrupted original invocation `ad65d43ed3f74704a617d9fbc6843f21` has a final receipt present in the supplied filesystem: False. Reviewed interruption bounds are 1791340744.65326 to 1791341547 UTC epoch seconds. No final or exit receipt is reconstructed or fabricated. Primary metrics retain all frozen selected IDs; completing the metrics does not resolve the original billing uncertainty.

## Historical response-parser repair

The saved first-response diagnostic records HTTP 200 with response hash `c69d7a4175615c73aacbe1966967b265e09a40a27edac1c77fb399eb9559bc4c` and original HTTP duration 0.418737 s. The initial score/probability consistency check rejected the serialized numeric precision. This was a local parser rejection, not a demonstrated service failure. The same preserved response is now uniquely settled in the ledger; no repair completion is inferred without that evidence.

API duration excludes the investigation and repair interval. Experiment wall time includes recorded gaps; later runner timing starts are distinct from the earlier HTTP duration. Current outcomes must come from immutable reevaluated snapshots, not from changing old raw response bytes.

## Limitations

This is an exposed public benchmark with a deterministic hash-selected pilot, not a newly blinded study. Paired bootstrap intervals describe the supplied query scope; they do not establish generalization or correct multiple comparisons. Full-text and truncated groups differ in their inputs and difficulty. Reranking cannot retrieve a gold skill missing from the fixed twenty-candidate pool; @50 coverage refers only to upstream dense retrieval.

The displayed score, confidence and probability fields do not always satisfy the documented arithmetic; the provider gives no exact serialization precision contract. Validation modes and numeric diagnostic flags are reported per pair/query; acceptance does not prove arithmetic agreement. Omitting or retrying queries selectively because of that disagreement could change the evaluation population. The exploratory no-confidence tie-break analysis never replaces the frozen primary results.

Truncation followed a conservative canonical-byte admission policy, not a verified native tokenizer boundary. Wall projections include measured local overhead but can miss future growth in ledger/cache validation and a different full-plan request mix. Interrupted invocations without final receipts have timing bounds, not exact durations. Receipt mtimes are timing proxies; use original WSL artifacts, since copying may change them. A reviewed physical replacement is a disclosed protocol exception, not ordinary zero-cost cache reuse. Any original unknown reservation remains separate from observed usage pending billing evidence. No full-run completion, budget clearance, absence of duplicate charge or future runtime guarantee is inferred.

[failure_queries.csv](failure_queries.csv) records the largest rises/falls. [failure_analysis.csv](failure_analysis.csv) gives skill IDs, gold grades, original dense ranks, native Jev ranks/scores and fusion ranks. Query text and skill text are deliberately omitted. Empty rank cells mean unavailable/not retrieved, not rank zero.

## Recorded resource evidence

Across saved hosted-run samples, minimum available memory was 4.525 GiB on the host and 4.797 GiB in WSL. GPU memory peaked at 0 MiB and temperature at 44 C; host CPU peaked at 66%. Host telemetry age stayed below 2.141 seconds in saved samples. No positive swap-counter increments or threshold violations were recorded. These extrema describe saved samples only. Run12 ends abruptly without exit or guard-stop receipt; its unrecorded final interval is not certified compliant. Run13 completed with exit0. See [resource audit](../hosted-resource-audit.json).
