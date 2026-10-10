# Jev typed API exploratory results

12 successful unique attempts, frozen requests/pools and score-derived rankings verified

| Method | NDCG@10 | Recall@10 | Completeness@5 |
|---|---:|---:|---:|
| SkillRet | 0.716236 | 0.902778 | 0.583333 |
| Jev Noul | 0.740272 | 0.972222 | 0.916667 |
| Jev Score | 0.758343 | 0.972222 | 0.833333 |

{
  "Jev Noul": {
    "delta_ndcg10": 0.024036345995787897,
    "paired_bootstrap95": [
      -0.15289581653311998,
      0.2043942353855913
    ],
    "wins": 5,
    "ties": 3,
    "losses": 4
  },
  "Jev Score": {
    "delta_ndcg10": 0.04210694387148509,
    "paired_bootstrap95": [
      -0.14176782391363185,
      0.22848766645947324
    ],
    "wins": 6,
    "ties": 2,
    "losses": 4
  }
}

12 already-exposed development queries; two exploratory hypotheses, unadjusted bootstrap intervals; no blind test or general improvement claim. All methods share frozen top20 coverage. API latency excludes retrieval and is not a matched hardware comparison.

Known token usage: $0.003292884. Keep $0.10 reserved plus $1 external, total $1.10 until billing reconciliation.