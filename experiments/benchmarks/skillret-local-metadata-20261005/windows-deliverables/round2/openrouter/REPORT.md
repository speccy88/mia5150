# OpenRouter stopped exploratory probe

```json
{
  "audit": "Frozen12 IDs/pools/request hashes verified; four valid permutations, one failed attempt, seven unattempted; no retries.",
  "attempts": 5,
  "known_reported_cost_usd": "0.0007057050",
  "reserved_usd": "0.50",
  "median_success_latency_seconds": 2.686270633499589,
  "results_sha256": "34f608faaed036afce0fe076ce2e88000f4913a2ca72e274a6b5582aaf502456",
  "caveat": "Stopped run, not a completed model-quality comparison. Eight empty outputs retain denominator12. Successful subset is selection-biased and only descriptive. HTTP status was not retained by original safe logger; no error body retained or retry performed.",
  "baseline_ndcg10": 0.7162358536576369,
  "fixed_denominator_ndcg10": 0.2660566826769815,
  "success_subset_ndcg10": {
    "baseline": 0.6362767355002242,
    "rerank": 0.7981700480309445
  }
}
```
