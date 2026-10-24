# Timing and missing-usage corrections

Independent review of the first published implementation found three issues. Predictions already produced remain immutable.

1. The full-retrieval runner originally computed dense dot products before its search timer. No full-retrieval benchmark had run. Its first measured version now times catalog indexing, dot products and ranking together. Batch query embedding remains a separate setup measurement.
2. Early domain-vector cases computed scores for the complete available catalog and filtered the result to the routed domain. This gives valid restricted-ranking semantics but cannot establish computational savings from searching only a subset. Later unattempted cases route first and compute scores only for that domain. Source hashes and a per-case implementation label distinguish these versions. Quality uses the same frozen domain restriction and pinned vectors. Fresh latency comparisons exclude the earlier implementation and use matched valid query cases from one implementation.
3. A transport/guard failure originally initialized output tokens to zero. That does not establish actual usage. New failures retain null usage. Reporting treats historical failed-call usage as unknown without modifying its raw receipt. Parseable-only micro precision shows its case denominator and unknown-cardinality count. Unqualified all-planned micro precision/F1 remain null when malformed or lost outputs have unknown cardinality. Planned exact-set and macro-zero-on-failure results remain defined.

These are implementation/measurement repairs, not outcome-driven changes to queries, gold labels, prompts, model pins, fusion parameters or the primary contrasts. Attempted failures are retained. Only rows explicitly marked unattempted may receive a later attempt.

## Exact interruption accounting

The second interruption snapshot contains5080 planned small-model cases:874 valid,100 preflight context exclusions,20 invalid selections,4 invalid/empty-domain outcomes,3 pipeline/resource failures and4079 unattempted. The earlier short progress message omitted the3 pipeline failures when listing the other failure categories. They remain in the ledger. More recent status snapshots supersede these counts as unattempted cases resume; attempted failures remain unchanged.
