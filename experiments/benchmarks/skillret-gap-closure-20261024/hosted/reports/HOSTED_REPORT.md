# Hosted diagnostic follow-up

Recorded 120/360 originally planned model/method cases. Qwen is the sole active benchmark model after the provider-specific continuation amendment; its target is 120 cases. DeepInfra models remain unattempted beyond their preserved canaries.

The shared panel has 24 previously exposed official queries: eight each with one, two and three positive qrels. Unweighted results estimate this balanced diagnostic, not official full-test prevalence. The object-wrapper v2 output contract differs from the local grammar. Retrieval inputs are reused; timing excludes retrieval and query embedding preparation.

Physical generation attempts, including all canaries and failures: **149**. Known billed generation costs: **US$0.02411946**. Unknown liabilities: **US$0.16** across 2 attempts. Actual total spend remains **unknown**. Guarded total is **US$0.18411946**, below the shared US$2/600 ceiling. These totals include canaries and routing; read-only HTTP events are separate.

|Method|Attempted / planned|Valid|Exact sets|Balanced macro F1|Known benchmark USD|
|---|---:|---:|---:|---:|---:|
|listing|24 / 24|24|5|0.5446|0.01609719|
|bm25|24 / 24|24|13|0.7236|0.00186564|
|vector|24 / 24|24|7|0.6292|0.00176991|
|domain_vector|24 / 24|24|4|0.5042|0.00248013|
|hybrid|24 / 24|24|10|0.6766|0.00189069|

Paired stratified bootstrap intervals resample independent query IDs within the three gold-cardinality strata, keeping methods paired. They are descriptive nominal 95% intervals over 24 exposed requests; no broad superiority, generalization, causal model-size or original 80-cell closure claim follows. Failed attempted decisions retain zero exact/F1 under the declared rule; unattempted cases remain unmeasured.

See [selection table](hosted-selection.csv), [paired intervals](hosted-paired-bootstrap.csv), [owner validation](hosted-validation.json), [physical API ledger](api-call-ledger.jsonl) and [guard status](status.json).
