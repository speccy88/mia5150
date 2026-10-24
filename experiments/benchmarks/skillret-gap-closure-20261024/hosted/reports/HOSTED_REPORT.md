# Hosted diagnostic follow-up

Recorded 120/360 originally planned model/method cases. Qwen is the sole active benchmark model after the provider-specific continuation amendment; its target is 120 cases. DeepInfra models remain unattempted beyond their preserved canaries.

The shared panel has 24 previously exposed official queries: eight each with one, two and three positive qrels. Unweighted results estimate this balanced diagnostic, not official full-test prevalence. The object-wrapper v2 output contract differs from the local grammar. Retrieval inputs are reused; timing excludes retrieval and query embedding preparation.

Physical generation attempts, including all canaries and failures: **150**. Known billed generation costs: **US$0.02411946**. Unknown liabilities: **US$0.19** across3 attempts. Actual total spend remains **unknown**. Guarded total is **US$0.21411946**, below the shared US$2/600 ceiling. These totals include canaries and routing; read-only HTTP events are separate.

|Method|Attempted / planned|Valid|Exact sets|Balanced macro F1|Known benchmark USD|
|---|---:|---:|---:|---:|---:|
|listing|24 / 24|24|5|0.5446|0.01609719|
|bm25|24 / 24|24|13|0.7236|0.00186564|
|vector|24 / 24|24|7|0.6292|0.00176991|
|domain_vector|24 / 24|24|4|0.5042|0.00248013|
|hybrid|24 / 24|24|10|0.6766|0.00189069|

Paired stratified bootstrap intervals resample independent query IDs within the three gold-cardinality strata, keeping methods paired. They are descriptive nominal 95% intervals over 24 exposed requests; no broad superiority, generalization, causal model-size or original 80-cell closure claim follows. Failed attempted decisions retain zero exact/F1 under the declared rule; unattempted cases remain unmeasured.

See [selection table](hosted-selection.csv), [paired intervals](hosted-paired-bootstrap.csv), [owner validation](hosted-validation.json), [physical API ledger](api-call-ledger.jsonl) and [guard status](status.json).

## Later exploratory Mistral arm and terminal scope

The later arm was frozen after Qwen results on the same 24 queries, N=100 catalogs, prompts, output contract and budget rules. Its Parasail BF16 deployment differs from Qwen/SiliconFlow FP8, so a successful comparison would still have provider/precision/timing confounding. Its first canary returned HTTP 429 with typed source upstream_provider_shared_pool, no Retry-After header, no generation ID and no usage/cost. No Mistral selector or routing case was attempted; no retry or further model followed.

The original frozen selector plan remains 360 cases: 120 completed Qwen and 240 unattempted DeepSeek/Gemma. The later added 120 Mistral cases are separately unattempted. Combined: 480 planned, 120 completed, 360 unattempted. No unattempted cell is assigned a performance zero. Only one model has successful benchmark evidence; a multi-model quality comparison remains incomplete.

Terminal generation count is 150: 147 successful Qwen requests plus three failed provider canaries. Recorded read-only GET count is 9 (earlier uninstrumented total remains unknown). Known charges stay US$0.02411946. Conservative unknown holds are US$0.14 DeepSeek, US$0.02 Gemma and US$0.03 Mistral; the final increase retains STOP and grants no continuation. Actual total cost is unknown; guarded total is US$0.21411946. See [terminal receipt](../mistral-reports/terminal-execution.json) and [unattempted Mistral cells](../mistral-reports/unattempted-mistral-cells.csv).
