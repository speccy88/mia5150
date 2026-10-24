# Proposal gap closure: measured local primary results

Research draft. The complete primary grid contains 200 exposed official queries × five methods × four catalog sizes =4000 planned cases. Additional 24-query seed/model sensitivity, 12 generated probes and full 4,392-query retrieval are separate stages. Their status follows execution receipts; they are not pooled into this primary analysis.

## Findings

At 6,006 skills, BM25 retrieved 46.17% of positive qrels in its top five and the selector exactly matched 9/200 sets. Vector retrieved 51.58% with 7/200 exact sets. Hybrid retrieved 57.17% with 11/200 exact sets. Domain vector returned 4/200 exact sets. These are outcomes of a small constrained 0.6B deployment, not skill execution or native SkillsMiddleware results.

Full listing required a mean 411,291 locally tokenized prompt tokens at 6,006 skills and all 200 cases failed the configured 8,192-token preflight. At 100 skills, 196 valid listing outputs produced zero exact sets, so context fit alone did not establish correct selection. The tokenized preflight length is not model-consumed usage.

The predeclared hybrid-minus-BM25 difference at 6,006 is about +1.02 percentage points. Its paired stratified 95% interval spans zero. The supplied proposal supports an accuracy/efficiency trade-off question, not a predetermined winner. Nominal family-adjusted intervals are also reported; no universal superiority claim follows from this exposed sample.

## Primary method-size table

|N|Method|Planned|Valid|Exact sets|Prevalence-weighted exact|Macro Recall@5|MRR@5|
|---:|---|---:|---:|---:|---:|---:|---:|
|10|listing|200|178|23|11.47%|unranked|unranked|
|10|bm25|200|196|46|22.93%|0.9608|0.8762|
|10|vector|200|193|40|19.92%|0.9233|0.9352|
|10|domain_vector|200|181|27|13.46%|0.5817|0.6822|
|10|hybrid|200|197|41|20.48%|0.9633|0.9539|
|100|listing|200|196|0|0.00%|unranked|unranked|
|100|bm25|200|195|26|12.96%|0.7608|0.7749|
|100|vector|200|198|23|11.44%|0.7675|0.8546|
|100|domain_vector|200|197|19|9.46%|0.5158|0.6108|
|100|hybrid|200|198|28|13.95%|0.8225|0.8708|
|1000|listing|200|0|0|0.00%|unranked|unranked|
|1000|bm25|200|197|24|11.96%|0.6192|0.6418|
|1000|vector|200|194|11|5.48%|0.6350|0.7201|
|1000|domain_vector|200|196|8|3.98%|0.4317|0.5092|
|1000|hybrid|200|192|10|5.00%|0.6975|0.7710|
|6006|listing|200|0|0|0.00%|unranked|unranked|
|6006|bm25|200|196|9|4.48%|0.4617|0.5323|
|6006|vector|200|197|7|3.48%|0.5158|0.5823|
|6006|domain_vector|200|198|4|1.99%|0.3483|0.3985|
|6006|hybrid|200|197|11|5.49%|0.5717|0.6371|

Exact estimates use official 1/2/3-positive prevalence weights 2143/4392,1703/4392,546/4392. Raw counts retain the sample denominator 200. Retrieval measures are sample-macro means with stage failures scored zero; conditional observed measures and their counts remain in CSV. Listing has full-library coverage 1 by construction and no semantic Recall@5.

## Efficiency and resources

Input/output tokens include each fresh routing stage. Preflight counts, returned model-evaluation counts and unknown failed usage have distinct meanings. Report successful latency separately from failure latency. Relative latency uses paired valid cases with BM25 plus the same selector at the same size and implementation; early implementation timing is retained but excluded from the corrected comparison. Startup/indexing are separate.

The local machine provides an RTX 2060 SUPER with 8 GB VRAM and 16 GB host RAM. Existing WSL settings cap the guest at 6 GB and two virtual CPUs with 2 GB swap. Those settings were read, never changed. Work uses one GPU job at a time and a CPU encoder. The official BGE ONNX export passed a non-evaluation PyTorch CPU parity test with maximum absolute difference 1.23e-7. CPU/CUDA setup interruptions, RAM guards and a result-append failure remain in receipts. A single generated-case outcome is unavailable after the append error and is not retried. No primary prediction was lost.

## Evidence and reproducibility

- [Frozen protocol](../protocol.json), [specification](../SPEC.md), [exact catalogs](../catalogs.json.gz) and [model provenance](../model-provenance.json).
- [Selection CSV](selection-summary.csv), [strata](strata.csv), [paired uncertainty](paired-primary.csv), [error taxonomy](error-taxonomy.csv), [domain diagnostics](domain-diagnostics.csv) and [independent validation](validation.json).
- [Measurement corrections](../MEASUREMENT_CORRECTIONS.md), [implementation corrections](../IMPLEMENTATION_CORRECTIONS.md) and raw/source snapshots in the study folder.
- Figures: [selection heatmap](selection-heatmap.png), [token growth](input-growth.png), [latency](latency.png), [candidate recall](candidate-recall.png) and [paired differences](paired-differences.png).

## Scope and remaining evidence

The supplied four-page proposal remains the primary historical target; the revised LaTeX proposal is a separate version. The original 80 model-method-size cells have explicit statuses. Local Qwen does not complete unrun hosted cells. All 4,392 public test queries are exposed; this 200-query frozen sample is not a fresh holdout. Qrels may omit plausible alternatives. Metadata views, pools and panels differ across historical studies, which remain supporting evidence only.

The balanced 24-query sensitivity must be reported separately from the primary prevalence estimate. Two local model configurations differ in architecture, training and quantization as well as parameter count; they are not a clean causal size experiment. Generated compositional labels are silver unions, not human adjudication or proof of dependency semantics. Sequence ordering, actual activation/execution, independently authored requests and two-person agreement remain pending. No human label or contribution is invented.

Hosted DeepSeek V4 Flash 0731 / DeepInfra FP8 is separately planned and authorized within a tighter US$1.10/600-call guard. The complete 24-query plan reserves 14,032,062 input and 74,112 output tokens over 579 requests, including routing/canaries, with no cache discount or retries. It estimates US$0.855 before 25% reserve. Credential availability and real provider canaries determine whether it can run; no hosted success or cost is invented.

## Draft authorship and review

Permitted team aliases: deepujoy88, speccy88, vsimard. This AI-assisted engineering and research draft requires team review. Individual contributions and final course submission approval remain for the team to confirm.

## Matched balanced model sensitivity at 6,006 skills

|Model|N|Method|Planned|Valid|Unattempted|Exact sets|Macro F1|
|---|---:|---|---:|---:|---:|---:|---:|
|qwen3_06_q8|6006|listing|24|0|0|0|0.0000|
|qwen3_06_q8|6006|bm25|24|24|0|2|0.3358|
|qwen3_06_q8|6006|vector|24|24|0|0|0.3098|
|qwen3_06_q8|6006|domain_vector|24|24|0|0|0.2266|
|qwen3_06_q8|6006|hybrid|24|24|0|1|0.3587|
|qwen35_4b_q4|6006|listing|24|0|0|0|0.0000|
|qwen35_4b_q4|6006|bm25|24|24|0|4|0.4264|
|qwen35_4b_q4|6006|vector|24|24|0|1|0.4052|
|qwen35_4b_q4|6006|domain_vector|24|24|0|0|0.2417|
|qwen35_4b_q4|6006|hybrid|24|24|0|1|0.4597|

## Generated silver-label probes

|Model|N|Method|Planned|Valid|Unattempted|Exact sets|Macro F1|
|---|---:|---|---:|---:|---:|---:|---:|
|qwen3_06_q8|100|listing|12|12|0|0|0.0714|
|qwen3_06_q8|100|bm25|12|12|0|1|0.3722|
|qwen3_06_q8|100|vector|12|12|0|0|0.4222|
|qwen3_06_q8|100|domain_vector|12|12|0|0|0.3667|
|qwen3_06_q8|100|hybrid|12|11|0|1|0.4250|
|qwen3_06_q8|6006|listing|12|0|0|0|0.0000|
|qwen3_06_q8|6006|bm25|12|12|0|0|0.2778|
|qwen3_06_q8|6006|vector|12|12|0|0|0.2583|
|qwen3_06_q8|6006|domain_vector|12|12|0|0|0.1889|
|qwen3_06_q8|6006|hybrid|12|12|0|0|0.1643|
|qwen35_4b_q4|100|listing|12|12|0|1|0.6119|
|qwen35_4b_q4|100|bm25|12|12|0|3|0.3500|
|qwen35_4b_q4|100|vector|12|12|0|3|0.4639|
|qwen35_4b_q4|100|domain_vector|12|12|0|1|0.3472|
|qwen35_4b_q4|100|hybrid|12|12|0|4|0.5056|
|qwen35_4b_q4|6006|listing|12|0|0|0|0.0000|
|qwen35_4b_q4|6006|bm25|12|12|0|2|0.2833|
|qwen35_4b_q4|6006|vector|12|12|0|0|0.3500|
|qwen35_4b_q4|6006|domain_vector|12|12|0|0|0.2250|
|qwen35_4b_q4|6006|hybrid|12|12|0|0|0.3278|

Balanced sensitivity is distinct from the primary prevalence-weighted estimate. Seeds do not add independent requests. Generated labels are silver unions; neither table establishes workflow order or human agreement. Inspect per-cell attempt status before interpreting any pending row.
