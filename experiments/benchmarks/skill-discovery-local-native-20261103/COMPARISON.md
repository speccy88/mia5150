# Descriptive comparison: identical libraries, distinct inference conditions

The public copies preserve measured outcomes. This comparison does not establish API/OAuth/local equivalence. The initial local series uses quantized models, a 32K window, a 500-token output cap and a native final-answer adapter. The separate v3 supplement adjusts context to actual input tokens +500 and compares f16 with Q8_0 KV at N=500. OAuth omits temperature/output controls and reports medium reasoning. No new model calls were made for the comparison.

![Measured quality and coverage](figures/coverage-comparison.png)

The plot uses the same 20 questions at each size. Every colored zero comes from actual measured outputs; missing scores indicate exclusions, blocks or unattempted sizes. Format errors and reasoning without final answers stay in the denominator of measured local outputs.

## Shared panel: N = 50, 100, 250, 500

These scores use 80 measurements per executed model: 20 questions repeated at four sizes. Intervals resample 20 question clusters with all four sizes together, 10,000 times, seed 20261103. They are descriptive and conditional on the biased lists, rather than treating 80 cases as independent. Blocked models receive no interval.

| Route | Model | Measurements | Top-1 exact | Question-cluster bootstrap, 95% | Outputs without errors | Median LLM (s) |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| local | gemma-3-4b-it | 0 | Not measured | — | 0 | — |
| local | gemma-4-e2b-it-q4 | 80 | 34/80 (42.5%) | 25.0% to 61.3% | 34 | 7.292 |
| local | h2o-lightning-4b-stock-q4km | 80 | 47/80 (58.8%) | 40.0% to 76.2% | 50 | 8.362 |
| local | lfm3-omni-2b-q4km | 80 | 57/80 (71.2%) | 53.8% to 87.5% | 75 | 2.036 |
| local | ling-3.0-tiny-q4km | 80 | 38/80 (47.5%) | 31.2% to 63.7% | 40 | 7.976 |
| local | llama-3.2-3b-instruct | 0 | Not measured | — | 0 | — |
| local | mistral-7b-instruct-v03-q4km | 0 | Not measured | — | 0 | — |
| local | nanbeige4.2-3b-q4km | 80 | 0/80 (0.0%) | 0.0% to 0.0% | 0 | 44.271 |
| local | nemotron-3.5-bolt-e4b-q4km | 80 | 21/80 (26.2%) | 15.0% to 38.8% | 21 | 9.637 |
| local | neohorse-1-9b-q4km | 80 | 45/80 (56.2%) | 36.2% to 75.0% | 46 | 12.655 |
| local | ornith-1.5-9b-q4km | 80 | 50/80 (62.5%) | 46.2% to 77.5% | 57 | 12.614 |
| local | phi-4-mini-instruct-q4km | 80 | 52/80 (65.0%) | 46.2% to 82.5% | 58 | 5.718 |
| local | qwen3-0.6b-q8 | 80 | 49/80 (61.3%) | 48.8% to 72.5% | 57 | 3.689 |
| local | qwen3-1.7b-q8 | 80 | 19/80 (23.8%) | 11.2% to 37.5% | 20 | 5.483 |
| local | qwen3-8b-q4km | 0 | Not measured | — | 0 | — |
| local | qwen3.5-4b-q4km | 80 | 28/80 (35.0%) | 22.5% to 47.5% | 32 | 8.854 |
| local | qwen3.5-9b-q4km | 80 | 25/80 (31.2%) | 17.5% to 46.2% | 27 | 13.490 |
| local | ternary-bonsai-2-9b-q2 | 80 | 23/80 (28.7%) | 17.5% to 40.0% | 25 | 10.925 |
| OAuth | gpt-6-luna | 80 | 62/80 (77.5%) | 60.0% to 92.5% | 80 | 2.593 |
| OAuth | gpt-6.1-sol | 80 | 60/80 (75.0%) | 56.2% to 91.2% | 80 | 5.239 |
| OAuth | gpt-6.1-terra | 80 | 63/80 (78.8%) | 61.3% to 95.0% | 80 | 2.758 |

## Limitations and evidence

- **Position bias:** all expected correct skills (gold labels) occupy positions 1–20, using the same seed as question selection. Differences cannot be generalized to random or central positions.
- **Initial local coverage:** 14 models executed, 1,120 outputs with actual usage; four models without inference. The 1,080 estimated-context rows do not measure native maximum model windows.
- **OAuth coverage:** 360 measured responses at N≤2000. At N=5000 each model has one HTTP string-length rejection and 19 unattempted cases, rather than an established token-context limit.
- **Nanbeige:** all 80 measured outputs lack a final answer within the 500-token budget. Its zero does not establish a general semantic inability.
- **Local hardware:** transformer/output layers and KV are on CUDA, but llama.cpp b11960 retains CPU_Mapped input/embedding weights and CPU preparation. This profile does not demonstrate zero CPU participation. See [hardware scope](evidence/NATIVE_GPU_SCOPE.md).
- **Latency:** local and OAuth HTTP timings include different work and output lengths. No hardware-speed ratio is claimed.
- **Native-v2 definition:** this corrects reasoning parsing; the biased scientific v1 protocol is unchanged. The scientific v2 proposed in the OAuth report was not executed.
- **External comparability:** agreement with the reference cloud runner remains unverified; duplicate names and lexical query hints remain.

[Local report](LOCAL_REPORT.md) · [OAuth report](../skill-discovery-oauth-v1-20261103/REPORT.md) · [Offline validation](validation/public-validation.json) · [Shared-panel summary](validation/common-panel-summary.csv)

CSV bytes are unchanged. Distributed databases are nonresumable audit exports. Full private HTTP outputs are not redistributed, so public validation recalculates ranking metrics, checks local CSV/SQLite consistency and reproduces OAuth CSV summaries without replaying every normalization from raw text. Transformation records and hashes describe privacy adaptations.

## Final Qwen8/Mistral supplement: separate v3 series

The two models produced 200 additional responses: 80 f16 responses per model at N=50/100/250/500, then 20 Q8_0 responses per model at N=500. Context is sized to actual input tokens +500, without the initial estimated 10% margin. Q4_K_M weights are unchanged; Q8_0 refers to the KV cache. Historical 32K/f16 blocks remain preserved.

| Model | Profile | N | n_ctx | KV | Top-1 /20 | Top-10 /20 | Without errors | Median LLM (s) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n100 | 100 | 5888 | f16 | 9 | 15 | 20 | 11.550 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n250 | 250 | 14080 | f16 | 8 | 12 | 19 | 20.081 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n50 | 50 | 3328 | f16 | 16 | 20 | 20 | 5.854 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n500 | 500 | 27392 | f16 | 3 | 10 | 19 | 37.395 |
| mistral-7b-instruct-v03-q4km | q8-adjusted-n500 | 500 | 27392 | q8_0 | 3 | 8 | 18 | 46.022 |
| qwen3-8b-q4km | f16-adjusted-n100 | 100 | 5120 | f16 | 8 | 8 | 8 | 11.937 |
| qwen3-8b-q4km | f16-adjusted-n250 | 250 | 12032 | f16 | 8 | 8 | 8 | 19.557 |
| qwen3-8b-q4km | f16-adjusted-n50 | 50 | 2816 | f16 | 10 | 10 | 10 | 9.938 |
| qwen3-8b-q4km | f16-adjusted-n500 | 500 | 23552 | f16 | 8 | 9 | 10 | 34.465 |
| qwen3-8b-q4km | q8-adjusted-n500 | 500 | 23552 | q8_0 | 10 | 10 | 11 | 42.221 |

| Paired comparison at N=500 | Top-1 f16 → Q8 | Top-10 f16 → Q8 | Only f16 correct | Only Q8 correct |
| --- | --- | --- | ---: | ---: |
| qwen3-8b-q4km | 8 → 10 | 9 → 10 | 1 | 3 |
| mistral-7b-instruct-v03-q4km | 3 → 3 | 10 → 8 | 0 | 0 |

Across the four f16 sizes, Qwen8 scores 34/80 (42.5%) and Mistral 36/80 (45.0%). Q8 results remain separate. The 20 paired questions establish neither a general improvement nor quantization equivalence. No N≥1000 case was measured in this supplement.

In total, 16 distinct local models now have measurements in at least one series: 1,120 responses in the initial series plus 200 in this supplement, or 1,320. Three OAuth models add 360 responses, giving 1,680 measured responses across profiles; these are not independent observations. Llama 3.2 and Gemma 3 were not run before the study was closed, as requested.

[Detailed v3 report](profiles-v3/PROFILE_REPORT.md) · [Validation](profiles-v3/validation/public-validation.json) · [Llama/Gemma closure](profiles-v3/CLOSURE.md)

