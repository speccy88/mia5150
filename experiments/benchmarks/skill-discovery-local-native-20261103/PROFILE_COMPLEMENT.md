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
