# Skill discovery — local results with the corrected native adapter

Status verified at 2026-11-03T09:52:39.825472+00:00. Suite: `completed`.

Of the 18 requested models, 14 produced measured responses. The series contains 2520 rows with 45 columns and 1120 responses with actual usage. Blocks and exclusions remain separate from measurements.

The comparative series is `results/local-native-v2/`. The earlier `local-v1/` series was preserved after four models as a diagnostic: its adapter could count internal reasoning as ranked answers. Its scores are not the final comparative results. See `evidence/NATIVE_PARSER_V2.md`.

## Model coverage

| Requested model | Status | Saved rows | Measured responses | Strict top-1 / measured responses | Top-10 / measured responses | Outputs without errors |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | completed | 140 | 80 | 49/80 (61.25%) | 51/80 (63.75%) | 57 |
| Qwen3-1.7B | completed | 140 | 80 | 19/80 (23.75%) | 19/80 (23.75%) | 20 |
| Qwen3.5-4B | completed | 140 | 80 | 28/80 (35.00%) | 30/80 (37.50%) | 32 |
| Ling-3.0-tiny | completed | 140 | 80 | 38/80 (47.50%) | 40/80 (50.00%) | 40 |
| Qwen3.5-9B | completed | 140 | 80 | 25/80 (31.25%) | 25/80 (31.25%) | 27 |
| Llama 3.2 3B | blocked: ACCESS_GATED_ACCEPTANCE_REQUIRED | 140 | 0 | - | - | 0 |
| Gemma 3 4B | blocked: ACCESS_GATED_ACCEPTANCE_REQUIRED | 140 | 0 | - | - | 0 |
| H2O-Lightning-4B | completed | 140 | 80 | 47/80 (58.75%) | 49/80 (61.25%) | 50 |
| Gemma 4 E2B | completed | 140 | 80 | 34/80 (42.50%) | 34/80 (42.50%) | 34 |
| Qwen3-8B | blocked: HARDWARE_VRAM_CAPACITY_EXCEEDED | 140 | 0 | - | - | 0 |
| NeoHorse-1-9B | completed | 140 | 80 | 45/80 (56.25%) | 45/80 (56.25%) | 46 |
| Ornith-1.5-9B | completed | 140 | 80 | 50/80 (62.50%) | 54/80 (67.50%) | 57 |
| Nemotron 3.5 Bolt E4B | completed | 140 | 80 | 21/80 (26.25%) | 21/80 (26.25%) | 21 |
| Ternary Bonsai 2 9B | completed | 140 | 80 | 23/80 (28.75%) | 25/80 (31.25%) | 25 |
| LFM3-Omni-2B | completed | 140 | 80 | 57/80 (71.25%) | 61/80 (76.25%) | 75 |
| Phi-4 Mini 3.8B | completed | 140 | 80 | 52/80 (65.00%) | 53/80 (66.25%) | 58 |
| Mistral 7B Instruct | blocked: HARDWARE_VRAM_CAPACITY_EXCEEDED | 140 | 0 | - | - | 0 |
| Nanbeige 4.2 3B | completed | 140 | 80 | 0/80 (0.00%) | 0/80 (0.00%) | 0 |

A measured response has actual input and output token usage. OOM, gated access, context exclusions and unattempted cases have no measured accuracy; they are not observed zero-accuracy results. A complete 140-row export does not imply 140 successful calls.

## Details by library size

| Model | N | Measured responses | Strict top-1 | Top-10 recall | Top-1 precision, non-error outputs | Outputs reaching the 500-token cap | Actual input tokens, min–max | Median LLM call (s) | Recorded issues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gemma-3-4b-it | 50 | 0 | - | - | - | 0 | - | - | ACCESS_GATED_ACCEPTANCE_REQUIRED=20 |
| gemma-3-4b-it | 100 | 0 | - | - | - | 0 | - | - | ACCESS_GATED_ACCEPTANCE_REQUIRED=20 |
| gemma-3-4b-it | 250 | 0 | - | - | - | 0 | - | - | ACCESS_GATED_ACCEPTANCE_REQUIRED=20 |
| gemma-3-4b-it | 500 | 0 | - | - | - | 0 | - | - | ACCESS_GATED_ACCEPTANCE_REQUIRED=20 |
| gemma-3-4b-it | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| gemma-3-4b-it | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| gemma-3-4b-it | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| gemma-4-e2b-it-q4 | 50 | 20 | 50.00% | 50.00% | 100.00% | 9 | 2247-2298 | 4.98 | REASONING_NO_FINAL_ANSWER=9, OK=10, INVALID_OUTPUT=1 |
| gemma-4-e2b-it-q4 | 100 | 20 | 45.00% | 45.00% | 100.00% | 11 | 4480-4531 | 5.88 | REASONING_NO_FINAL_ANSWER=11, OK=9 |
| gemma-4-e2b-it-q4 | 250 | 20 | 40.00% | 40.00% | 100.00% | 12 | 11787-11838 | 8.59 | REASONING_NO_FINAL_ANSWER=12, OK=8 |
| gemma-4-e2b-it-q4 | 500 | 20 | 35.00% | 35.00% | 100.00% | 12 | 23499-23550 | 13.93 | REASONING_NO_FINAL_ANSWER=12, OK=7, INVALID_OUTPUT=1 |
| gemma-4-e2b-it-q4 | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| gemma-4-e2b-it-q4 | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| gemma-4-e2b-it-q4 | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| h2o-lightning-4b-stock-q4km | 50 | 20 | 45.00% | 45.00% | 100.00% | 11 | 2239-2292 | 7.10 | REASONING_NO_FINAL_ANSWER=7, OK=9, MALFORMED_REASONING_BOUNDARY=2, INVALID_OUTPUT=2 |
| h2o-lightning-4b-stock-q4km | 100 | 20 | 60.00% | 70.00% | 85.71% | 9 | 4415-4468 | 6.70 | REASONING_NO_FINAL_ANSWER=5, OK=14, MALFORMED_REASONING_BOUNDARY=1 |
| h2o-lightning-4b-stock-q4km | 250 | 20 | 65.00% | 65.00% | 100.00% | 10 | 11633-11686 | 10.94 | REASONING_NO_FINAL_ANSWER=6, OK=13, MALFORMED_REASONING_BOUNDARY=1 |
| h2o-lightning-4b-stock-q4km | 500 | 20 | 65.00% | 65.00% | 92.86% | 11 | 23177-23230 | 20.71 | REASONING_NO_FINAL_ANSWER=6, OK=14 |
| h2o-lightning-4b-stock-q4km | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| h2o-lightning-4b-stock-q4km | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| h2o-lightning-4b-stock-q4km | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| lfm3-omni-2b-q4km | 50 | 20 | 80.00% | 80.00% | 84.21% | 0 | 2251-2306 | 0.59 | OK=19, INVALID_OUTPUT=1 |
| lfm3-omni-2b-q4km | 100 | 20 | 75.00% | 80.00% | 75.00% | 0 | 4457-4512 | 1.10 | OK=20 |
| lfm3-omni-2b-q4km | 250 | 20 | 70.00% | 75.00% | 73.68% | 0 | 11651-11706 | 2.96 | OK=19, INVALID_OUTPUT=1 |
| lfm3-omni-2b-q4km | 500 | 20 | 60.00% | 70.00% | 70.59% | 0 | 23137-23192 | 5.96 | OK=17, INVALID_OUTPUT=3 |
| lfm3-omni-2b-q4km | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| lfm3-omni-2b-q4km | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| lfm3-omni-2b-q4km | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| ling-3.0-tiny-q4km | 50 | 20 | 60.00% | 65.00% | 92.31% | 8 | 2245-2296 | 3.31 | OK=13, REASONING_NO_FINAL_ANSWER=7 |
| ling-3.0-tiny-q4km | 100 | 20 | 45.00% | 50.00% | 90.00% | 10 | 4446-4497 | 6.47 | OK=10, REASONING_NO_FINAL_ANSWER=10 |
| ling-3.0-tiny-q4km | 250 | 20 | 40.00% | 40.00% | 100.00% | 14 | 11706-11757 | 11.89 | OK=8, REASONING_NO_FINAL_ANSWER=12 |
| ling-3.0-tiny-q4km | 500 | 20 | 45.00% | 45.00% | 100.00% | 10 | 23299-23350 | 21.55 | OK=9, REASONING_NO_FINAL_ANSWER=10, INVALID_OUTPUT=1 |
| ling-3.0-tiny-q4km | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| ling-3.0-tiny-q4km | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| ling-3.0-tiny-q4km | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| llama-3.2-3b-instruct | 50 | 0 | - | - | - | 0 | - | - | ACCESS_GATED_ACCEPTANCE_REQUIRED=20 |
| llama-3.2-3b-instruct | 100 | 0 | - | - | - | 0 | - | - | ACCESS_GATED_ACCEPTANCE_REQUIRED=20 |
| llama-3.2-3b-instruct | 250 | 0 | - | - | - | 0 | - | - | ACCESS_GATED_ACCEPTANCE_REQUIRED=20 |
| llama-3.2-3b-instruct | 500 | 0 | - | - | - | 0 | - | - | ACCESS_GATED_ACCEPTANCE_REQUIRED=20 |
| llama-3.2-3b-instruct | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| llama-3.2-3b-instruct | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| llama-3.2-3b-instruct | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| mistral-7b-instruct-v03-q4km | 50 | 0 | - | - | - | 0 | - | - | HARDWARE_VRAM_CAPACITY_EXCEEDED=20 |
| mistral-7b-instruct-v03-q4km | 100 | 0 | - | - | - | 0 | - | - | HARDWARE_VRAM_CAPACITY_EXCEEDED=20 |
| mistral-7b-instruct-v03-q4km | 250 | 0 | - | - | - | 0 | - | - | HARDWARE_VRAM_CAPACITY_EXCEEDED=20 |
| mistral-7b-instruct-v03-q4km | 500 | 0 | - | - | - | 0 | - | - | HARDWARE_VRAM_CAPACITY_EXCEEDED=20 |
| mistral-7b-instruct-v03-q4km | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| mistral-7b-instruct-v03-q4km | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| mistral-7b-instruct-v03-q4km | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| nanbeige4.2-3b-q4km | 50 | 20 | 0.00% | 0.00% | - | 20 | 2275-2326 | 17.30 | REASONING_NO_FINAL_ANSWER=20 |
| nanbeige4.2-3b-q4km | 100 | 20 | 0.00% | 0.00% | - | 20 | 4519-4570 | 27.21 | REASONING_NO_FINAL_ANSWER=20 |
| nanbeige4.2-3b-q4km | 250 | 20 | 0.00% | 0.00% | - | 20 | 11910-11961 | 61.31 | REASONING_NO_FINAL_ANSWER=20 |
| nanbeige4.2-3b-q4km | 500 | 20 | 0.00% | 0.00% | - | 20 | 23750-23801 | 122.00 | REASONING_NO_FINAL_ANSWER=20 |
| nanbeige4.2-3b-q4km | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| nanbeige4.2-3b-q4km | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| nanbeige4.2-3b-q4km | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| nemotron-3.5-bolt-e4b-q4km | 50 | 20 | 30.00% | 30.00% | 100.00% | 14 | 2247-2298 | 8.23 | OK=6, REASONING_NO_FINAL_ANSWER=14 |
| nemotron-3.5-bolt-e4b-q4km | 100 | 20 | 45.00% | 45.00% | 100.00% | 12 | 4480-4531 | 9.57 | OK=9, REASONING_NO_FINAL_ANSWER=11 |
| nemotron-3.5-bolt-e4b-q4km | 250 | 20 | 20.00% | 20.00% | 100.00% | 16 | 11787-11838 | 14.30 | REASONING_NO_FINAL_ANSWER=16, OK=4 |
| nemotron-3.5-bolt-e4b-q4km | 500 | 20 | 10.00% | 10.00% | 100.00% | 18 | 23499-23550 | 22.73 | REASONING_NO_FINAL_ANSWER=18, OK=2 |
| nemotron-3.5-bolt-e4b-q4km | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| nemotron-3.5-bolt-e4b-q4km | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| nemotron-3.5-bolt-e4b-q4km | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| neohorse-1-9b-q4km | 50 | 20 | 65.00% | 65.00% | 100.00% | 7 | 2239-2292 | 6.43 | OK=13, REASONING_NO_FINAL_ANSWER=6, INVALID_OUTPUT=1 |
| neohorse-1-9b-q4km | 100 | 20 | 55.00% | 55.00% | 100.00% | 9 | 4415-4468 | 7.54 | REASONING_NO_FINAL_ANSWER=9, OK=11 |
| neohorse-1-9b-q4km | 250 | 20 | 60.00% | 60.00% | 100.00% | 8 | 11633-11686 | 16.46 | REASONING_NO_FINAL_ANSWER=8, OK=12 |
| neohorse-1-9b-q4km | 500 | 20 | 45.00% | 45.00% | 90.00% | 11 | 23177-23230 | 29.74 | REASONING_NO_FINAL_ANSWER=10, OK=10 |
| neohorse-1-9b-q4km | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| neohorse-1-9b-q4km | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| neohorse-1-9b-q4km | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| ornith-1.5-9b-q4km | 50 | 20 | 75.00% | 85.00% | 88.24% | 3 | 2239-2292 | 7.69 | OK=17, REASONING_NO_FINAL_ANSWER=2, INVALID_OUTPUT=1 |
| ornith-1.5-9b-q4km | 100 | 20 | 70.00% | 75.00% | 87.50% | 5 | 4415-4468 | 9.39 | REASONING_NO_FINAL_ANSWER=4, OK=16 |
| ornith-1.5-9b-q4km | 250 | 20 | 55.00% | 55.00% | 100.00% | 10 | 11633-11686 | 18.63 | OK=11, REASONING_NO_FINAL_ANSWER=9 |
| ornith-1.5-9b-q4km | 500 | 20 | 50.00% | 55.00% | 76.92% | 9 | 23177-23230 | 29.49 | OK=13, REASONING_NO_FINAL_ANSWER=7 |
| ornith-1.5-9b-q4km | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| ornith-1.5-9b-q4km | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| ornith-1.5-9b-q4km | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| phi-4-mini-instruct-q4km | 50 | 20 | 85.00% | 85.00% | 85.00% | 1 | 2127-2180 | 1.56 | OK=20 |
| phi-4-mini-instruct-q4km | 100 | 20 | 65.00% | 65.00% | 86.67% | 0 | 4221-4274 | 2.33 | OK=15, INVALID_OUTPUT=5 |
| phi-4-mini-instruct-q4km | 250 | 20 | 55.00% | 60.00% | 91.67% | 2 | 11019-11072 | 6.28 | OK=12, INVALID_OUTPUT=8 |
| phi-4-mini-instruct-q4km | 500 | 20 | 55.00% | 55.00% | 100.00% | 2 | 21852-21905 | 14.51 | OK=11, INVALID_OUTPUT=9 |
| phi-4-mini-instruct-q4km | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| phi-4-mini-instruct-q4km | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| phi-4-mini-instruct-q4km | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| qwen3-0.6b-q8 | 50 | 20 | 70.00% | 75.00% | 93.33% | 6 | 2199-2252 | 1.92 | OK=15, REASONING_NO_FINAL_ANSWER=4, INVALID_OUTPUT=1 |
| qwen3-0.6b-q8 | 100 | 20 | 60.00% | 60.00% | 92.31% | 7 | 4334-4387 | 2.36 | OK=13, REASONING_NO_FINAL_ANSWER=6, INVALID_OUTPUT=1 |
| qwen3-0.6b-q8 | 250 | 20 | 50.00% | 55.00% | 76.92% | 6 | 11473-11526 | 4.94 | REASONING_NO_FINAL_ANSWER=6, OK=13, INVALID_OUTPUT=1 |
| qwen3-0.6b-q8 | 500 | 20 | 65.00% | 65.00% | 81.25% | 4 | 22878-22931 | 9.48 | OK=16, REASONING_NO_FINAL_ANSWER=2, INVALID_OUTPUT=2 |
| qwen3-0.6b-q8 | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| qwen3-0.6b-q8 | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| qwen3-0.6b-q8 | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| qwen3-1.7b-q8 | 50 | 20 | 20.00% | 20.00% | 100.00% | 16 | 2199-2252 | 4.50 | REASONING_NO_FINAL_ANSWER=16, OK=4 |
| qwen3-1.7b-q8 | 100 | 20 | 25.00% | 25.00% | 100.00% | 15 | 4334-4387 | 5.27 | REASONING_NO_FINAL_ANSWER=15, OK=5 |
| qwen3-1.7b-q8 | 250 | 20 | 30.00% | 30.00% | 85.71% | 14 | 11473-11526 | 8.31 | REASONING_NO_FINAL_ANSWER=13, OK=7 |
| qwen3-1.7b-q8 | 500 | 20 | 20.00% | 20.00% | 100.00% | 16 | 22878-22931 | 14.32 | REASONING_NO_FINAL_ANSWER=15, OK=4, INVALID_OUTPUT=1 |
| qwen3-1.7b-q8 | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| qwen3-1.7b-q8 | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| qwen3-1.7b-q8 | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| qwen3-8b-q4km | 50 | 0 | - | - | - | 0 | - | - | HARDWARE_VRAM_CAPACITY_EXCEEDED=20 |
| qwen3-8b-q4km | 100 | 0 | - | - | - | 0 | - | - | HARDWARE_VRAM_CAPACITY_EXCEEDED=20 |
| qwen3-8b-q4km | 250 | 0 | - | - | - | 0 | - | - | HARDWARE_VRAM_CAPACITY_EXCEEDED=20 |
| qwen3-8b-q4km | 500 | 0 | - | - | - | 0 | - | - | HARDWARE_VRAM_CAPACITY_EXCEEDED=20 |
| qwen3-8b-q4km | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| qwen3-8b-q4km | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| qwen3-8b-q4km | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| qwen3.5-4b-q4km | 50 | 20 | 5.00% | 5.00% | 100.00% | 19 | 2239-2292 | 7.08 | REASONING_NO_FINAL_ANSWER=19, OK=1 |
| qwen3.5-4b-q4km | 100 | 20 | 50.00% | 55.00% | 83.33% | 14 | 4415-4468 | 8.27 | OK=12, REASONING_NO_FINAL_ANSWER=8 |
| qwen3.5-4b-q4km | 250 | 20 | 40.00% | 40.00% | 100.00% | 16 | 11633-11686 | 12.70 | OK=8, REASONING_NO_FINAL_ANSWER=12 |
| qwen3.5-4b-q4km | 500 | 20 | 45.00% | 50.00% | 81.82% | 13 | 23177-23230 | 20.62 | OK=11, REASONING_NO_FINAL_ANSWER=8, INVALID_OUTPUT=1 |
| qwen3.5-4b-q4km | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| qwen3.5-4b-q4km | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| qwen3.5-4b-q4km | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| qwen3.5-9b-q4km | 50 | 20 | 35.00% | 35.00% | 100.00% | 14 | 2239-2292 | 10.85 | REASONING_NO_FINAL_ANSWER=13, OK=7 |
| qwen3.5-9b-q4km | 100 | 20 | 35.00% | 35.00% | 100.00% | 14 | 4415-4468 | 12.61 | REASONING_NO_FINAL_ANSWER=13, OK=7 |
| qwen3.5-9b-q4km | 250 | 20 | 20.00% | 20.00% | 100.00% | 16 | 11633-11686 | 18.88 | REASONING_NO_FINAL_ANSWER=16, OK=4 |
| qwen3.5-9b-q4km | 500 | 20 | 35.00% | 35.00% | 77.78% | 16 | 23177-23230 | 29.81 | OK=9, REASONING_NO_FINAL_ANSWER=11 |
| qwen3.5-9b-q4km | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| qwen3.5-9b-q4km | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| qwen3.5-9b-q4km | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| ternary-bonsai-2-9b-q2 | 50 | 20 | 0.00% | 0.00% | - | 20 | 2239-2292 | 8.86 | REASONING_NO_FINAL_ANSWER=20 |
| ternary-bonsai-2-9b-q2 | 100 | 20 | 30.00% | 40.00% | 75.00% | 15 | 4415-4468 | 10.51 | OK=8, REASONING_NO_FINAL_ANSWER=12 |
| ternary-bonsai-2-9b-q2 | 250 | 20 | 40.00% | 40.00% | 100.00% | 13 | 11633-11686 | 16.28 | OK=8, REASONING_NO_FINAL_ANSWER=12 |
| ternary-bonsai-2-9b-q2 | 500 | 20 | 45.00% | 45.00% | 100.00% | 12 | 23177-23230 | 26.57 | INVALID_OUTPUT=1, OK=9, REASONING_NO_FINAL_ANSWER=10 |
| ternary-bonsai-2-9b-q2 | 1000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| ternary-bonsai-2-9b-q2 | 2000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |
| ternary-bonsai-2-9b-q2 | 5000 | 0 | - | - | - | 0 | - | - | CONTEXT_LIMIT_ESTIMATED=20 |

Rates over measured outputs include missing final lists and formatting failures. Precision conditional on non-error outputs uses another denominator. Errors and invalid final-answer ranks are never removed to promote a correct answer.

## Shared conditions and limitations

- 6,006 skills, 5,801 distinct names and the 20 exact questions; sizes 50, 100, 250, 500, 1,000, 2,000 and 5,000; seed 42; top 10. No full bodies: names and descriptions capped at 300 characters.
- `random.Random(42).sample(all_skills,N)` is restarted at each size. Each question gets a copy, with last-item replacement if and only if its gold ID is missing. **Verified intrinsic bias: the 20 supplied gold IDs are already the first 20 seed-42 draws; their positions remain 1–20 at every size.** The runner adds no gold prefix. This is not a test of uniformly distributed answer positions.
- All 140 v1/native-v2 candidate plans and prompt hashes are identical. The corrected adapter only adds parsing state for reasoning already prefilled by the template; prompts and thinking mode are unchanged. Delimited internal blocks are excluded from ranks. An unfinished reasoning block at the cap produces `REASONING_NO_FINAL_ANSWER` while retaining actual usage.
- RTX 2060 SUPER, 8,192 MiB; driver 610.88; llama.cpp b11960 CUDA 12.4. Transformer/output layers, f16 KV and observed recurrent state are on CUDA. Effective context 32,768, one slot, fit=off, no context shift, warmup, RAM cache or checkpoints; batch 512 / microbatch 128. **Exact scope: llama.cpp retains default CPU_Mapped input/embedding weights**, even with all transformer/output layers on GPU. This profile is not CPU-free computation. Transformer-layer fallback to fit VRAM is rejected. The buffer audit records these allocations. Preparatory conversion and quantization also use CPU.
- `/apply-template` then `/tokenize` provide actual counts; IDs are sent to `/completion`. Temperature 0, seed 42, 500-token output cap, no truncation. The estimated v1 gate `44.5*N+200 > 90% of context` excludes 60 rows per model at sizes 1,000/2,000/5,000 (1,080 rows across all 18 models); those are not measured input-token counts.
- **The 60 excluded rows per model reflect the chosen n_ctx filter, not intrinsic model limits.** Qwen3.5 advertises 262,144 native tokens (an extension to 1,010,000); Ling configures 131,072 native tokens and a YaRN recipe for 262,144 tokens. This series loads only 32,768, a uniform conservative choice without per-model maximum-VRAM search. No tokenizer request or inference measures excluded sizes. Published windows, publisher configs, GGUF values and observed n_ctx are separated in the context-window audit and notes.
- Reasoning models can spend all 500 tokens before the final list. Budget/format failures describe this configuration, not their best possible ability. Quantization differs and is identified per model.
- Multimodal models are tested with text only. H2O uses its official ordinary-chat export without a typed-decision shim. Instruct/IT versions are used when available; Mistral is explicitly v0.3.
- The 20 questions form a small, partly lexical sample; duplicate names prevent recovery of some gold IDs through name-only responses. The original reference cloud runner was not supplied.
- OAuth Luna/Sol/Terra uses the same candidates, but its route guarantees neither temperature 0 nor a 500-token output cap. It remains a separate variant; strict cross-route equivalence is not claimed.
- Latencies describe this Windows environment with 16 GB RAM. Downloads and CPU conversions could run concurrently, so this is not an isolated speed benchmark. Local service charges are USD 0, excluding electricity.

## Weights and provenance

| Model | Quantization | Verified source |
|---|---|---|
| Qwen3-0.6B | Q8_0 | [Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) |
| Qwen3-1.7B | Q8_0 | [Qwen/Qwen3-1.7B-GGUF](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF) |
| Qwen3.5-4B | Q4_K_M; installed Ollama artifact, digest confirmed against the official registry | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Ling-3.0-tiny | Q4_K_M | [inclusionAI/Ling-3.0-tiny-GGUF](https://huggingface.co/inclusionAI/Ling-3.0-tiny-GGUF) |
| Qwen3.5-9B | Q4_K_M; installed Ollama artifact, digest confirmed against the official registry | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Llama 3.2 3B | not downloaded: gated access | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Gemma 3 4B | not downloaded: gated access | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) |
| H2O-Lightning-4B | Q4_K_M, official stock ordinary-chat export | [h2oai/h2o-lightning-4b-gguf](https://huggingface.co/h2oai/h2o-lightning-4b-gguf) |
| Gemma 4 E2B | Official QAT Q4_0 | [google/gemma-4-E2B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf) |
| Qwen3-8B | Q4_K_M | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) |
| NeoHorse-1-9B | Q4_K_M | [TokenRhythm/NeoHorse-1-9B-GGUF](https://huggingface.co/TokenRhythm/NeoHorse-1-9B-GGUF) |
| Ornith-1.5-9B | Q4_K_M | [ornith-ai/Ornith-1.5-9B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF) |
| Nemotron 3.5 Bolt E4B | Q4_K_M | [nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB) |
| Ternary Bonsai 2 9B | Official ternary Q2_0 packing, not Q4 | [prism-ml/Ternary-Bonsai-2-9B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf) |
| LFM3-Omni-2B | Q4_K_M | [LiquidAI/LFM3-Omni-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF) |
| Phi-4 Mini 3.8B | Q4_K_M, local conversion of pinned official weights | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) |
| Mistral 7B Instruct | Q4_K_M, local conversion of pinned official weights | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| Nanbeige 4.2 3B | Q4_K_M, local conversion of pinned official weights | [Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) |

Commits, hashes, sizes and licenses are recorded in source inventories, verified Ollama manifests and weight provenance files. Installed Ling matches the official GGUF. Llama 3.2 3B and Gemma 3 4B require manual gated-access acceptance; no bypass or ungated substitute was used.

## Evidence, validation and delivery

All 34 synthetic tests pass, including seven native-parser regressions. `evidence/result-review-native-v2.json` checks the 45 columns, unique keys, cardinalities, actual usage, SQLite integrity and CSV/SQLite equality, and independently recalculates rank metrics. SQLite is the transactional authority; CSV is replaced atomically after each row.

Each model retains configuration, manifest and source-linked launch/GPU evidence. Initial failures are preserved. Raw outputs are held in ignored private response captures, without full prompts; no OAuth credential was copied into this local series. Public exports remove process identifiers and private paths while retaining audit hashes.

The benchmark/report generators do not send email or push code. Weights, archives, tools, environments, private full input files and raw captures are excluded. Public integration and the consolidated Markdown report are handled separately.

## Physical-VRAM admission

Under Windows WDDM, CUDA allocation can succeed despite a declared budget exceeding physical VRAM. The launcher therefore checks the sum of model, KV, recurrent-state and maximum compute CUDA buffers before inference. Excess produces `HARDWARE_VRAM_CAPACITY_EXCEEDED`, distinct from an observed CUDA OOM. It does not silently offload layers or change context.

Qwen3-8B at 32K declares 9,097.34 MiB against 8,192 MiB physical VRAM. The 21 responses acquired before this control remain in `results/diagnostic-vram-oversubscription/`; the next pending inference transaction (`100:1`) has an uncertain outcome and was not retried. They are excluded from the comparative series; its canonical ledger records hardware outcomes without measured accuracy. Corrected guard validation rejected the same load without inference. Exact launch hashes and allocations are audited in the native-buffer evidence.

Mistral-7B-Instruct-v0.3 is likewise rejected before generation: 8,230.52 MiB declared against 8,192 MiB physical VRAM. It is not assigned an observed zero; measured accuracy is absent. Converted originals and startup allocations are preserved.

Starting with Ornith, historical WDDM telemetry and its summary record verified-process dedicated/shared counters. Nonzero shared usage can include host allocations or driver staging; alone it proves neither transformer offload nor paging of a specific CUDA buffer. This sampled collection cannot reconstruct residency for earlier models.

Consolidated public provenance, including local conversions and installed Ollama artifacts, is in `evidence/model-artifact-provenance.json`. It remains available when ignored weight directories are not transferred.
