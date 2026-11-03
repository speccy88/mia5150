# Skill Discovery — résultats locaux, adaptateur natif corrigé

État vérifié au 2026-11-03T09:52:39.825472+00:00. Lot : `completed`.

Sur les 18 modèles demandés, 14 ont fourni des réponses mesurées. Le lot contient 2520 lignes de 45 colonnes et 1120 réponses avec usage réel. Les blocages et exclusions restent séparés des mesures.

Le lot comparatif courant est `results/local-native-v2/`. Le lot `local-v1/` est conservé après quatre modèles comme diagnostic : son adaptateur pouvait compter du raisonnement interne dans les rangs. Les scores v1 ne sont pas les résultats comparatifs finaux. Voir `evidence/NATIVE_PARSER_V2.md`.

## Couverture des modèles

| Modèle demandé | État | Lignes sauvegardées | Réponses mesurées | Top-1 strict / réponses mesurées | Top-10 / réponses mesurées | Sorties sans erreur |
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

Une réponse mesurée possède des tokens réels d'entrée et de sortie. Les OOM, les accès gated, les limites de contexte et les essais non exécutés restent sans taux mesuré; ils ne sont pas interprétés comme des zéros de précision. Un export complet de 140 lignes ne signifie pas 140 appels réussis.

## Détail par taille

| Modèle | N | Réponses mesurées | Top-1 strict | Rappel Top-10 | Précision Top-1, sorties sans erreur | Sorties au plafond 500 | Tokens entrée réels, min-max | Médiane appel LLM (s) | Issues enregistrées |
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

Les taux sur réponses mesurées incluent les réponses sans liste finale ou de mauvais format. La précision sur sorties sans erreur utilise un autre dénominateur. Les erreurs et rangs invalides de la réponse finale ne sont jamais retirés pour promouvoir une bonne réponse.

## Conditions communes et limites

- 6 006 skills, 5 801 noms distincts, 20 questions exactes; tailles 50, 100, 250, 500, 1 000, 2 000 et 5 000; seed 42; top 10. Aucun corps complet : noms et descriptions limitées à 300 caractères.
- `random.Random(42).sample(all_skills,N)` est réinitialisé pour chaque taille. Copie par question; remplacement du dernier élément si et seulement si le gold ID manque. **Biais intrinsèque vérifié : les 20 gold IDs fournis sont déjà les 20 premiers tirages du seed 42; leurs positions restent 1–20 à toutes les tailles.** Aucun préfixe gold n’est ajouté par le runner. Ce test ne mesure pas des positions uniformément réparties.
- Les 140 plans de candidats et hashes de prompts v1/native-v2 sont identiques. L’adaptateur corrigé ajoute uniquement un état de parsing pour le raisonnement déjà prérempli par le template; il ne change pas le prompt ni le mode thinking. Les blocs internes délimités sont exclus des rangs. Une pensée inachevée au plafond donne `REASONING_NO_FINAL_ANSWER`, avec les tokens réels conservés.
- RTX 2060 Super 8 192 MiB; pilote 610.88; llama.cpp b11960 CUDA 12.4. Les couches transformer et de sortie, le KV f16 et les états récurrents observés sont sur CUDA; contexte effectif 32 768, un slot, `fit=off`, sans context shift, warmup, cache RAM ou checkpoints; batch 512 / microbatch 128. **Portée précise : llama.cpp garde l’entrée/les embeddings dans un buffer CPU_Mapped par défaut**, même avec toutes les couches transformer/sortie sur GPU. Ce profil n’est donc pas un calcul sans participation CPU. Aucun repli de couches transformer pour tenir en VRAM n’est accepté. Les journaux détaillent ces buffers dans `evidence/native-buffer-audit.json`. La conversion/quantification préparatoire emploie aussi le CPU.
- `/apply-template` puis `/tokenize` fournissent le comptage réel; les IDs sont transmis à `/completion`. Température 0, seed 42, sortie plafonnée à 500 tokens. Aucune troncature. La barrière v1 estimée `44.5*N+200 > 90% du contexte` bloque 60 lignes aux tailles 1 000/2 000/5 000 à ce contexte; ce n’est pas un nombre de tokens mesuré.
- **Ces 60 lignes sont un filtre du n_ctx choisi, pas la limite intrinsèque des modèles.** Qwen3.5 publie 262 144 tokens natifs (extension annoncée à 1 010 000); Ling configure 131 072 natifs avec une recette YaRN à 262 144. Le lot charge seulement 32 768, choix uniforme et conservateur sans recherche du maximum VRAM par modèle. Aucun tokenizer/appel ne mesure les tailles écartées. Fenêtres publiées, config éditeur, GGUF et n_ctx observé sont séparés dans `evidence/context-window-audit.json` et `evidence/CONTEXT_WINDOWS.md`.
- Les modèles reasoning peuvent consommer les 500 tokens avant leur liste finale. Ces erreurs de budget/format décrivent cette configuration, pas leur meilleure capacité possible. Les quantifications sont différentes et indiquées par modèle.
- Le multimodal est testé en texte seul. H2O utilise son export officiel stock de chat ordinaire, sans shim de décisions typées. Les versions Instruct/IT sont utilisées quand disponibles; Mistral est explicitement v0.3.
- Les 20 questions forment un petit échantillon en partie lexical; les noms dupliqués empêchent de récupérer certains gold IDs avec une réponse par noms. Le runner original de Vincent n’a pas été fourni à ce lot.
- Le lot OAuth Luna/Sol/Terra utilise les mêmes candidats, mais sa route ne garantit ni température 0 ni sortie plafonnée à 500; il doit rester une variante distincte. Aucune équivalence stricte entre routes n’est revendiquée.
- Les latences sont des mesures dans cet environnement Windows 16 Go de RAM. Téléchargements/conversions CPU ont pu tourner simultanément : ce n’est pas un benchmark de vitesse isolé. Coût de service local 0 USD, hors électricité.

## Poids et provenance

| Modèle | Quantification | Source vérifiée |
|---|---|---|
| Qwen3-0.6B | Q8_0 | [Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) |
| Qwen3-1.7B | Q8_0 | [Qwen/Qwen3-1.7B-GGUF](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF) |
| Qwen3.5-4B | Q4_K_M; fichier Ollama installé, digest confirmé par le registre officiel | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Ling-3.0-tiny | Q4_K_M | [inclusionAI/Ling-3.0-tiny-GGUF](https://huggingface.co/inclusionAI/Ling-3.0-tiny-GGUF) |
| Qwen3.5-9B | Q4_K_M; fichier Ollama installé, digest confirmé par le registre officiel | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Llama 3.2 3B | non téléchargé : accès gated | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Gemma 3 4B | non téléchargé : accès gated | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) |
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

Commits, SHA-256, tailles et licences : `evidence/source-inventory.json`, manifests Ollama vérifiés et fichiers `.source.json` des poids. Ling installé est identique au GGUF officiel. Llama 3.2 3B et Gemma 3 4B demandent une acceptation d’accès manuelle; aucun contournement ni substitut non gated n’a été utilisé.

## Preuves, validation et livraison

34 tests synthétiques passent, dont sept régressions de parsing natif. `evidence/result-review-native-v2.json` vérifie les 45 colonnes, les clés uniques, les cardinalités, les vrais tokens, l’intégrité SQLite, l’égalité CSV/SQLite et recalcule les métriques depuis les rangs. Les sources transactionnelles sont les SQLite; les CSV sont remplacés atomiquement après chaque ligne.

Chaque modèle conserve configuration, manifest, journaux de lancement et preuve GPU liée au PID/heure/poids/exécutable. Les erreurs initiales sont préservées. Les sorties brutes se trouvent dans `private-responses/`, exclu de Git; aucun prompt complet n’y est sauvegardé. Aucun identifiant/token OAuth n’a été copié dans ce lot.

Aucun email ni push n’est exécuté depuis ce lot. Les poids, archives, outils, environnements, données sources privées et sorties brutes sont exclus par `.gitignore`. Le parent coordonne l’intégration GitHub et le rapport Markdown commun destiné à Vincent.

## Admission par la VRAM physique

Sous Windows WDDM, une allocation CUDA peut réussir malgré un budget supérieur à la VRAM physique. Le lanceur contrôle désormais la somme des buffers CUDA modèle + KV + états récurrents + maximum compute avant toute inférence. Un dépassement produit `HARDWARE_VRAM_CAPACITY_EXCEEDED`, distinct d’un OOM CUDA réellement observé; aucun repli de couches ou changement de contexte n’est tenté.

Qwen3-8B au profil 32K déclare 9 097,34 MiB pour 8 192 MiB physiques. Ses 21 réponses acquises avant ce contrôle sont préservées dans `results/diagnostic-vram-oversubscription/`, avec l’intention suivante non rejouée. Elles sont exclues du lot comparatif; son registre canonique contient des issues matérielles sans taux mesuré. La validation corrigée du garde-fou a refusé le même chargement sans inférence. Les empreintes et allocations des lancements exacts sont auditées dans `evidence/native-buffer-audit.json`.

Mistral-7B-Instruct-v0.3 est également refusé avant génération : 8 230,52 MiB déclarés pour 8 192 MiB physiques. Aucun score nul ne lui est attribué : le taux mesuré reste absent. Les originaux convertis et les allocations de démarrage sont conservés.

À partir d’Ornith, `evidence/wddm-memory-telemetry.jsonl` et son résumé enregistrent les compteurs dédiés/partagés du processus vérifié. Un usage partagé non nul peut comprendre des allocations hôte ou du staging pilote; il ne prouve à lui seul ni offload transformer ni paging d’un buffer CUDA précis. Cette collecte ponctuelle ne reconstitue pas la résidence des modèles précédents.

La provenance publique regroupée, y compris les conversions locales et les poids Ollama installés, est livrée dans `evidence/model-artifact-provenance.json`; elle reste disponible lorsque les répertoires de poids exclus de Git ne sont pas transférés.
