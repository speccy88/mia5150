## Complément final Qwen8/Mistral : série v3 séparée

Les deux modèles ont fourni 200 réponses supplémentaires : 80 f16 par modèle à N=50/100/250/500, puis 20 Q8_0 par modèle à N=500. Le contexte est adapté aux tokens réels +500, sans la marge estimée de 10 % du lot initial. Les poids Q4_K_M sont inchangés ; Q8_0 désigne le cache KV. Les blocages 32K/f16 historiques restent conservés.

| Modèle | Profil | N | n_ctx | KV | Top-1 /20 | Top-10 /20 | Sans erreur | Médiane LLM (s) |
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

| Comparaison appariée N=500 | Top-1 f16 → Q8 | Top-10 f16 → Q8 | f16 seul correct | Q8 seul correct |
| --- | --- | --- | ---: | ---: |
| qwen3-8b-q4km | 8 → 10 | 9 → 10 | 1 | 3 |
| mistral-7b-instruct-v03-q4km | 3 → 3 | 10 → 8 | 0 | 0 |

Sur les quatre tailles f16, Qwen8 atteint 34/80 (42,5 %) et Mistral 36/80 (45,0 %). Les résultats Q8 restent séparés. Les 20 questions appariées ne prouvent ni une amélioration générale ni une équivalence de quantification. Aucun N≥1000 n’a été mesuré dans ce complément.

Au total, 16 modèles locaux distincts ont maintenant des mesures dans au moins une série : 1 120 réponses dans le lot initial, plus 200 dans ce complément, soit 1 320. Les trois modèles OAuth ajoutent 360 réponses, soit 1 680 réponses mesurées tous profils confondus ; ces nombres ne sont pas des observations indépendantes. Llama 3.2 et Gemma 3 restent non exécutés à la clôture demandée.

[Rapport détaillé v3](profiles-v3/PROFILE_REPORT.md) · [Validation](profiles-v3/validation/public-validation.json) · [Clôture Llama/Gemma](profiles-v3/CLOSURE.md)
