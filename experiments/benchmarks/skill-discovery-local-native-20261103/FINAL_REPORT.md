# Rapport final — découverte de skills, modèles locaux et OAuth

Les essais sont clos avec les mesures disponibles. Au total, 16 modèles locaux
distincts ont produit des mesures dans au moins une série : 1 120 réponses dans
le lot initial et 200 dans le complément Qwen8/Mistral. Les trois modèles OAuth
ajoutent 360 réponses. Ces 1 680 réponses réutilisent les mêmes 20 questions et
ne constituent pas 1 680 observations indépendantes. Llama 3.2 et Gemma 3 restent
non exécutés à la clôture demandée.

Les séries sont conservées séparément : le lot initial local utilise 32K/f16 et
un seuil de contexte estimé ; la série v3 adapte n_ctx aux tokens réels +500 sans
la marge de 10 %, et compare le cache Q8_0 à f16 à N=500 ; OAuth ne permet pas les
mêmes contrôles de température et de longueur de sortie. Les golds sont toujours
aux positions 1–20. Les scores décrivent ces conditions biaisées et exposées.

Le rapport ci-dessous rassemble le comparatif avec couverture et incertitude,
les détails du lot initial local, les profils finaux et l'analyse OAuth. Le code,
les configurations, les CSV inchangés, les exports SQLite assainis, les attributions
MIT/Apache des 6 006 skills et les manifestes accompagnent les rapports. Les bruts
privés, credentials, chemins personnels et fichiers runtime actifs sont exclus.



---

# Comparatif final

# Comparaison descriptive : bibliothèque identique, conditions d'inférence distinctes

Les nouvelles copies publiques préservent les réponses mesurées. Cette comparaison ne valide pas une équivalence API/OAuth/local : les locaux sont quantifiés, limités à 32K et 500 tokens de sortie, avec un adaptateur de réponse finale natif ; OAuth omet température/plafond de sortie et rapporte un raisonnement medium. Aucun nouveau modèle n'a été appelé.

![Qualité mesurée et couverture](https://raw.githubusercontent.com/speccy88/mia5150/main/experiments/benchmarks/skill-discovery-local-native-20261103/figures/coverage-comparison.png)

Le graphique utilise les mêmes 20 questions à chaque taille. Chaque zéro coloré résulte de réponses effectivement mesurées ; les cases sans score signalent une exclusion, un blocage ou une taille non exécutée. Les erreurs de format et raisonnements sans réponse finale restent dans le dénominateur des sorties locales mesurées.

## Panneau commun N = 50, 100, 250, 500

Les scores ci-dessous utilisent 80 mesures par modèle exécuté, issues de 20 questions répétées sur quatre tailles. Les intervalles rééchantillonnent 20 blocs question avec leurs quatre tailles, 10 000 fois, graine 20261103. Ils sont descriptifs et conditionnels aux listes biaisées, sans interprétation comme 80 essais indépendants. Aucun intervalle n'est fourni aux modèles bloqués.

| Voie | Modèle | Mesures | Top-1 exact | Bootstrap par question, 95 % | Sorties sans erreur | Médiane LLM (s) |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| local | gemma-3-4b-it | 0 | Non mesuré | — | 0 | — |
| local | gemma-4-e2b-it-q4 | 80 | 34/80 (42.5%) | 25.0% à 61.3% | 34 | 7.292 |
| local | h2o-lightning-4b-stock-q4km | 80 | 47/80 (58.8%) | 40.0% à 76.2% | 50 | 8.362 |
| local | lfm3-omni-2b-q4km | 80 | 57/80 (71.2%) | 53.8% à 87.5% | 75 | 2.036 |
| local | ling-3.0-tiny-q4km | 80 | 38/80 (47.5%) | 31.2% à 63.7% | 40 | 7.976 |
| local | llama-3.2-3b-instruct | 0 | Non mesuré | — | 0 | — |
| local | mistral-7b-instruct-v03-q4km | 0 | Non mesuré | — | 0 | — |
| local | nanbeige4.2-3b-q4km | 80 | 0/80 (0.0%) | 0.0% à 0.0% | 0 | 44.271 |
| local | nemotron-3.5-bolt-e4b-q4km | 80 | 21/80 (26.2%) | 15.0% à 38.8% | 21 | 9.637 |
| local | neohorse-1-9b-q4km | 80 | 45/80 (56.2%) | 36.2% à 75.0% | 46 | 12.655 |
| local | ornith-1.5-9b-q4km | 80 | 50/80 (62.5%) | 46.2% à 77.5% | 57 | 12.614 |
| local | phi-4-mini-instruct-q4km | 80 | 52/80 (65.0%) | 46.2% à 82.5% | 58 | 5.718 |
| local | qwen3-0.6b-q8 | 80 | 49/80 (61.3%) | 48.8% à 72.5% | 57 | 3.689 |
| local | qwen3-1.7b-q8 | 80 | 19/80 (23.8%) | 11.2% à 37.5% | 20 | 5.483 |
| local | qwen3-8b-q4km | 0 | Non mesuré | — | 0 | — |
| local | qwen3.5-4b-q4km | 80 | 28/80 (35.0%) | 22.5% à 47.5% | 32 | 8.854 |
| local | qwen3.5-9b-q4km | 80 | 25/80 (31.2%) | 17.5% à 46.2% | 27 | 13.490 |
| local | ternary-bonsai-2-9b-q2 | 80 | 23/80 (28.7%) | 17.5% à 40.0% | 25 | 10.925 |
| OAuth | gpt-6-luna | 80 | 62/80 (77.5%) | 60.0% à 92.5% | 80 | 2.593 |
| OAuth | gpt-6.1-sol | 80 | 60/80 (75.0%) | 56.2% à 91.2% | 80 | 5.239 |
| OAuth | gpt-6.1-terra | 80 | 63/80 (78.8%) | 61.3% à 95.0% | 80 | 2.758 |

## Limites et preuves

- **Biais de position :** tous les golds sont aux positions 1–20, avec la même graine que la sélection des questions. Les différences ne généralisent pas à des positions aléatoires ou centrales.
- **Couverture locale :** 14 modèles exécutés, 1 120 réponses avec usage réel ; quatre modèles sans inférence. Les 1 080 lignes de contexte estimé ne mesurent pas la fenêtre native maximale de chaque modèle.
- **Couverture OAuth :** 360 réponses mesurées à N ≤ 2000 ; N=5000 compte un rejet de longueur de chaîne HTTP et 19 non-exécutions par modèle, pas une limite de contexte en tokens établie.
- **Nanbeige :** ses 80 sorties mesurées restent sans réponse finale dans le budget 500 ; son zéro ne signifie pas une incapacité sémantique générale.
- **Matériel local :** transformer, sortie et KV sur CUDA, mais poids d’entrée/embeddings CPU_Mapped et préparation CPU conservés par llama.cpp b11960. Ce profil ne prouve pas une absence totale de calcul CPU. Voir [la portée matérielle](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-local-native-20261103/evidence/NATIVE_GPU_SCOPE.md).
- **Latence :** les temps locaux et HTTP OAuth incluent des travaux et longueurs de sortie différents. Aucun ratio de vitesse matérielle n’est revendiqué.
- **Définition native-v2 :** correction du parsing du raisonnement ; le protocole scientifique biaisé v1 reste inchangé. La v2 scientifique proposée dans le rapport OAuth n’a pas été exécutée.
- **Comparabilité externe :** la concordance avec le runner cloud de référence reste non vérifiée ; les noms dupliqués et indices lexicaux des questions demeurent.

[Rapport local](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-local-native-20261103/LOCAL_REPORT.md) · [Rapport OAuth](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-oauth-v1-20261103/REPORT.md) · [Validation hors ligne](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-local-native-20261103/validation/public-validation.json) · [Résumé du panneau commun](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-local-native-20261103/validation/common-panel-summary.csv)

Les CSV sont inchangés ; les bases distribuées sont des exports d’audit non reprisables. Les sorties HTTP privées complètes ne sont pas redistribuées, donc les vérifications publiques recalculent rangs/métriques/persistance sans rejouer chaque normalisation depuis le texte brut. Voir les transformations et empreintes pour les adaptations de confidentialité.

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

[Rapport détaillé v3](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-local-native-20261103/profiles-v3/PROFILE_REPORT.md) · [Validation](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-local-native-20261103/profiles-v3/validation/public-validation.json) · [Clôture Llama/Gemma](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-local-native-20261103/profiles-v3/CLOSURE.md)



---

# Lot local initial : résultats historiques conservés

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


---

# Complément Qwen8/Mistral : contexte et KV, série v3

# Profils locaux de contexte et de KV — résultats séparés

État : completed; vérification 2026-11-03T11:32:21.712806+00:00. 200 réponses mesurées, plafond de 200.

Les chargements initiaux à 32K/f16 dépassaient la capacité de 8 192 MiB : 9 097,34 MiB CUDA pour Qwen3-8B et 8 230,52 pour Mistral v0.3. Ce complément mesure des contextes plus petits en f16 et un cache Q8_0 distinct, en conservant les mêmes poids.

Le lot historique 32K/f16 et ses archives sont préservés. Ces mesures emploient un autre protocole de contexte : préflight réel du tokenizer + 500 tokens, arrondi au multiple de 256 supérieur. Les 20 questions, candidats, prompts, seed 42, température 0, rangs et plafond 500 restent identiques. Aucun résultat n’est mélangé avec le lot v2.

N désigne la taille du catalogue présenté. Les données validées contiennent 6 006 skills et 5 801 noms distincts; les 20 questions originales sont reprises. Les prompts utilisent seulement le nom et la description limitée à 300 caractères, sans corps complet. Le sampler et le scoring restent identiques, avec leurs doublons de noms et leur biais gold aux positions 1–20.

Écart explicite : le filtre estimé initial `44,5×N+200 > 90 % de n_ctx` ne s’applique pas dans cette version v3. Le contrôle devient `tokens réels d’entrée  + 500 ≤ n_ctx`, sans marge de 10% supplémentaire. Les contextes peuvent être proches de cette réserve exacte; chaque entrée est retokenisée avant génération et aucune troncature n’est acceptée. Ce complément ne constitue donc pas un protocole initial inchangé.

Matériel : RTX 2060 SUPER, 8 192 MiB. Runtime : llama.cpp b11960 CUDA 12.4, commit `8642541b255bfb5dc39cd351a178be09013a55d6`. Les poids Q4_K_M restent identiques au lot précédent : [Qwen3-8B officiel](https://huggingface.co/Qwen/Qwen3-8B-GGUF) et conversion locale épinglée de [Mistral v0.3 officiel](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3). Les SHA des poids, de l’exécutable et des préfixes de journaux sont liés à chaque lancement. La notation Q8_0 de ce rapport désigne le cache KV, pas une autre quantification des poids.

## Mesures par profil

| Modèle | Profil | N | n_ctx | KV | Réponses | Top-1 | Top-10 | Sans erreur | Médiane LLM (s) | Issues |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| qwen3-8b-q4km | f16-adjusted-n50 | 50 | 2816 | f16 | 20 | 10/20 (50.00%) | 10.0/20 (50.00%) | 10 | 9.94 | OK=10, REASONING_NO_FINAL_ANSWER=10 |
| qwen3-8b-q4km | f16-adjusted-n100 | 100 | 5120 | f16 | 20 | 8/20 (40.00%) | 8.0/20 (40.00%) | 8 | 11.94 | OK=8, REASONING_NO_FINAL_ANSWER=12 |
| qwen3-8b-q4km | f16-adjusted-n250 | 250 | 12032 | f16 | 20 | 8/20 (40.00%) | 8.0/20 (40.00%) | 8 | 19.56 | REASONING_NO_FINAL_ANSWER=12, OK=8 |
| qwen3-8b-q4km | f16-adjusted-n500 | 500 | 23552 | f16 | 20 | 8/20 (40.00%) | 9.0/20 (45.00%) | 10 | 34.46 | REASONING_NO_FINAL_ANSWER=10, OK=10 |
| qwen3-8b-q4km | q8-adjusted-n500 | 500 | 23552 | q8_0 | 20 | 10/20 (50.00%) | 10.0/20 (50.00%) | 11 | 42.22 | OK=11, REASONING_NO_FINAL_ANSWER=8, EMPTY_OUTPUT=1 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n50 | 50 | 3328 | f16 | 20 | 16/20 (80.00%) | 20.0/20 (100.00%) | 20 | 5.85 | OK=20 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n100 | 100 | 5888 | f16 | 20 | 9/20 (45.00%) | 15.0/20 (75.00%) | 20 | 11.55 | OK=20 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n250 | 250 | 14080 | f16 | 20 | 8/20 (40.00%) | 12.0/20 (60.00%) | 19 | 20.08 | OK=19, INVALID_OUTPUT=1 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n500 | 500 | 27392 | f16 | 20 | 3/20 (15.00%) | 10.0/20 (50.00%) | 19 | 37.40 | OK=19, INVALID_OUTPUT=1 |
| mistral-7b-instruct-v03-q4km | q8-adjusted-n500 | 500 | 27392 | q8_0 | 20 | 3/20 (15.00%) | 8.0/20 (40.00%) | 18 | 46.02 | OK=18, INVALID_OUTPUT=2 |

## Totaux f16 aux quatre tailles

| Modèle | Réponses | Top-1 | Top-10 | Sorties sans erreur |
|---|---:|---:|---:|---:|
| mistral-7b-instruct-v03-q4km | 80 | 36/80 (45.00%) | 57.0/80 (71.25%) | 78 |
| qwen3-8b-q4km | 80 | 34/80 (42.50%) | 35.0/80 (43.75%) | 36 |

Ces totaux regroupent les quatre contextes f16 ajustés; les résultats Q8_0 restent dans leur comparaison appariée.

## Admissions matérielles

| Modèle | Profil | n_ctx | KV | État | CUDA déclaré MiB | WDDM admission MiB | WDDM maximum échantillonné MiB |
|---|---|---:|---|---|---:|---:|---:|
| qwen3-8b-q4km | preflight-f16-4096 | 4096 | f16 | completed_no_generation | 5058.34 | 5242.92 | - |
| qwen3-8b-q4km | q8-32768-admission-only | 32768 | q8_0 | completed_no_generation | 7049.42 | 7242.95 | 7242.95 |
| qwen3-8b-q4km | f16-adjusted-n50 | 2816 | f16 | completed | 4878.03 | 5062.91 | 5078.91 |
| qwen3-8b-q4km | f16-adjusted-n100 | 5120 | f16 | completed | 5202.59 | 5386.92 | 5402.92 |
| qwen3-8b-q4km | f16-adjusted-n250 | 12032 | f16 | completed | 6176.28 | 6362.93 | 6378.93 |
| qwen3-8b-q4km | f16-adjusted-n500 | 23552 | f16 | completed | 7799.09 | 7986.95 | 8002.95 |
| qwen3-8b-q4km | q8-adjusted-n500 | 23552 | q8_0 | completed | 6322.67 | 6510.93 | 6532.93 |
| mistral-7b-instruct-v03-q4km | preflight-f16-4096 | 4096 | f16 | completed_no_generation | 4639.52 | 4824.91 | - |
| mistral-7b-instruct-v03-q4km | q8-32768-admission-only | 32768 | q8_0 | completed_no_generation | 6419.60 | 6612.94 | 6612.94 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n50 | 3328 | f16 | completed | 4543.34 | 4726.91 | 4742.91 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n100 | 5888 | f16 | completed | 4863.96 | 5048.91 | 5064.91 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n250 | 14080 | f16 | completed | 5889.96 | 6076.93 | 6092.93 |
| mistral-7b-instruct-v03-q4km | f16-adjusted-n500 | 27392 | f16 | completed | 7557.21 | 7746.95 | 7762.95 |
| mistral-7b-instruct-v03-q4km | q8-adjusted-n500 | 27392 | q8_0 | completed | 6040.29 | 6230.94 | 6248.94 |

Les chargements `admission-only` et les préflights n’exécutent aucune génération. K et V doivent avoir le type demandé dans les journaux, Flash Attention doit être active, les couches transformer/sortie et le KV doivent rester CUDA. Le total CUDA déclaré et le total engagé WDDM sont contrôlés avant inférence. L’entrée/les embeddings CPU_Mapped par défaut sont explicitement conservés; aucun repli de couches transformer n’est autorisé. Ces compteurs restent distincts d’une trace exhaustive de résidence.

Les latences incluent le préfill et la génération, avec longueur de sortie variable plafonnée à 500; le chargement des poids et leurs hashes sont exclus du temps LLM. Elles décrivent cette session Windows avec télémétrie, sans isolation complète de la machine.

## Comparaison appariée de qualité

| Modèle | N | n_ctx commun | Questions appariées | Top-1 f16 / Q8 | Top-10 f16 / Q8 | Textes bruts identiques |
|---|---:|---:|---:|---|---|---:|
| qwen3-8b-q4km | 500 | 23552 | 20 | 8 / 10 | 9 / 10 | 0 |
| mistral-7b-instruct-v03-q4km | 500 | 27392 | 20 | 3 / 3 | 10 / 8 | 11 |

Q8_0 est une quantification avec arrondi : 32 valeurs int8 et une échelle f16, soit 8,5 bits/valeur contre 16 pour f16. Les [essais initiaux de llama.cpp](https://github.com/ggml-org/llama.cpp/pull/7412) motivent ce seul format quantifié ici, sans garantir son équivalence. Les mêmes 20 questions au même contexte permettent une comparaison locale; ce petit jeu ne remplace pas une évaluation générale de perplexité.

Les sorties sans finale au plafond 500 sont des échecs de format/génération de ce profil. Les refus matériels n’ont aucun taux mesuré. Le biais gold aux positions 1–20 du seed 42 reste présent. Aucun contexte pour N≥1000, nouveau poids, autre quantification ou appel hébergé n’est testé.

Preuves : `evidence/profile-review.json`, `evidence/runtime-kv-support.json`, `evidence/candidate-comparison.json`, préflights et journaux sous `runtime/`; SQLite transactionnels et CSV 45 colonnes sous `results/`. Les réponses brutes sont vérifiées par SHA et restent privées. Les fichiers `baseline-preservation-*.json` vérifient les archives précédentes.

Pour Llama 3.2 et Gemma 3, voir [la clôture de ces deux accès](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-local-native-20261103/profiles-v3/CLOSURE.md). Aucune connexion, acceptation, publication ou livraison par email n’est exécutée ici.


Clôture finale : Llama 3.2 et Gemma 3 ne sont pas poursuivis, conformément à la décision utilisateur. Aucun résultat mesuré ne leur est attribué.


---

# Résultats OAuth et limites méthodologiques

# Les trois modèles restent proches sur ce test, avec une couverture limitée à 2 000 skills

Étude de découverte de skills par listing, réalisée le 3 novembre 2026. Modèles : `gpt-6-luna`, `gpt-6.1-sol`, `gpt-6.1-terra`. Cette exécution est une **variante OAuth du protocole v1**, pas une reproduction strictement équivalente à une API avec température 0 et sortie limitée à 500 tokens.

## Ce que les résultats permettent de dire

- Les trois modèles ont produit **120 réponses valides chacun**, sur les tailles 50, 100, 250, 500, 1 000 et 2 000. Leur score top-1 exact se situe entre **72,5 % et 75,0 %** sur ces six tailles. Les différences appariées restent compatibles avec zéro sur les 20 questions disponibles.
- À 5 000 skills, chaque modèle a subi **un rejet HTTP 400 `string_above_max_length`**. Les 19 autres essais de cette taille n’ont pas été exécutés. Ce rejet de longueur de chaîne ne mesure pas une limite de contexte du modèle et ne vaut pas 0 % de précision.
- L’audit démontre un biais plus précis que la seule règle « remplacer le dernier » : **les 20 golds sont déjà aux positions 1 à 20 pour les sept tailles**. Le remplacement final ne s’est jamais déclenché. Les performances sont donc conditionnelles à une position initiale très favorable des réponses attendues.
- Les données brutes v1 sont conservées. La comparabilité avec les résultats cloud précédents reste **non vérifiée**, faute du runner original et de ses listes ordonnées. Les différences OAuth constituent de toute façon une condition expérimentale distincte.

## Couverture réelle et performances observées

| Modèle | Réponses / prévues | Top-1 exact, tailles mesurées | Latence moyenne / médiane (s) | Tokens entrée / sortie, rapportés |
| --- | --- | --- | --- | --- |
| Luna | 120/140 | 88/120 (73,3 %) | 2,872 / 2,743 | 3 444 430 / 11 830 |
| Sol 6.1 | 120/140 | 87/120 (72,5 %) | 5,952 / 5,896 | 3 444 430 / 11 812 |
| Terra 6.1 | 120/140 | 90/120 (75,0 %) | 3,088 / 2,939 | 3 444 430 / 5 194 |

Les pourcentages ci-dessus sont conditionnels aux **six tailles entièrement mesurées**, et non aux 140 lignes prévues. Les latences couvrent l’aller-retour HTTP et la génération, sans la pause d’une seconde entre requêtes. Elles dépendent aussi du réseau, du service, du cache de préfixe éventuel et de la longueur des sorties : ce ne sont pas des mesures isolées du matériel du modèle.

Les trois CSV contiennent bien **420 lignes au total, avec les 45 colonnes demandées**. Ils se décomposent en **360 réponses valides, 3 rejets et 57 lignes `NOT_RUN_AFTER_MODEL_STOP`**. Il y a eu 363 requêtes de benchmark, plus les 3 petits essais d’admission, soit 366 POST d’inférence demandés au service. Les rejets n’ont pas été retentés ; aucun quota ni rate limit n’a été observé. Aucun benchmark local GPU n’a été exécuté dans cette étude.

Les champs numériques à zéro des lignes non exécutées ne sont que des valeurs de structure du format CSV : **ils ne constituent pas des observations**. Ces lignes sont exclues des scores, intervalles et courbes ci-dessous. Aucun prix API n’a été appliqué : `est_cost_usd` reste vide. L’usage a été demandé via la connexion OAuth consentie, sans clé API classique, achat de crédits ni bascule de facturation. Le coût monétaire réel du forfait n’a pas été mesuré.

### Top-1 exact par taille

Chaque cellule mesurée porte sur les **mêmes 20 questions**, réutilisées à la taille suivante. Les intervalles sont des intervalles de Wilson à 95 % calculés avec **n = 20 par taille**.

| Taille | Luna, exact et IC 95 % | Sol 6.1, exact et IC 95 % | Terra 6.1, exact et IC 95 % |
| --- | --- | --- | --- |
| 50 | 17/20 = 85,0 % [64,0–94,8 %] | 18/20 = 90,0 % [69,9–97,2 %] | 17/20 = 85,0 % [64,0–94,8 %] |
| 100 | 16/20 = 80,0 % [58,4–91,9 %] | 14/20 = 70,0 % [48,1–85,5 %] | 15/20 = 75,0 % [53,1–88,8 %] |
| 250 | 15/20 = 75,0 % [53,1–88,8 %] | 14/20 = 70,0 % [48,1–85,5 %] | 15/20 = 75,0 % [53,1–88,8 %] |
| 500 | 14/20 = 70,0 % [48,1–85,5 %] | 14/20 = 70,0 % [48,1–85,5 %] | 16/20 = 80,0 % [58,4–91,9 %] |
| 1000 | 13/20 = 65,0 % [43,3–81,9 %] | 13/20 = 65,0 % [43,3–81,9 %] | 13/20 = 65,0 % [43,3–81,9 %] |
| 2000 | 13/20 = 65,0 % [43,3–81,9 %] | 14/20 = 70,0 % [48,1–85,5 %] | 14/20 = 70,0 % [48,1–85,5 %] |
| 5000 | Non mesuré : 1 rejet, 19 non exécutés | Non mesuré : 1 rejet, 19 non exécutés | Non mesuré : 1 rejet, 19 non exécutés |

![Précision et latence par taille de bibliothèque](https://raw.githubusercontent.com/speccy88/mia5150/main/experiments/benchmarks/skill-discovery-oauth-v1-20261103/figures/scaling.png)

La baisse observée entre 50 et 2 000 skills est de 20 points pour Luna et Sol, et de 15 points pour Terra. Elle décrit ce jeu de questions et ces listes ; elle ne démontre pas une loi générale de dégradation. Les golds toujours placés au début ne testent notamment pas le cas où la réponse est enfouie au milieu d’un long contexte.

### Couverture, latence et tokens par taille et modèle

Les tokens sont ceux de `response.completed.usage`, rapportés par le serveur. Les tokens de sortie incluent le raisonnement lorsqu’il est comptabilisé par le service. Les rejets ne retournent pas d’usage exploitable ; aucune estimation ne remplace ces valeurs manquantes.

| Modèle | N | Réponses valides | Rejets | Non exécutés | Latence moyenne (s) | Tokens entrée, moyenne | Tokens sortie, moyenne |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Luna | 50 | 20/20 | 0 | 0 | 2,408 | 2142,75 | 90,85 |
| Luna | 100 | 20/20 | 0 | 0 | 2,697 | 4236,75 | 90,30 |
| Luna | 250 | 20/20 | 0 | 0 | 2,861 | 11034,75 | 95,90 |
| Luna | 500 | 20/20 | 0 | 0 | 3,061 | 21867,75 | 102,15 |
| Luna | 1000 | 20/20 | 0 | 0 | 3,017 | 43843,75 | 111,35 |
| Luna | 2000 | 20/20 | 0 | 0 | 3,190 | 89095,75 | 100,95 |
| Luna | 5000 | 0/20 | 1 | 19 | — | — | — |
| Sol 6.1 | 50 | 20/20 | 0 | 0 | 4,425 | 2142,75 | 67,40 |
| Sol 6.1 | 100 | 20/20 | 0 | 0 | 5,248 | 4236,75 | 83,95 |
| Sol 6.1 | 250 | 20/20 | 0 | 0 | 5,929 | 11034,75 | 109,95 |
| Sol 6.1 | 500 | 20/20 | 0 | 0 | 6,472 | 21867,75 | 117,95 |
| Sol 6.1 | 1000 | 20/20 | 0 | 0 | 6,130 | 43843,75 | 102,20 |
| Sol 6.1 | 2000 | 20/20 | 0 | 0 | 7,507 | 89095,75 | 109,15 |
| Sol 6.1 | 5000 | 0/20 | 1 | 19 | — | — | — |
| Terra 6.1 | 50 | 20/20 | 0 | 0 | 2,721 | 2142,75 | 26,90 |
| Terra 6.1 | 100 | 20/20 | 0 | 0 | 2,865 | 4236,75 | 32,70 |
| Terra 6.1 | 250 | 20/20 | 0 | 0 | 2,946 | 11034,75 | 42,75 |
| Terra 6.1 | 500 | 20/20 | 0 | 0 | 3,312 | 21867,75 | 47,80 |
| Terra 6.1 | 1000 | 20/20 | 0 | 0 | 3,248 | 43843,75 | 50,20 |
| Terra 6.1 | 2000 | 20/20 | 0 | 0 | 3,437 | 89095,75 | 59,35 |
| Terra 6.1 | 5000 | 0/20 | 1 | 19 | — | — | — |

Les 360 réponses valides totalisent **10 333 290 tokens d’entrée** et **28 836 tokens de sortie rapportés**. Les essais d’admission, séparés de ces résultats, totalisent 429 tokens d’entrée et 50 de sortie. Les totaux n’imputent aucun faux zéro d’usage aux trois rejets.

### Qualité du classement

| Modèle | N | R@1 | R@3 | R@5 | R@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Luna | 50 | 85,0 % | 90,0 % | 95,0 % | 95,0 % | 0,8767 | 0,8943 |
| Luna | 100 | 80,0 % | 85,0 % | 85,0 % | 95,0 % | 0,8368 | 0,8624 |
| Luna | 250 | 75,0 % | 85,0 % | 85,0 % | 85,0 % | 0,7917 | 0,8065 |
| Luna | 500 | 70,0 % | 80,0 % | 80,0 % | 80,0 % | 0,7417 | 0,7565 |
| Luna | 1000 | 65,0 % | 70,0 % | 75,0 % | 85,0 % | 0,7030 | 0,7376 |
| Luna | 2000 | 65,0 % | 75,0 % | 85,0 % | 85,0 % | 0,7117 | 0,7452 |
| Sol 6.1 | 50 | 90,0 % | 95,0 % | 100,0 % | 100,0 % | 0,9267 | 0,9443 |
| Sol 6.1 | 100 | 70,0 % | 90,0 % | 95,0 % | 100,0 % | 0,8183 | 0,8633 |
| Sol 6.1 | 250 | 70,0 % | 80,0 % | 90,0 % | 90,0 % | 0,7558 | 0,7909 |
| Sol 6.1 | 500 | 70,0 % | 80,0 % | 80,0 % | 85,0 % | 0,7488 | 0,7732 |
| Sol 6.1 | 1000 | 65,0 % | 70,0 % | 75,0 % | 80,0 % | 0,6931 | 0,7181 |
| Sol 6.1 | 2000 | 70,0 % | 70,0 % | 75,0 % | 80,0 % | 0,7183 | 0,7372 |
| Terra 6.1 | 50 | 85,0 % | 95,0 % | 95,0 % | 95,0 % | 0,8833 | 0,9000 |
| Terra 6.1 | 100 | 75,0 % | 85,0 % | 90,0 % | 90,0 % | 0,8017 | 0,8259 |
| Terra 6.1 | 250 | 75,0 % | 80,0 % | 85,0 % | 95,0 % | 0,7981 | 0,8326 |
| Terra 6.1 | 500 | 80,0 % | 80,0 % | 80,0 % | 80,0 % | 0,8000 | 0,8000 |
| Terra 6.1 | 1000 | 65,0 % | 75,0 % | 80,0 % | 80,0 % | 0,7100 | 0,7324 |
| Terra 6.1 | 2000 | 70,0 % | 75,0 % | 75,0 % | 80,0 % | 0,7333 | 0,7494 |

Les métriques complètes à K = 1, 3, 5 et 10 figurent dans les CSV. Avec un seul gold, un hit à un rang r ≤ K donne R@K = 1, P@K = 1/K, NDCG@K = 1/log2(r+1), MRR@K = AP@K = 1/r. Ainsi P@10 ne peut pas dépasser 0,1 ; ce n’est pas une précision de sélection top-1 de 10 %.

| Modèle | Noms invalides dans les listes, cumul | Top-1 supplémentaire par fallback de casse |
| --- | --- | --- |
| Luna | 30 | 0 |
| Sol 6.1 | 4 | 0 |
| Terra 6.1 | 1 | 0 |

Une réponse « valide » signifie ici qu’elle a terminé sans erreur technique et contient au moins un nom reconnu. Elle peut encore contenir des noms invalides dans d’autres rangs. Les positions invalides et les répétitions sont conservées : une bonne réponse classée deuxième n’est jamais promue première après suppression d’une mauvaise entrée. Le fallback de casse est suivi séparément ; il n’a ajouté aucun succès top-1 dans ce run.

## Incertitude : 20 questions, pas 140 observations indépendantes

Les 120 réponses valides d’un modèle sont 20 questions répétées sur six tailles. Un intervalle binomial calculé comme si n = 120 ou 140 était indépendant sous-estimerait cette dépendance. Pour la moyenne globale, le bootstrap rééchantillonne **20 blocs « question »** ; chaque bloc conserve ses six tailles et les trois modèles. Les comparaisons entre modèles utilisent les mêmes indices rééchantillonnés : 10 000 réplicats, graine d’analyse 20261103, intervalles percentiles à 95 %.

| Modèle | Moyenne sur les six tailles | Intervalle bootstrap par question, 95 % |
| --- | --- | --- |
| Luna | 73,3 % | 55,0–90,0 % |
| Sol 6.1 | 72,5 % | 54,2–90,0 % |
| Terra 6.1 | 75,0 % | 56,7–90,8 % |

| Comparaison, gauche moins droite | Écart observé (points) | Intervalle apparié 95 % (points) |
| --- | --- | --- |
| Luna − Sol 6.1 | 0,83 | -2,50 à 4,17 |
| Luna − Terra 6.1 | -1,67 | -5,83 à 1,67 |
| Sol 6.1 − Terra 6.1 | -2,50 | -8,33 à 1,67 |

Tous les intervalles appariés contiennent zéro. Cela ne prouve pas l’égalité des modèles ; l’échantillon ne permet pas d’affirmer une supériorité fiable. Les intervalles de Wilson par taille supposent des cas Bernoulli distincts conditionnellement à la liste. Le bootstrap décrit la sensibilité à ces 20 cas, avec listes et paramètres observés fixés. Ni l’un ni l’autre ne couvre le biais de position, la fuite des noms, les autres graines de bibliothèques, la variabilité du décodage ou toute la population des tâches. Ils restent descriptifs, sans correction de multiplicité ni revendication confirmatoire.

## Audit du protocole et limites qui changent l’interprétation

### Le gold est toujours au début, et non forcé à la fin dans ce run

La reconstruction indépendante suit exactement l’ordre reçu, `random.Random(42).sample(all_skills, N)`, puis remplace le dernier élément seulement si le `gold_id` est absent. Pour les trois modèles, **les 140 séquences ordonnées de candidats et leurs empreintes de messages logiques correspondent aux manifestes du run**.

| N, IDs | Noms distincts | Gold remplacés en dernier | Positions des 20 golds | Caractères dans instructions |
| --- | --- | --- | --- | --- |
| 50 | 50 | 0 | 1 à 20, dans l’ordre des questions | 10 295 |
| 100 | 100 | 0 | 1 à 20, dans l’ordre des questions | 20 851 |
| 250 | 250 | 0 | 1 à 20, dans l’ordre des questions | 54 255 |
| 500 | 499 | 0 | 1 à 20, dans l’ordre des questions | 108 801 |
| 1000 | 992 | 0 | 1 à 20, dans l’ordre des questions | 217 958 |
| 2000 | 1973 | 0 | 1 à 20, dans l’ordre des questions | 438 935 |
| 5000 | 4856 | 0 | 1 à 20, dans l’ordre des questions | 1 094 067 |

Les gold IDs du fichier reçu coïncident aussi exactement, dans l’ordre des questions, avec `random.Random(42).sample(all_skills, 20)`. C’est un couplage effectif du jeu de questions et du tirage de candidats. La règle de remplacement en dernière position demeure une limite potentielle du protocole général, mais **elle n’a pas été activée ici**. L’hypothèse de tirages indépendants donnant environ 0,8 % de présence à N=50 ne décrit donc pas ce run. L’avantage positionnel observé est un biais de début de liste. Rien n’a été réordonné après découverte de ce point.

### Noms dupliqués, fuite des noms et pertinence unique

Les sources contiennent **6 006 IDs pour 5 801 noms distincts**, soit **153 groupes de noms dupliqués et 358 enregistrements concernés**. Trois des 20 golds portent un nom non unique dans la source. Aucun dédoublonnage n’a été effectué. Un modèle répondant uniquement par nom ne peut pas distinguer deux IDs portant le même nom : ces scores portent sur le **nom attendu**, pas sur une identité UUID prouvée.

Une vérification conservatrice trouve une correspondance exacte du nom normalisé dans **6 requêtes sur 20** (casse et séparateurs normalisés). Ce comptage n’exclut pas d’autres indices lexicaux dans les 14 restantes. Les requêtes guidées par le nom rendent le test en partie lexical. De plus, un seul gold est fourni alors que plusieurs skills peuvent être sémantiquement acceptables ; une alternative pertinente peut donc être comptée fausse. Aucune annotation n’a été corrigée après lecture des sorties.

### Le rejet à 5 000 est une limite d’enveloppe, pas une mesure de contexte

À N=5 000, la chaîne `instructions` reconstruite contient **1 094 067 caractères** (1 095 016 octets UTF-8). Les trois services ont renvoyé `string_above_max_length`. Les réponses assainies conservées ne donnent pas la valeur maximale exacte du champ ; elle n’est donc pas inventée ici. Le code d’erreur est un rejet de longueur de chaîne, et les indicateurs `context_limit` restent faux. Il ne permet pas de conclure que les modèles eux-mêmes ne peuvent pas traiter les tokens correspondants.

Le catalogue OAuth du compte annonçait **272 000 tokens pour Luna** et **500 000 pour Sol et Terra**, contre un maximum de 1 050 000 sur les fiches API publiques. Les limites du catalogue authentifié ont été utilisées. La barrière v1 `44,5 × N + 200 > 90 % de la fenêtre` n’a bloqué aucun essai : à 5 000, elle estimait 222 700 tokens. C’est une estimation, pas un comptage réel du prompt rejeté. Aucun dépassement de contexte en tokens n’a été mesuré.

### Différences OAuth et comparabilité cloud

La voie officielle OAuth utilise Responses en streaming, `store=false`, et le texte système dans `instructions`. Elle ne permet pas d’imposer `temperature=0` ni `max_output_tokens=500` : ces champs sont absents, sans modification silencieuse. Le raisonnement n’a pas été fixé dans la requête ; les 360 réponses terminées déclarent `medium`. Il n’y a eu ni outil agent, ni retrieval externe, ni exécution des skills. Les descriptions sont limitées à 300 caractères Unicode par le protocole.

Les modèles ont été exécutés séquentiellement, avec une pause d’une seconde hors latence par requête. Le cache de préfixe du fournisseur n’est pas contrôlé sur cette route. Les latences de sorties de longueurs différentes ne sont pas des mesures matérielles comparables à des modèles locaux GPU. Les alias de modèles n’ont pas été remplacés par des versions datées inventées ; l’identité renvoyée par le serveur a été auditée pendant le run.

Les deux fichiers reçus et le sampling littéral sont vérifiés, mais le runner et les candidats ordonnés du run cloud de référence n’ont pas été fournis. Il faut vérifier la portée de son RNG, les chaînes effectives, le tokenizer, les paramètres et les dénominateurs avant toute fusion de courbes. La formule de précision de sélection est TP / requêtes valides ; le rappel de sélection sur tous les cas est TP / 20 à une taille entièrement exécutée. Le rappel n’est pas simplement le taux de réponses : une réponse fausse n’est pas un TP. La disponibilité est une mesure séparée.

## Validation et reproduction

- Schéma exact de 45 colonnes, unicité de 140 couples taille/question par modèle et golds concordants.
- Correspondance champ par champ entre CSV et snapshot transactionnel SQLite des 363 essais enregistrés.
- Recalcul indépendant de toutes les métriques à K, des sommes de tokens et des dénominateurs ; latences finies et non négatives ; coûts inconnus laissés vides.
- Vérification indépendante des séquences de candidats et hashes des 140 messages logiques pour chacun des trois modèles.
- **28 tests synthétiques réussis**, sans nouvelle inférence. Un test a mis en évidence un verrou de fichier Windows transitoire ; le runner partageable ajoute un retry borné du remplacement atomique du fichier, jamais du POST modèle. Le code exact mesuré reste archivé séparément.

Les classements/rangs exportés sont conservés, ainsi que les hashes des réponses. Le texte HTTP intégral des réponses n’a pas été sauvegardé ; on ne peut donc pas réauditer indépendamment chaque normalisation de format ou de suffixe à partir du texte brut. Le contrôle arithmétique ne doit pas être présenté comme une validation sémantique des golds.

Le paquet fournit une dérivée minimale des données : ID, nom et les 300 premiers caractères de description, dans l’ordre original. **Les 140 prompts logiques restent identiques**, ce qui est vérifié contre les manifestes. Le fichier complet original de 117,6 Mo, avec corps de skills inutilisés, n’est pas redistribué. Les deux hashes d’origine et le hash de la dérivée figurent dans [la provenance des données](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-oauth-v1-20261103/data/PROVENANCE.json). Le runner de reproduction accepte explicitement le hash de cette dérivée ; le code exact utilisé pour les mesures reste dans `source-original/`. Cette adaptation de packaging et le retry de fichiers ne sont pas une correction du sampling v1 et ne changent aucun résultat mesuré.

Pour reproduire les calculs et les figures sans appeler de modèle :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-oauth.txt -r requirements-analysis.txt
.\.venv\Scripts\python.exe analyze_public.py
```

Pour refaire des mesures, installer séparément les dépendances OAuth, connecter son propre compte et exécuter l’admission avant la suite, comme décrit dans [README.md](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-oauth-v1-20261103/README.md). Cela consomme l’usage de ce compte et peut produire d’autres réponses ; aucun nouveau run n’a été lancé pour ce rapport. Aucun fichier d’accès, état OAuth, identifiant d’installation, lien de review, jeton DPAPI ou journal sensible ne fait partie du paquet.

## Corrections proposées pour une v2 distincte

1. **Séparer les flux aléatoires** de sélection des questions, des candidats et de leur ordre. Après inclusion du gold, mélanger systématiquement la liste, même si le gold était déjà présent. Contrôler et, si souhaité, équilibrer les positions début/milieu/fin. Enregistrer plusieurs graines et la position exacte du gold.
2. **Écrire davantage de tâches indépendantes sans nom de skill**, avec diversité de domaines et revue à l’aveugle. Partir d’un jeu pilote d’au moins 100 cas, puis dimensionner l’échantillon selon la précision statistique recherchée ; ce nombre n’est pas une garantie de puissance.
3. **Annoter plusieurs réponses pertinentes lorsque nécessaire** et utiliser des IDs non ambigus. Adapter explicitement les métriques IR à plusieurs golds. Garder séparés le respect du format, la validité des noms, la disponibilité et la qualité sémantique.
4. **Traiter la limite de transport avant les grands runs** : mesurer caractères/octets/tokens et tester un format d’entrée officiellement accepté. Une modification de découpage ou de rôle des messages doit être nommée, validée et comparée dans un bras distinct ; aucune entrée refusée n’a été fractionnée ou retentée dans v1.
5. **Séparer les conditions OAuth et API/locales**. Consigner les contrôles effectivement supportés et le raisonnement observé ; répéter le décodage pour estimer sa variabilité. Ne pas supposer que tous les modèles acceptent température, seed ou plafond de sortie.
6. **Préenregistrer l’analyse et conserver les éléments d’audit utiles** : texte des sorties si le cadre de partage le permet, paramètres effectifs, traces d’erreurs assainies, ordre/positions, usages et cache lorsqu’ils sont fournis. Rapporter les comparaisons appariées par question et distinguer l’incertitude d’échantillonnage des biais systématiques.

Ces propositions n’ont pas été appliquées rétroactivement. Les trois jeux de résultats v1 restent intacts ; un nouveau protocole demanderait une nouvelle exécution séparée.

## Sources et fichiers d’appui

- Données fournies : `test.jsonl` et `test_queries.json`, hashes SHA-256 d’origine dans [PROVENANCE.json](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-oauth-v1-20261103/data/PROVENANCE.json).
- [Résultats Luna](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-oauth-v1-20261103/results/gpt-6-luna/details.csv), [Sol](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-oauth-v1-20261103/results/gpt-6.1-sol/details.csv), [Terra](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-oauth-v1-20261103/results/gpt-6.1-terra/details.csv), [analyse reproductible](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-oauth-v1-20261103/results/analysis.json), [avertissement méthodologique](https://github.com/speccy88/mia5150/blob/main/experiments/benchmarks/skill-discovery-oauth-v1-20261103/results/METHODOLOGY_WARNING.json).
- [OpenAI — usage du forfait OAuth et inférence](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference).
- [OpenAI — limitations de la route OAuth](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations).
- [Fiche API Luna](https://developers.openai.com/api/docs/models/gpt-6-luna), [Sol 6.1](https://developers.openai.com/api/docs/models/gpt-6.1-sol), [Terra 6.1](https://developers.openai.com/api/docs/models/gpt-6.1-terra). Les valeurs de catalogue OAuth observées sont conservées séparément, sans identité de compte.

### Admission, hors scores de benchmark

| Modèle | Résultat | Durée (s) | Tokens entrée / sortie |
| --- | --- | --- | --- |
| Luna | Admis, réponse exacte | 2,154 | 143 / 32 |
| Sol 6.1 | Admis, réponse exacte | 2,010 | 143 / 9 |
| Terra 6.1 | Admis, réponse exacte | 2,038 | 143 / 9 |

