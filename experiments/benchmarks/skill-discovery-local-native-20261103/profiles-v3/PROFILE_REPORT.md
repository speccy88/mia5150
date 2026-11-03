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

Pour Llama 3.2 et Gemma 3, voir [la clôture de ces deux accès](CLOSURE.md). Aucune connexion, acceptation, publication ou livraison par email n’est exécutée ici.


Clôture finale : Llama 3.2 et Gemma 3 ne sont pas poursuivis, conformément à la décision utilisateur. Aucun résultat mesuré ne leur est attribué.
