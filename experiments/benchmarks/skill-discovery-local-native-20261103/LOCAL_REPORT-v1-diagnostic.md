# Skill Discovery — lot local Windows

État au 2026-11-03T04:54:01.513021+00:00.

Ce fichier est une section destinée au rapport commun. Il ne déclenche aucun envoi ni publication.

## État des 18 modèles

| Modèle demandé | État | Lignes enregistrées | Réponses mesurées | Sorties sans erreur | Top-1 / réponses mesurées |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | export complet | 140 | 80 | 66 | 49/80 (61.3%) |
| Qwen3-1.7B | export complet; TRANSPORT_ERROR=1, GPU_SERVER_NOT_RUNNING=13 | 140 | 66 | 22 | 15/66 (22.7%) |
| Qwen3.5-4B | export complet | 140 | 80 | 64 | 0/80 (0.0%) |
| Ling-3.0-tiny | export complet | 140 | 80 | 39 | 0/80 (0.0%) |
| Qwen3.5-9B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Llama 3.2 3B | accès gated : acceptation utilisateur requise | 0 | 0 | 0 | 0/0 (—) |
| Gemma 3 4B | accès gated : acceptation utilisateur requise | 0 | 0 | 0 | 0/0 (—) |
| H2O-Lightning-4B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Gemma 4 E2B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Qwen3-8B | en attente | 0 | 0 | 0 | 0/0 (—) |
| NeoHorse-1-9B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Ornith-1.5-9B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Nemotron 3.5 Bolt E4B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Ternary Bonsai 2 9B | en attente | 0 | 0 | 0 | 0/0 (—) |
| LFM3-Omni-2B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Phi-4 Mini 3.8B | en attente | 0 | 0 | 0 | 0/0 (—) |
| Mistral 7B Instruct | en attente | 0 | 0 | 0 | 0/0 (—) |
| Nanbeige 4.2 3B | en attente | 0 | 0 | 0 | 0/0 (—) |

Un export de 140 lignes signifie que les issues ont été enregistrées, pas que 140 appels ont réussi. Les erreurs matérielles, les accès bloqués et les dépassements de contexte ne constituent pas des observations de précision nulle.

## Résultats par taille

| Modèle | N | Réponses mesurées | Top-1 exact | Top-10 exact | Précision Top-1, sorties sans erreur | Erreurs/limites |
|---|---:|---:|---:|---:|---:|---|
| ling-3.0-tiny-q4km | 50 | 20 | 0.0% | 15.0% | 0.0% | INVALID_OUTPUT: 16, OK: 4 |
| ling-3.0-tiny-q4km | 100 | 20 | 0.0% | 55.0% | 0.0% | OK: 13, INVALID_OUTPUT: 7 |
| ling-3.0-tiny-q4km | 250 | 20 | 0.0% | 55.0% | 0.0% | OK: 14, INVALID_OUTPUT: 6 |
| ling-3.0-tiny-q4km | 500 | 20 | 0.0% | 35.0% | 0.0% | INVALID_OUTPUT: 12, OK: 8 |
| ling-3.0-tiny-q4km | 1000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| ling-3.0-tiny-q4km | 2000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| ling-3.0-tiny-q4km | 5000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3-0.6b-q8 | 50 | 20 | 70.0% | 80.0% | 82.4% | OK: 17, INVALID_OUTPUT: 3 |
| qwen3-0.6b-q8 | 100 | 20 | 60.0% | 75.0% | 66.7% | OK: 18, INVALID_OUTPUT: 2 |
| qwen3-0.6b-q8 | 250 | 20 | 50.0% | 60.0% | 71.4% | INVALID_OUTPUT: 6, OK: 14 |
| qwen3-0.6b-q8 | 500 | 20 | 65.0% | 70.0% | 76.5% | OK: 17, INVALID_OUTPUT: 3 |
| qwen3-0.6b-q8 | 1000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3-0.6b-q8 | 2000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3-0.6b-q8 | 5000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3-1.7b-q8 | 50 | 20 | 20.0% | 30.0% | 66.7% | INVALID_OUTPUT: 14, OK: 6 |
| qwen3-1.7b-q8 | 100 | 20 | 25.0% | 40.0% | 62.5% | INVALID_OUTPUT: 12, OK: 8 |
| qwen3-1.7b-q8 | 250 | 20 | 30.0% | 35.0% | 75.0% | OK: 8, INVALID_OUTPUT: 12 |
| qwen3-1.7b-q8 | 500 | 6 | 0.0% | 0.0% | — | INVALID_OUTPUT: 6, TRANSPORT_ERROR: 1, GPU_SERVER_NOT_RUNNING: 13 |
| qwen3-1.7b-q8 | 1000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3-1.7b-q8 | 2000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3-1.7b-q8 | 5000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3.5-4b-q4km | 50 | 20 | 0.0% | 10.0% | 0.0% | OK: 14, INVALID_OUTPUT: 6 |
| qwen3.5-4b-q4km | 100 | 20 | 0.0% | 75.0% | 0.0% | OK: 17, INVALID_OUTPUT: 3 |
| qwen3.5-4b-q4km | 250 | 20 | 0.0% | 60.0% | 0.0% | OK: 16, INVALID_OUTPUT: 4 |
| qwen3.5-4b-q4km | 500 | 20 | 0.0% | 75.0% | 0.0% | OK: 17, INVALID_OUTPUT: 3 |
| qwen3.5-4b-q4km | 1000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3.5-4b-q4km | 2000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |
| qwen3.5-4b-q4km | 5000 | 0 | — | — | — | CONTEXT_LIMIT_ESTIMATED: 20 |

Les taux mesurés utilisent exclusivement les lignes portant des tokens réels d’entrée et de sortie. La précision sur les sorties sans erreur a un dénominateur distinct. Les rangs invalides ou répétés restent à leur position; ils ne sont jamais retirés pour promouvoir une réponse.

## Protocole et limites qui changent l’interprétation

- Données fournies : 6 006 skills, 5 801 noms distincts, 20 questions exactes. Sept tailles : 50, 100, 250, 500, 1 000, 2 000, 5 000. Aucun dédoublonnage.
- Chaque taille réinitialise `random.Random(42).sample(all_skills, N)`. Chaque question garde cette bibliothèque, avec remplacement du dernier élément uniquement si son gold ID est absent. Les prompts utilisent seulement le nom et la description tronquée à 300 caractères; aucun corps complet.
- **Biais de position vérifié :** les 20 gold IDs fournis sont déjà les 20 premiers éléments de cet échantillonnage de seed 42. Ils occupent donc les positions 1–20 à toutes les tailles; aucun remplacement final ne s’applique sur ces données. Ce n’est pas un préfixage ajouté au runner. Les plans locaux et Luna/Sol/Terra examinés sont identiques. Ce lot ne mesure pas une recherche à positions uniformément réparties. Toute variante corrigée devra avoir une version et un run distincts.
- Serveur local demandé et vérifié quand chargé : contexte réel 32 768, un slot, toutes les couches et le KV f16 sur CUDA, `fit=off`, sans context shift, sans warmup, sans cache RAM ni checkpoints. Batch 512 / microbatch 128. CPU réservé à la tokenisation/au contrôle et à la préparation des fichiers; aucune inférence CPU ni offload partiel.
- Tokenisation réelle par `/apply-template` puis `/tokenize`, passage des IDs à `/completion`, température 0, seed 42, plafond 500 tokens de sortie; aucune troncature admise. La barrière v1 estimée `44.5*N+200 > 90% du contexte` bloque les tailles 1 000/2 000/5 000 à ce contexte, soit 60 lignes par modèle. Cette estimation n’est pas un comptage de tokens.
- Le mode thinking par défaut du modèle est conservé. Un plafond de 500 tokens peut être consommé par le raisonnement avant la liste attendue; les sorties invalides et le nombre de réponses au plafond restent explicitement visibles dans les preuves.
- Les quantifications diffèrent selon les artefacts disponibles. Les résultats comparent ces variantes précises, pas seulement des nombres de paramètres.
- La variante OAuth hôte n’impose ni température 0 ni plafond de sortie 500. Même avec des candidats identiques, ses paramètres ne sont donc pas strictement équivalents au local. Le runner original de Vincent n’a pas été fourni à ce lot; aucune reproduction octet pour octet de son exécution n’est revendiquée.
- Les 20 questions constituent un petit échantillon, en partie lexical; les noms dupliqués empêchent de résoudre exactement certains IDs dans le format de réponse par noms.
- Les temps sont ceux des appels dans cet environnement Windows 16 Go de RAM. Des téléchargements, installations et conversions CPU préparatoires ont eu lieu en parallèle de certains modèles; ces latences ne constituent pas un essai de vitesse isolé. Coût de service local 0 USD, hors électricité.

## Artefacts et identités

| Modèle | Quantification / provenance | Source éditeur vérifiée |
|---|---|---|
| Qwen3-0.6B | Q8_0 | [Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) |
| Qwen3-1.7B | Q8_0 | [Qwen/Qwen3-1.7B-GGUF](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF) |
| Qwen3.5-4B | Q4_K_M; poids Ollama déjà présents, pas un GGUF publié par Qwen | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Ling-3.0-tiny | Q4_K_M | [inclusionAI/Ling-3.0-tiny-GGUF](https://huggingface.co/inclusionAI/Ling-3.0-tiny-GGUF) |
| Qwen3.5-9B | Q4_K_M; poids Ollama déjà présents, pas un GGUF publié par Qwen | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
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

SHA-256, commits, tailles et licences sont dans `evidence/source-inventory.json` et les fichiers `.source.json` des poids. Ling installé correspond exactement au SHA-256 du fichier officiel. Phi, Mistral v0.3 et Nanbeige sont convertis/quantifiés depuis les sources officielles épinglées. H2O utilise son export officiel stock de chat ordinaire; le prompt ne sollicite pas son API de décisions typées. Le multimodal est testé en texte seul, sans projecteur chargé.

## Validation et incidents

Les 27 tests synthétiques du moteur ont passé avant les appels. `review_local_results.py` recompute les métriques depuis les rangs et vérifie largeur du CSV, clés uniques, cardinalités par taille, cohérence tokens, intégrité SQLite, égalité CSV/SQLite et plans de candidats/prompt. Sa portée et son horodatage sont enregistrés dans `evidence/result-review.json`.

Incident Qwen3-1.7B : 66 réponses mesurées; OOM CUDA dans la prélecture de la requête 67 (`500:6`), puis 13 lignes `GPU_SERVER_NOT_RUNNING` et 60 blocs estimés de contexte. Le CSV initial est conservé, aucun appel rejoué; voir `evidence/qwen17-runtime-incident.json` et le journal CUDA. Ces 14 erreurs matérielles/runtime sont exclues des taux mesurés.

Les journaux de lancement, preuves CUDA, SQLite, CSV, manifests et empreintes restent locaux. Aucun token OAuth n’a été copié dans ce lot. Aucun email ni push n’a été effectué depuis ce lot. Les poids, les données sources originales, les environnements et les téléchargements sont exclus du dépôt par `.gitignore`; l’intégration et l’envoi du rapport commun relèvent de l’orchestrateur.
