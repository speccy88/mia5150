# Verdict — échantillonnage et positions gold

PASS pour la construction demandée : `random.Random(42).sample(all_skills, N)`
sur les 6 006 lignes, copie par question, puis remplacement du dernier élément
uniquement si le gold ID est absent. Aucun préfixe gold n'est ajouté au code.

La reproduction indépendante lit directement les JSON d'origine, sans appeler
`bench.make_plan`. Elle confirme que les 20 gold IDs fournis correspondent déjà
aux 20 premiers tirages de seed 42. Le gold de la question i occupe donc la
position i pour chaque taille 50, 100, 250, 500, 1000, 2000, 5000. Aucun swap
n'est nécessaire sur ces données. Ce biais de position est réel et doit figurer
dans toute conclusion; un test à positions réparties serait une variante distincte.

Le code exécuté est `bench.py` SHA-256
`992ac9549bcada4ba29018ede5e7c932a78a98578092eaab4d10792a8d378e9c`.
Les manifests locaux et les trois manifests OAuth examinés ont exactement les
mêmes plans de candidats et hashes de prompts. Le runner original de Vincent
n'était pas disponible pour établir une reproduction exacte de son propre run.

Preuves : `gold-position-independent-check.json` et `gold-position-audit.json`.

Lot : `./`.
État dynamique : `results/local-v1/suite-status.json`.
Rapport récapitulatif : `LOCAL_REPORT.md` (snapshot daté; à régénérer après la suite).

Qwen3-0.6B : 140 lignes, 80 réponses mesurées, 60 limites de contexte estimées.
Qwen3-1.7B : 140 lignes, 66 réponses mesurées, OOM à la requête 67, puis 13 lignes
serveur arrêté, plus 60 limites estimées. Aucun OOM n'est résumé en précision nulle.
Les résultats originaux sont conservés et aucune inférence n'a été rejouée.
