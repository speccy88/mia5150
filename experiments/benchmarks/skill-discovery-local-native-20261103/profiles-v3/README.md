# Complément final : contextes adaptés et cache KV, série v3 distincte

Lire [PROFILE_REPORT.md](PROFILE_REPORT.md), [le comparatif final](../COMPARISON.md)
et [les transformations de publication](PUBLICATION_TRANSFORMS.json).

200 réponses : pour chacun de Qwen3-8B et Mistral v0.3, 80 réponses f16 aux tailles
50/100/250/500, puis 20 réponses Q8_0 au même contexte que f16 à N=500. Les poids
Q4_K_M ne changent pas. Q8_0 désigne ici le cache KV, pas une nouvelle quantification
des poids. Les admissions sans génération ne sont pas des résultats de qualité.

La règle initiale d'estimation avec marge 10 % est remplacée par les tokens réels
d'entrée +500, avec n_ctx arrondi au multiple de 256 supérieur. Les listes, questions,
descriptions, température 0, plafond de sortie 500 et scoring sont conservés. Ce
changement de contexte est explicite : ne pas fusionner cette série avec le lot
historique 32K/f16 qui avait bloqué ces deux modèles. Aucun N>=1000 n'a été mesuré.
Le biais des golds aux positions 1-20 reste présent.

## Vérification publique hors ligne

```powershell
python -m unittest discover -s tests -v
python validate_profiles_public.py
```

Le validateur recalcule les métriques depuis les CSV, vérifie les exports SQLite,
les préflights tokenizer, les plans/hashes et les comparaisons appariées. Les bruts
restent privés : `review_profiles.py` est le code d'audit historique, qui nécessite
ces captures pour un replay complet. Les bases publiques ne sont pas reprisables.
Le fichier de données minimal produit les mêmes prompts ; les attributions et
licences de chaque skill sont dans [le catalogue du lot parent](../data/ATTRIBUTION.md).
Les sources/licences des poids sont dans [la provenance parent](../evidence/model-artifact-provenance.json).

Les observations GPU archivées sont des preuves historiques assainies. Les couches
transformer/sortie et le KV sont CUDA, avec embeddings d'entrée CPU_Mapped documentés.
Les 14 chargements ont été vérifiés localement contre les préfixes de journaux
scellés et les compteurs CUDA/WDDM ; les journaux complets et identifiants de processus
ne sont pas redistribués. Les limites de ces compteurs demeurent celles du rapport.

Pour refaire des mesures, utiliser un dossier neuf, ses propres poids vérifiés et
de nouvelles preuves de processus. Les chemins source et le hash de la dérivée de
données sont adaptés pour cette arborescence, sans modifier les paramètres mesurés.
`source-measured/` conserve les versions originales du runner et du lanceur.
Llama/Gemma restent non exécutés : voir [CLOSURE.md](CLOSURE.md).
